"""Local preprocessing for FCSR sampling and synthetic training data."""

from __future__ import annotations

import copy
import gzip
import hashlib
import json
import math
import random
import re
import time
from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import Any

import numpy as np

from contract_schema import compute_source_hash, validate_contract
from data_io import stream_jsonl, write_jsonl_atomic
from retrieval import BM25Index, unicode_tokens


_BENCHMARK_ID_FIELDS = (
    "gold_skill_ids",
    "gt_skill_ids",
    "core_gold_skill_ids",
    "core_gt_ids",
    "auxiliary_gold_skill_ids",
    "auxiliary_gt_ids",
    "all_gold_skill_ids",
)
_LANGUAGE_PATTERN = re.compile(r"^[a-z]{2,3}(?:-[a-z0-9]+)*$")


@dataclass(frozen=True)
class SamplingResult:
    records: list[dict[str, Any]]
    skill_ids: list[str]
    stratum_population: dict[str, int]
    stratum_selected: dict[str, int]
    excluded_count: int
    duplicate_count: int


def collect_benchmark_skill_ids(tasks_path: str | Path) -> set[str]:
    skill_ids: set[str] = set()
    for task in stream_jsonl(tasks_path):
        for field in _BENCHMARK_ID_FIELDS:
            values = task.get(field, [])
            if isinstance(values, list):
                skill_ids.update(value for value in values if isinstance(value, str) and value)
        relevance = task.get("relevance", {})
        if isinstance(relevance, dict):
            skill_ids.update(
                value for value in relevance if isinstance(value, str) and value
            )
    return skill_ids


def normalize_category(skill: dict[str, Any]) -> str:
    category = skill.get("category")
    if isinstance(category, str) and category.strip():
        return category.strip().lower()
    skill_id = skill.get("skill_id")
    if isinstance(skill_id, str) and "/" in skill_id:
        prefix = skill_id.split("/", 1)[0].strip().lower()
        if prefix:
            return prefix
    return "other"


def detect_language_group(skill: dict[str, Any]) -> str:
    explicit = skill.get("language")
    if isinstance(explicit, str):
        normalized = explicit.strip().lower().replace("_", "-")
        if _LANGUAGE_PATTERN.fullmatch(normalized):
            return normalized

    text = " ".join(
        value
        for field in ("name", "description", "body")
        if isinstance((value := skill.get(field)), str)
    )[:4000]
    if any("\u3040" <= character <= "\u30ff" for character in text):
        return "kana"
    if any("\uac00" <= character <= "\ud7af" for character in text):
        return "hangul"
    if any("\u4e00" <= character <= "\u9fff" for character in text):
        return "han"
    if any("\u0400" <= character <= "\u04ff" for character in text):
        return "cyrillic"
    if any("\u0600" <= character <= "\u06ff" for character in text):
        return "arabic"
    if any(("A" <= character <= "Z") or ("a" <= character <= "z") for character in text):
        return "latin"
    return "und"


def stratified_sample(
    skills: Iterable[dict[str, Any]],
    excluded_ids: set[str],
    sample_size: int,
    seed: int,
) -> SamplingResult:
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_content: set[str] = set()
    excluded_count = 0
    duplicate_count = 0

    for skill in skills:
        skill_id = skill.get("skill_id")
        if not isinstance(skill_id, str) or not skill_id.strip():
            continue
        if skill_id in excluded_ids:
            excluded_count += 1
            continue
        if not any(
            isinstance(skill.get(field), str) and skill[field].strip()
            for field in ("name", "description", "body")
        ):
            continue
        content_hash = _content_hash(skill)
        if content_hash in seen_content:
            duplicate_count += 1
            continue
        seen_content.add(content_hash)
        stratum = f"{normalize_category(skill)}::{detect_language_group(skill)}"
        groups[stratum].append(skill)

    eligible_count = sum(len(records) for records in groups.values())
    if sample_size > eligible_count:
        raise ValueError(
            f"sample_size {sample_size} exceeds eligible skill count {eligible_count}"
        )

    allocation = _allocate_strata(
        {stratum: len(records) for stratum, records in groups.items()},
        sample_size,
    )
    selected: list[dict[str, Any]] = []
    for stratum in sorted(groups):
        records = sorted(groups[stratum], key=lambda item: item["skill_id"])
        rng = random.Random(_stable_seed(seed, stratum))
        rng.shuffle(records)
        selected.extend(records[: allocation[stratum]])
    random.Random(seed).shuffle(selected)

    return SamplingResult(
        records=selected,
        skill_ids=[record["skill_id"] for record in selected],
        stratum_population={key: len(value) for key, value in sorted(groups.items())},
        stratum_selected={key: allocation[key] for key in sorted(allocation)},
        excluded_count=excluded_count,
        duplicate_count=duplicate_count,
    )


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_sampling_manifest(
    result: SamplingResult,
    sample_size: int,
    seed: int,
    pool_hash: str,
    excluded_ids: set[str],
) -> dict[str, Any]:
    excluded_payload = json.dumps(sorted(excluded_ids), separators=(",", ":"))
    return {
        "schema_version": "sampling_v1",
        "sample_size": sample_size,
        "seed": seed,
        "pool_sha256": pool_hash,
        "excluded_ids_sha256": hashlib.sha256(
            excluded_payload.encode("utf-8")
        ).hexdigest(),
        "selected_skill_ids": result.skill_ids,
        "stratum_population": result.stratum_population,
        "stratum_selected": result.stratum_selected,
        "excluded_count": result.excluded_count,
        "duplicate_count": result.duplicate_count,
    }


def _allocate_strata(populations: dict[str, int], sample_size: int) -> dict[str, int]:
    allocation = {key: 0 for key in populations}
    keys = sorted(key for key, size in populations.items() if size > 0)
    remaining = sample_size

    if remaining >= len(keys):
        for key in keys:
            allocation[key] = 1
        remaining -= len(keys)

    while remaining:
        available = [key for key in keys if allocation[key] < populations[key]]
        weights = {key: math.sqrt(populations[key]) for key in available}
        weight_sum = sum(weights.values())
        ideals = {key: remaining * weights[key] / weight_sum for key in available}
        additions = {
            key: min(populations[key] - allocation[key], math.floor(ideals[key]))
            for key in available
        }
        added = sum(additions.values())
        if added:
            for key, count in additions.items():
                allocation[key] += count
            remaining -= added
            continue

        ranked = sorted(
            available,
            key=lambda key: (-(ideals[key] - math.floor(ideals[key])), key),
        )
        for key in ranked[:remaining]:
            allocation[key] += 1
        remaining = 0
    return allocation


def _content_hash(skill: dict[str, Any]) -> str:
    payload = {
        field: skill.get(field) if isinstance(skill.get(field), str) else ""
        for field in ("name", "description", "body")
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _stable_seed(seed: int, value: str) -> int:
    digest = hashlib.sha256(f"{seed}:{value}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


@dataclass(frozen=True)
class LLMConfig:
    model: str = "deepseek-v4-flash"
    provider: str = "deepseek"
    temperature: float = 0.0
    thinking: str = "disabled"
    max_attempts: int = 3
    backoff_seconds: float = 2.0
    contract_prompt_version: str = "contract_v2_prompt_002"
    query_prompt_version: str = "contract_query_prompt_005"
    limit: int | None = None


@dataclass(frozen=True)
class PipelineSummary:
    attempted: int = 0
    succeeded: int = 0
    skipped: int = 0
    failed: int = 0


def build_contract_messages(
    skill: dict[str, Any],
    validation_error: str | None = None,
) -> list[dict[str, str]]:
    source = {
        "skill_id": skill.get("skill_id", ""),
        "name": skill.get("name", ""),
        "description": str(skill.get("description", ""))[:2000],
        "body": str(skill.get("body", ""))[:12000],
        "category": skill.get("category", ""),
    }
    schema = {
        "source_languages": ["BCP-47 language tag"],
        "canonical_language": "en",
        "capability": {
            "summary": "English capability summary",
            "evidence_quotes": [
                {"source_field": "name|description|body", "quote": "exact quote"}
            ],
        },
        "operations": [
            {
                "action": "English verb",
                "target": "English target",
                "outcome": "English outcome or null",
                "qualifiers": [],
                "evidence_quotes": [
                    {"source_field": "name|description|body", "quote": "exact quote"}
                ],
            }
        ],
        "inputs": [
            {
                "artifact": "English artifact",
                "format": "format or null",
                "required": True,
                "constraints": [],
                "evidence_quotes": [
                    {"source_field": "name|description|body", "quote": "exact quote"}
                ],
            }
        ],
        "outputs": [],
        "preconditions": [
            {
                "statement": "English statement",
                "evidence_quotes": [
                    {"source_field": "name|description|body", "quote": "exact quote"}
                ],
            }
        ],
        "constraints": [],
        "dependencies": [
            {
                "name": "dependency",
                "type": "software|service|hardware|knowledge|data|other",
                "required": True,
                "evidence_quotes": [
                    {"source_field": "name|description|body", "quote": "exact quote"}
                ],
            }
        ],
        "exclusions": [],
        "quality_criteria": [],
    }
    retry_instruction = ""
    if validation_error:
        retry_instruction = (
            "\n\nYour previous response failed deterministic validation:\n"
            f"{validation_error[:1000]}\n"
            "Regenerate the entire JSON object. Correct every invalid citation and "
            "do not repeat the rejected quote."
        )
    return [
        {
            "role": "system",
            "content": (
                "You extract evidence-grounded Skill Contracts. Write all semantic "
                "fields in concise English, preserve source language tags, and return "
                "one JSON object only. Never infer unsupported facts. Every retained "
                "semantic item must cite at least one exact, contiguous quote copied "
                "from name, description, or body. Copy quotes directly from the source "
                "JSON without reconstructing code or removing Markdown markers such "
                "as **, backticks, brackets, punctuation, or whitespace. Prefer short "
                "quotes contained within one source line. Omit unsupported optional "
                "items."
            ),
        },
        {
            "role": "user",
            "content": (
                "Extract the contract using exactly this response shape:\n"
                f"{json.dumps(schema, ensure_ascii=False)}\n\n"
                "Source skill:\n"
                f"{json.dumps(source, ensure_ascii=False)}"
                f"{retry_instruction}"
            ),
        },
    ]

def build_query_messages(
    skill: dict[str, Any],
    contract: dict[str, Any],
    validation_error: str | None = None,
) -> list[dict[str, str]]:
    contract_view = {
        field: copy.deepcopy(contract[field])
        for field in (
            "capability",
            "operations",
            "inputs",
            "outputs",
            "preconditions",
            "constraints",
            "dependencies",
            "exclusions",
            "quality_criteria",
        )
    }
    for value in contract_view.values():
        _remove_evidence_ids(value)
    category = normalize_category(skill)
    retry_instruction = ""
    if validation_error:
        retry_instruction = (
            "\nYour previous response failed deterministic validation:\n"
            f"{validation_error[:1000]}\n"
            "Regenerate the query and correct the stated problem."
        )
    return [
        {
            "role": "system",
            "content": (
                "You create single-positive retrieval training queries grounded in "
                "one supplied Skill Contract. Return one JSON object with a single "
                "'query' field containing a realistic task request."
            ),
        },
        {
            "role": "user",
            "content": (
                "Write one concrete task that can be fulfilled primarily and "
                "completely by the one supplied Skill Contract. The query MUST "
                "contain 80-180 English words.\n"
                "Grounding rules:\n"
                "1. Treat the Contract operations, outputs, constraints, and quality "
                "criteria as a strict allowlist of requested work. Every imperative verb "
                "and requested deliverable in the query must map directly to that "
                "allowlist; inputs and dependencies provide context only.\n"
                "2. You may add neutral scenario context, but it must not introduce a "
                "new required capability. Do not require other specialized capabilities "
                "that would need additional Skills.\n"
                "3. Treat the surrounding business workflow as already existing. "
                "Domain actions outside this Contract may appear only as context or as "
                "work performed by an existing component or agent; they must not become "
                "requested deliverables. Do not ask to build the surrounding application "
                "or implement unrelated business logic. Bad: Build a checkout flow and "
                "add error handling. Good: The checkout flow already exists; refactor "
                "only its error handling.\n"
                "4. Do not invent URLs, file paths, API endpoints, package, library, "
                "framework, model, vendor, dataset size, deadline, threshold, or retry "
                "count unless it is explicitly present in the Contract.\n"
                "5. For orchestration Contracts, assume specialized agents already "
                "exist. Ask only for orchestration, routing, coordination, monitoring, "
                "or synthesis supported by the Contract; do not ask the orchestrator "
                "to implement the agents' domain algorithms.\n"
                "6. Do not mention or paraphrase the source Skill name, mention the "
                "Contract, or list unrelated optional work.\n"
                f"Category: {category}\n"
                f"Contract: {json.dumps(contract_view, ensure_ascii=False)}"
                f"{retry_instruction}"
            ),
        },
    ]


def extract_contracts(
    sample_path: str | Path,
    output_path: str | Path,
    failure_path: str | Path,
    client: Any,
    config: LLMConfig,
    progress: Callable[[PipelineSummary], None] | None = None,
) -> PipelineSummary:
    completed = {
        (record.get("skill_id"), record.get("source_hash"))
        for record in _stream_if_exists(output_path)
    }
    counts = {"attempted": 0, "succeeded": 0, "skipped": 0, "failed": 0}

    for index, skill in enumerate(stream_jsonl(sample_path)):
        if config.limit is not None and index >= config.limit:
            break
        source_hash = compute_source_hash(skill)
        key = (skill.get("skill_id"), source_hash)
        if key in completed:
            counts["skipped"] += 1
            if progress is not None:
                progress(PipelineSummary(**counts))
            continue

        counts["attempted"] += 1
        contract = None
        error: Exception | None = None
        attempts = 0
        validation_error: str | None = None
        for attempts in range(1, config.max_attempts + 1):
            try:
                response = client.complete(
                    messages=build_contract_messages(skill, validation_error),
                    model=config.model,
                    temperature=config.temperature,
                    response_format={"type": "json_object"},
                    extra_body={"thinking": {"type": config.thinking}},
                )
                semantic = _parse_json_object(response)
                contract = _materialize_contract(semantic, skill, config, attempts)
                validate_contract(contract, skill)
                break
            except Exception as exc:
                error = exc
                contract = None
                validation_error = (
                    f"{type(exc).__name__}: {exc}"
                    if isinstance(exc, ValueError)
                    else None
                )
                _retry_sleep(config, attempts)

        if contract is None:
            counts["failed"] += 1
            _append_jsonl(
                failure_path,
                _failure_record("contract", skill, source_hash, attempts, error),
            )
            if progress is not None:
                progress(PipelineSummary(**counts))
            continue

        _append_jsonl(output_path, contract)
        completed.add(key)
        counts["succeeded"] += 1
        if progress is not None:
            progress(PipelineSummary(**counts))

    _prune_resolved_failures(failure_path, "contract", completed)
    return PipelineSummary(**counts)


def generate_queries(
    sample_path: str | Path,
    contracts_path: str | Path,
    output_path: str | Path,
    failure_path: str | Path,
    client: Any,
    config: LLMConfig,
    progress: Callable[[PipelineSummary], None] | None = None,
) -> PipelineSummary:
    contracts = {
        (record.get("skill_id"), record.get("source_hash")): record
        for record in _stream_if_exists(contracts_path)
    }
    completed = {
        (record.get("positive_skill_id"), record.get("source_hash"))
        for record in _stream_if_exists(output_path)
    }
    counts = {"attempted": 0, "succeeded": 0, "skipped": 0, "failed": 0}

    for index, skill in enumerate(stream_jsonl(sample_path)):
        if config.limit is not None and index >= config.limit:
            break
        source_hash = compute_source_hash(skill)
        key = (skill.get("skill_id"), source_hash)
        if key in completed:
            counts["skipped"] += 1
            if progress is not None:
                progress(PipelineSummary(**counts))
            continue

        counts["attempted"] += 1
        contract = contracts.get(key)
        if contract is None:
            counts["failed"] += 1
            _append_jsonl(
                failure_path,
                _failure_record(
                    "query",
                    skill,
                    source_hash,
                    0,
                    ValueError("missing contract for current source hash"),
                ),
            )
            if progress is not None:
                progress(PipelineSummary(**counts))
            continue

        generated_query = None
        error: Exception | None = None
        attempts = 0
        validation_error: str | None = None
        for attempts in range(1, config.max_attempts + 1):
            try:
                response = client.complete(
                    messages=build_query_messages(skill, contract, validation_error),
                    model=config.model,
                    temperature=config.temperature,
                    response_format={"type": "json_object"},
                    extra_body={"thinking": {"type": config.thinking}},
                )
                payload = _parse_json_object(response)
                query = payload.get("query")
                if not isinstance(query, str) or not query.strip():
                    raise ValueError("generated query must be a non-empty string")
                query = query.strip()
                word_count = len(query.split())
                if not 80 <= word_count <= 180:
                    raise ValueError(
                        "generated query must contain 80-180 English words; "
                        f"received {word_count}"
                    )
                if _contains_skill_name(query, str(skill.get("name", ""))):
                    raise ValueError("generated query contains the source skill name")
                generated_query = query
                break
            except Exception as exc:
                error = exc
                generated_query = None
                validation_error = (
                    f"{type(exc).__name__}: {exc}"
                    if isinstance(exc, ValueError)
                    else None
                )
                _retry_sleep(config, attempts)

        if generated_query is None:
            counts["failed"] += 1
            _append_jsonl(
                failure_path,
                _failure_record("query", skill, source_hash, attempts, error),
            )
            if progress is not None:
                progress(PipelineSummary(**counts))
            continue

        _append_jsonl(
            output_path,
            {
                "query_id": f"syn::{skill['skill_id']}",
                "query": generated_query,
                "positive_skill_id": skill["skill_id"],
                "source_hash": source_hash,
                "generator": {
                    "provider": config.provider,
                    "model": config.model,
                    "prompt_version": config.query_prompt_version,
                    "attempts": attempts,
                },
            },
        )
        completed.add(key)
        counts["succeeded"] += 1
        if progress is not None:
            progress(PipelineSummary(**counts))

    _prune_resolved_failures(failure_path, "query", completed)
    return PipelineSummary(**counts)


_EVIDENCE_SOURCE_FIELDS = ("name", "description", "body")
_IGNORABLE_MARKDOWN_CHARS = frozenset({"*", "`"})
_MIN_PROJECTED_QUOTE_LENGTH = 12


def _project_evidence_text(text: str) -> tuple[str, list[int], list[int]]:
    projected: list[str] = []
    starts: list[int] = []
    ends: list[int] = []
    for index, character in enumerate(text):
        if character in _IGNORABLE_MARKDOWN_CHARS:
            continue
        if character.isspace():
            if projected and projected[-1] == " ":
                ends[-1] = index + 1
            else:
                projected.append(" ")
                starts.append(index)
                ends.append(index + 1)
            continue
        projected.append(character)
        starts.append(index)
        ends.append(index + 1)
    return "".join(projected), starts, ends


def _align_evidence_quote(
    skill: dict[str, Any],
    source_field: str,
    quote: str,
    context: str,
) -> tuple[str, str, int, list[str]]:
    ordered_fields = (source_field,) + tuple(
        field for field in _EVIDENCE_SOURCE_FIELDS if field != source_field
    )
    source_texts: dict[str, str] = {}
    for field in ordered_fields:
        source_text = skill.get(field, "")
        if not isinstance(source_text, str):
            if field == source_field:
                raise ValueError(f"skill.{field} must be a string")
            continue
        source_texts[field] = source_text
        start = source_text.find(quote)
        if start >= 0:
            warnings = []
            if field != source_field:
                warnings.append(
                    f"evidence_quote_source_field_corrected:{context}:"
                    f"{source_field}->{field}"
                )
            return field, quote, start, warnings

    projected_quote, _, _ = _project_evidence_text(quote)
    if len(projected_quote.strip()) >= _MIN_PROJECTED_QUOTE_LENGTH:
        for field in ordered_fields:
            source_text = source_texts.get(field)
            if source_text is None:
                continue
            projected_source, starts, ends = _project_evidence_text(source_text)
            projected_start = projected_source.find(projected_quote)
            if projected_start < 0:
                continue
            projected_end = projected_start + len(projected_quote) - 1
            raw_start = starts[projected_start]
            raw_end = ends[projected_end]
            while (
                raw_start > 0
                and source_text[raw_start - 1] in _IGNORABLE_MARKDOWN_CHARS
            ):
                raw_start -= 1
            while (
                raw_end < len(source_text)
                and source_text[raw_end] in _IGNORABLE_MARKDOWN_CHARS
            ):
                raw_end += 1
            warnings = [f"evidence_quote_markdown_aligned:{context}:{field}"]
            if field != source_field:
                warnings.append(
                    f"evidence_quote_source_field_corrected:{context}:"
                    f"{source_field}->{field}"
                )
            return field, source_text[raw_start:raw_end], raw_start, warnings

    raise ValueError(
        f"evidence quote not found in skill.{source_field}: {quote!r}"
    )


def _materialize_contract(
    semantic: dict[str, Any],
    skill: dict[str, Any],
    config: LLMConfig,
    attempts: int,
) -> dict[str, Any]:
    required = {
        "source_languages",
        "canonical_language",
        "capability",
        "operations",
        "inputs",
        "outputs",
        "preconditions",
        "constraints",
        "dependencies",
        "exclusions",
        "quality_criteria",
    }
    missing = required.difference(semantic)
    if missing:
        raise ValueError(f"contract response missing fields: {sorted(missing)}")
    unexpected = set(semantic).difference(required)
    if unexpected:
        raise ValueError(f"contract response has unexpected fields: {sorted(unexpected)}")

    evidence: list[dict[str, Any]] = []
    evidence_lookup: dict[tuple[str, str], str] = {}
    warnings: list[str] = []

    def convert(item: dict[str, Any], context: str) -> dict[str, Any]:
        if not isinstance(item, dict):
            raise ValueError("semantic contract items must be JSON objects")
        result = dict(item)
        quotes = result.pop("evidence_quotes", None)
        if not isinstance(quotes, list) or not quotes:
            raise ValueError("each semantic item must contain evidence_quotes")
        references: list[str] = []
        citation_errors: list[str] = []
        for citation_index, citation in enumerate(quotes):
            if not isinstance(citation, dict):
                citation_errors.append("evidence quote must be a JSON object")
                continue
            source_field = citation.get("source_field")
            quote = citation.get("quote")
            if source_field not in _EVIDENCE_SOURCE_FIELDS:
                citation_errors.append(
                    "evidence source_field must be name, description, or body"
                )
                continue
            if not isinstance(quote, str) or not quote:
                citation_errors.append("evidence quote must be a non-empty string")
                continue
            try:
                aligned_field, aligned_quote, start, alignment_warnings = (
                    _align_evidence_quote(skill, source_field, quote, context)
                )
            except ValueError as exc:
                citation_errors.append(str(exc))
                warnings.append(
                    f"dropped_invalid_evidence_quote:{context}[{citation_index}]"
                )
                continue
            warnings.extend(alignment_warnings)
            key = (aligned_field, aligned_quote)
            evidence_id = evidence_lookup.get(key)
            if evidence_id is None:
                evidence_id = f"ev_{len(evidence) + 1:04d}"
                evidence_lookup[key] = evidence_id
                evidence.append(
                    {
                        "evidence_id": evidence_id,
                        "source_field": aligned_field,
                        "quote": aligned_quote,
                        "start_char": start,
                        "end_char": start + len(aligned_quote),
                    }
                )
            if evidence_id not in references:
                references.append(evidence_id)
        if not references:
            detail = citation_errors[0] if citation_errors else "no valid evidence"
            raise ValueError(f"{context} has no valid evidence: {detail}")
        result["evidence_ids"] = references
        return result

    capability = convert(semantic["capability"], "capability")
    collection_fields = (
        "operations",
        "inputs",
        "outputs",
        "preconditions",
        "constraints",
        "dependencies",
        "exclusions",
        "quality_criteria",
    )
    converted: dict[str, list[dict[str, Any]]] = {}
    for field in collection_fields:
        items = semantic[field]
        if not isinstance(items, list):
            raise ValueError(f"{field} must be a list")
        converted[field] = []
        for index, item in enumerate(items):
            context = f"{field}[{index}]"
            try:
                converted[field].append(convert(item, context))
            except ValueError:
                warnings.append(f"dropped_unsupported_item:{context}")

    if len(str(skill.get("description", ""))) > 2000:
        warnings.append("source_description_truncated")
    if len(str(skill.get("body", ""))) > 12000:
        warnings.append("source_body_truncated")

    return {
        "schema_version": "contract_v2",
        "skill_id": skill["skill_id"],
        "source_hash": compute_source_hash(skill),
        "source_languages": semantic["source_languages"],
        "canonical_language": semantic["canonical_language"],
        "capability": capability,
        **converted,
        "evidence": evidence,
        "extraction": {
            "method": "llm",
            "provider": config.provider,
            "model": config.model,
            "prompt_version": config.contract_prompt_version,
            "temperature": config.temperature,
            "status": "validated",
            "attempts": attempts,
            "warnings": list(dict.fromkeys(warnings)),
        },
    }

def _parse_json_object(response: Any) -> dict[str, Any]:
    if not isinstance(response, str):
        raise ValueError("LLM response must be a string")
    text = response.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            text = "\n".join(lines[1:-1])
            if text.lstrip().startswith("json"):
                text = text.lstrip()[4:].lstrip()
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("LLM response must be one JSON object")
    return payload


def _stream_if_exists(path: str | Path) -> Iterable[dict[str, Any]]:
    candidate = Path(path)
    if not candidate.exists():
        return ()
    return stream_jsonl(candidate)


def _append_jsonl(path: str | Path, record: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if destination.name.endswith(".gz") else open
    with opener(destination, "at", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
        handle.write("\n")
        handle.flush()


def _failure_record(
    stage: str,
    skill: dict[str, Any],
    source_hash: str,
    attempts: int,
    error: Exception | None,
) -> dict[str, Any]:
    return {
        "stage": stage,
        "skill_id": skill.get("skill_id"),
        "source_hash": source_hash,
        "attempts": attempts,
        "error_type": type(error).__name__ if error is not None else "UnknownError",
        "error": str(error) if error is not None else "unknown error",
    }


def _prune_resolved_failures(
    failure_path: str | Path,
    stage: str,
    resolved_keys: set[tuple[Any, Any]],
) -> int:
    candidate = Path(failure_path)
    if not candidate.exists() or not resolved_keys:
        return 0
    records = list(stream_jsonl(candidate))
    kept = [
        record
        for record in records
        if not (
            record.get("stage") == stage
            and (record.get("skill_id"), record.get("source_hash"))
            in resolved_keys
        )
    ]
    removed = len(records) - len(kept)
    if removed:
        write_jsonl_atomic(candidate, kept)
    return removed

def _retry_sleep(config: LLMConfig, attempts: int) -> None:
    if attempts < config.max_attempts and config.backoff_seconds > 0:
        time.sleep(min(config.backoff_seconds * (2 ** (attempts - 1)), 8.0))


def _contains_skill_name(query: str, skill_name: str) -> bool:
    tokens = re.findall(r"\w+", skill_name.casefold(), flags=re.UNICODE)
    if not tokens:
        return False
    pattern = r"(?<!\w)" + r"[\W_]+".join(map(re.escape, tokens)) + r"(?!\w)"
    return re.search(pattern, query.casefold(), flags=re.UNICODE) is not None


def _remove_evidence_ids(value: Any) -> None:
    if isinstance(value, dict):
        value.pop("evidence_ids", None)
        for child in value.values():
            _remove_evidence_ids(child)
    elif isinstance(value, list):
        for child in value:
            _remove_evidence_ids(child)

@dataclass(frozen=True)
class FilterResult:
    kept: list[dict[str, Any]]
    removed: list[dict[str, str]]


def filter_identity_and_overlap(
    positive: dict[str, Any],
    candidates: Iterable[dict[str, Any]],
    threshold: float = 0.85,
) -> FilterResult:
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")
    kept: list[dict[str, Any]] = []
    removed: list[dict[str, str]] = []
    positive_id = positive.get("skill_id")
    positive_name = _normalize_text(str(positive.get("name", "")))
    positive_body = str(positive.get("body", "")).strip()
    positive_trigrams = _character_trigrams(positive_body)

    for candidate in candidates:
        candidate_id = str(candidate.get("skill_id", ""))
        reason = None
        if candidate_id == positive_id:
            reason = "same_skill_id"
        elif (
            positive_name
            and _normalize_text(str(candidate.get("name", ""))) == positive_name
        ):
            reason = "same_normalized_name"
        else:
            candidate_body = str(candidate.get("body", "")).strip()
            if positive_body and candidate_body == positive_body:
                reason = "same_body"
            elif positive_trigrams and candidate_body:
                overlap = _jaccard(
                    positive_trigrams,
                    _character_trigrams(candidate_body),
                )
                if overlap >= threshold:
                    reason = "high_body_trigram_overlap"
        if reason is None:
            kept.append(candidate)
        else:
            removed.append({"skill_id": candidate_id, "reason": reason})
    return FilterResult(kept=kept, removed=removed)


def mine_local_negatives(
    queries: Iterable[dict[str, Any]],
    pool: Iterable[dict[str, Any]],
    seed: int,
    overlap_threshold: float = 0.85,
    stage: Callable[[str], None] | None = None,
    progress: Callable[[int], None] | None = None,
) -> Iterable[dict[str, Any]]:
    if stage is not None:
        stage("loading_skills")
    pool_records = [
        record
        for record in pool
        if isinstance(record.get("skill_id"), str) and record["skill_id"]
    ]
    by_id = {record["skill_id"]: record for record in pool_records}
    if stage is not None:
        stage("building_bm25")
    documents = [_skill_search_text(record) for record in pool_records]
    bm25 = BM25Index(documents)
    stable_pool_order = sorted(
        range(len(pool_records)),
        key=lambda index: pool_records[index]["skill_id"],
    )
    category_members: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in pool_records:
        category_members[normalize_category(record)].append(record)

    if stage is not None:
        stage("mining_queries")
    for processed, query in enumerate(queries, start=1):
        positive_id = query.get("positive_skill_id")
        positive = by_id.get(positive_id)
        if positive is None:
            raise ValueError(f"positive skill not found in pool: {positive_id!r}")

        scores, bm25_head = bm25.rank(str(query.get("query", "")), limit=256)
        head_set = set(bm25_head)
        bm25_order = chain(
            bm25_head,
            (index for index in stable_pool_order if index not in head_set),
        )
        same_category = list(category_members[normalize_category(positive)])
        random.Random(_stable_seed(seed, f"{query.get('query_id')}::category")).shuffle(
            same_category
        )
        random_order = _lazy_random_order(
            pool_records,
            _stable_seed(seed, f"{query.get('query_id')}::random"),
        )

        selected: list[dict[str, Any]] = []
        selected_ids: set[str] = set()
        removed_ids: set[str] = set()
        filtered: list[dict[str, str]] = []

        def add_candidates(
            records: Iterable[tuple[dict[str, Any], float]],
            source: str,
            target: int,
        ) -> None:
            source_count = 0
            for candidate, score in records:
                candidate_id = candidate["skill_id"]
                if candidate_id in selected_ids or candidate_id in removed_ids:
                    continue
                result = filter_identity_and_overlap(
                    positive,
                    [candidate],
                    threshold=overlap_threshold,
                )
                if result.removed:
                    removed_ids.add(candidate_id)
                    filtered.extend(result.removed)
                    continue
                selected.append(
                    {
                        "skill_id": candidate_id,
                        "source": source,
                        "score": float(score),
                    }
                )
                selected_ids.add(candidate_id)
                source_count += 1
                if source_count >= target:
                    break

        add_candidates(
            ((pool_records[index], scores[index]) for index in bm25_order),
            "bm25",
            3,
        )
        add_candidates(((record, 0.0) for record in same_category), "same_category", 2)
        add_candidates(((record, 0.0) for record in random_order), "random", 1)

        result = {
            "query_id": query.get("query_id"),
            "query": query.get("query"),
            "positive_skill_id": positive_id,
            "source_hash": query.get("source_hash"),
            "negative_candidates": selected,
            "filtered": filtered,
        }
        if progress is not None:
            progress(processed)
        yield result


def _lazy_random_order(
    records: list[dict[str, Any]],
    seed: int,
) -> Iterable[dict[str, Any]]:
    rng = random.Random(seed)
    visited: set[int] = set()
    while len(visited) < len(records):
        index = rng.randrange(len(records))
        if index not in visited:
            visited.add(index)
            yield records[index]


def _skill_search_text(skill: dict[str, Any]) -> str:
    return "\n".join(
        str(skill.get(field, ""))
        for field in ("name", "description", "body")
    )

def _normalize_text(text: str) -> str:
    return " ".join(unicode_tokens(text))


def _character_trigrams(text: str) -> set[str]:
    normalized = re.sub(r"\s+", " ", text.casefold()).strip()
    if not normalized:
        return set()
    if len(normalized) < 3:
        return {normalized}
    return {
        normalized[index : index + 3]
        for index in range(len(normalized) - 2)
    }


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0

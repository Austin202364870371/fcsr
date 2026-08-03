"""Deterministic Contract-guided candidate construction for multi-Skill tasks."""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Iterable

from retrieval import unicode_tokens


_ARTIFACT_STOPWORDS = frozenset({
    "analysis", "and", "are", "as", "be", "by", "code", "configuration",
    "configurations", "content", "data", "details", "document", "documents",
    "documentation", "error", "errors", "file", "files", "for", "from", "in",
    "information", "input", "inputs", "into", "is", "item", "items", "list",
    "lists", "message", "messages", "object", "objects", "of", "on", "or",
    "output", "outputs", "plan", "plans", "project", "projects", "report", "reports",
    "request", "requests", "response", "responses", "result", "results", "system",
    "task", "tasks", "test", "testing", "tests", "text", "the", "to", "user",
    "users", "value", "values", "with",
})
_PRODUCER_ACTIONS = frozenset({
    "build", "collect", "compile", "convert", "create", "derive", "extract", "fetch",
    "generate", "implement", "produce", "transform", "write",
})
_CONSUMER_ACTIONS = frozenset({
    "analyze", "analyse", "audit", "compare", "deploy", "evaluate", "integrate",
    "publish", "render", "review", "summarize", "test", "validate", "verify",
    "visualize",
})


@dataclass(frozen=True)
class CandidateSettings:
    max_pairs: int = 7342
    max_triples: int = 1000
    max_pairs_per_source: int = 16
    max_artifact_frequency: int = 5


@dataclass(frozen=True)
class CandidateResult:
    eligible_skill_ids: list[str]
    pairs: list[dict[str, Any]]
    triples: list[dict[str, Any]]
    rejections: list[dict[str, Any]]

    @property
    def candidates(self) -> list[dict[str, Any]]:
        return [*self.pairs, *self.triples]


def build_compositional_candidates(
    contracts: Iterable[dict[str, Any]],
    single_skill_queries: Iterable[dict[str, Any]],
    benchmark_skill_ids: set[str],
    settings: CandidateSettings | None = None,
) -> CandidateResult:
    """Build ordered pair/triple candidates from validated Contract handoffs."""
    options = settings or CandidateSettings()
    _validate_settings(options)
    query_hashes = _query_hashes(single_skill_queries)
    eligible: dict[str, dict[str, Any]] = {}
    rejections: list[dict[str, Any]] = []

    for contract in sorted(contracts, key=lambda item: str(item.get("skill_id", ""))):
        skill_id = contract.get("skill_id")
        if not isinstance(skill_id, str) or not skill_id:
            continue
        if contract.get("extraction", {}).get("status") != "validated":
            rejections.append(_eligibility_rejection(skill_id, "contract_not_validated"))
            continue
        if skill_id in benchmark_skill_ids:
            rejections.append(_eligibility_rejection(skill_id, "benchmark_skill"))
            continue
        source_hash = contract.get("source_hash")
        if not isinstance(source_hash, str) or source_hash not in query_hashes.get(skill_id, set()):
            rejections.append(_eligibility_rejection(skill_id, "missing_or_stale_single_skill_query"))
            continue
        if skill_id in eligible:
            rejections.append(_eligibility_rejection(skill_id, "duplicate_contract"))
            continue
        eligible[skill_id] = contract

    features = {skill_id: _features(contract) for skill_id, contract in eligible.items()}
    output_frequency = Counter(
        token for feature in features.values() for token in feature.output_tokens
    )
    input_index: dict[str, set[str]] = defaultdict(set)
    for skill_id, feature in features.items():
        for token in feature.required_input_tokens:
            if output_frequency[token] <= options.max_artifact_frequency:
                input_index[token].add(skill_id)

    retained_edges: list[dict[str, Any]] = []
    for source_id in sorted(features):
        source = features[source_id]
        source_edges = []
        for target_id in _handoff_targets(source, input_index):
            target = features[target_id]
            edge_or_rejection = _build_edge(source, target, output_frequency)
            if "reasons" in edge_or_rejection:
                rejections.append(edge_or_rejection)
            else:
                source_edges.append(edge_or_rejection)
        source_edges.sort(key=lambda item: (-item["score"], item["to_skill_id"]))
        retained_edges.extend(source_edges[: options.max_pairs_per_source])
        for edge in source_edges[options.max_pairs_per_source :]:
            rejections.append(
                _rejection(
                    [edge["from_skill_id"], edge["to_skill_id"]],
                    "pair_selection",
                    ["source_pair_limit"],
                )
            )

    retained_edges.sort(key=lambda item: (-item["score"], item["from_skill_id"], item["to_skill_id"]))
    pair_edges = retained_edges[: options.max_pairs]
    for edge in retained_edges[options.max_pairs :]:
        rejections.append(
            _rejection(
                [edge["from_skill_id"], edge["to_skill_id"]],
                "pair_selection",
                ["global_pair_limit"],
            )
        )
    pairs = [_pair_candidate(edge) for edge in pair_edges]
    triples = _triple_candidates(retained_edges, options.max_triples, rejections)
    rejections.sort(key=lambda item: (item["stage"], item["skill_ids"], item["reasons"]))
    return CandidateResult(
        eligible_skill_ids=sorted(eligible),
        pairs=pairs,
        triples=triples,
        rejections=rejections,
    )


@dataclass(frozen=True)
class _Features:
    skill_id: str
    output_artifacts: tuple[frozenset[str], ...]
    required_input_artifacts: tuple[frozenset[str], ...]
    output_tokens: frozenset[str]
    required_input_tokens: frozenset[str]
    output_formats: frozenset[str]
    input_formats: frozenset[str]
    dependency_tokens: frozenset[str]
    action_tokens: frozenset[str]


def _features(contract: dict[str, Any]) -> _Features:
    output_artifacts = _artifact_token_sets(contract.get("outputs", []))
    required_input_artifacts = _artifact_token_sets(
        item for item in contract.get("inputs", []) if item.get("required") is True
    )
    return _Features(
        skill_id=contract["skill_id"],
        output_artifacts=output_artifacts,
        required_input_artifacts=required_input_artifacts,
        output_tokens=_flatten_token_sets(output_artifacts),
        required_input_tokens=_flatten_token_sets(required_input_artifacts),
        output_formats=_formats(contract.get("outputs", [])),
        input_formats=_formats(
            item for item in contract.get("inputs", []) if item.get("required") is True
        ),
        dependency_tokens=_named_tokens(contract.get("dependencies", []), "name"),
        action_tokens=_named_tokens(contract.get("operations", []), "action"),
    )


def _artifact_token_sets(
    items: Iterable[dict[str, Any]],
) -> tuple[frozenset[str], ...]:
    artifacts = []
    for item in items:
        if not isinstance(item, dict):
            continue
        tokens = frozenset(
            token
            for token in unicode_tokens(str(item.get("artifact", "")))
            if len(token) >= 3 and token not in _ARTIFACT_STOPWORDS
        )
        if tokens:
            artifacts.append(tokens)
    return tuple(artifacts)


def _flatten_token_sets(items: Iterable[frozenset[str]]) -> frozenset[str]:
    return frozenset(token for item in items for token in item)


def _formats(items: Iterable[dict[str, Any]]) -> frozenset[str]:
    return frozenset(
        value.strip().casefold()
        for item in items
        if isinstance(item, dict)
        and isinstance((value := item.get("format")), str)
        and value.strip()
    )


def _named_tokens(items: Iterable[dict[str, Any]], field: str) -> frozenset[str]:
    return frozenset(
        token
        for item in items
        if isinstance(item, dict)
        for token in unicode_tokens(str(item.get(field, "")))
        if len(token) >= 3
    )


def _handoff_targets(source: _Features, output_index: dict[str, set[str]]) -> list[str]:
    targets = {
        skill_id
        for token in source.output_tokens
        for skill_id in output_index.get(token, set())
        if skill_id != source.skill_id
    }
    return sorted(targets)


def _build_edge(
    source: _Features,
    target: _Features,
    output_frequency: Counter[str],
) -> dict[str, Any]:
    raw_overlap = source.output_tokens & target.required_input_tokens
    if not raw_overlap:
        return _rejection(
            [source.skill_id, target.skill_id],
            "pair_validation",
            ["missing_artifact_handoff"],
        )
    handoffs = _qualified_artifact_handoffs(source, target)
    if not handoffs:
        return _rejection(
            [source.skill_id, target.skill_id],
            "pair_validation",
            ["weak_artifact_handoff"],
        )
    matched_tokens = sorted({token for handoff in handoffs for token in handoff})
    relation = _operation_relation(source.action_tokens, target.action_tokens)
    if relation is None:
        return _rejection(
            [source.skill_id, target.skill_id],
            "pair_validation",
            ["missing_complementary_operation"],
        )
    matched_formats = sorted(source.output_formats & target.input_formats)
    dependency_tokens = sorted(source.output_tokens & target.dependency_tokens)
    rarity_score = sum(1 / output_frequency[token] for token in matched_tokens)
    score = round(4 + rarity_score + len(matched_formats) + len(dependency_tokens) + 2, 6)
    return {
        "from_skill_id": source.skill_id,
        "to_skill_id": target.skill_id,
        "matched_artifact_tokens": matched_tokens,
        "matched_formats": matched_formats,
        "matched_dependency_tokens": dependency_tokens,
        "operation_relation": relation,
        "score": score,
    }


def _qualified_artifact_handoffs(
    source: _Features,
    target: _Features,
) -> tuple[frozenset[str], ...]:
    matches = []
    for output in source.output_artifacts:
        for required_input in target.required_input_artifacts:
            overlap = output & required_input
            if not overlap:
                continue
            union = output | required_input
            if output == required_input or (
                len(overlap) >= 2 and len(overlap) / len(union) >= 0.5
            ):
                matches.append(overlap)
    return tuple(matches)

def _operation_relation(
    source_actions: frozenset[str], target_actions: frozenset[str]
) -> str | None:
    if not target_actions & _CONSUMER_ACTIONS:
        return None
    if source_actions & _PRODUCER_ACTIONS:
        return "producer_to_consumer"
    return "artifact_handoff_to_consumer"


def _pair_candidate(edge: dict[str, Any]) -> dict[str, Any]:
    skill_ids = [edge["from_skill_id"], edge["to_skill_id"]]
    return {
        "candidate_id": _candidate_id("pair", skill_ids),
        "candidate_type": "pair",
        "skill_ids": skill_ids,
        "score": edge["score"],
        "edges": [edge],
    }


def _triple_candidates(
    edges: list[dict[str, Any]],
    max_triples: int,
    rejections: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    outgoing: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for edge in edges:
        outgoing[edge["from_skill_id"]].append(edge)

    triples = []
    for first in edges:
        for second in outgoing[first["to_skill_id"]]:
            skill_ids = [first["from_skill_id"], first["to_skill_id"], second["to_skill_id"]]
            if len(set(skill_ids)) != 3:
                continue
            triples.append(
                {
                    "candidate_id": _candidate_id("triple", skill_ids),
                    "candidate_type": "triple",
                    "skill_ids": skill_ids,
                    "score": round(first["score"] + second["score"], 6),
                    "edges": [first, second],
                }
            )
    triples.sort(key=lambda item: (-item["score"], item["skill_ids"]))
    selected = triples[:max_triples]
    for triple in triples[max_triples:]:
        rejections.append(_rejection(triple["skill_ids"], "triple_selection", ["global_triple_limit"]))
    return selected


def _query_hashes(queries: Iterable[dict[str, Any]]) -> dict[str, set[str]]:
    hashes: dict[str, set[str]] = defaultdict(set)
    for query in queries:
        skill_id = query.get("positive_skill_id")
        source_hash = query.get("source_hash")
        if isinstance(skill_id, str) and skill_id and isinstance(source_hash, str) and source_hash:
            hashes[skill_id].add(source_hash)
    return hashes


def _eligibility_rejection(skill_id: str, reason: str) -> dict[str, Any]:
    return _rejection([skill_id], "eligibility", [reason])


def _rejection(skill_ids: list[str], stage: str, reasons: list[str]) -> dict[str, Any]:
    return {
        "candidate_id": _candidate_id(stage, skill_ids),
        "skill_ids": skill_ids,
        "stage": stage,
        "reasons": reasons,
    }


def _candidate_id(kind: str, skill_ids: list[str]) -> str:
    payload = f"{kind}\0" + "\0".join(skill_ids)
    return f"comp::{kind}::{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:16]}"


def _validate_settings(settings: CandidateSettings) -> None:
    for field in ("max_pairs", "max_triples", "max_pairs_per_source", "max_artifact_frequency"):
        if getattr(settings, field) <= 0:
            raise ValueError(f"{field} must be positive")
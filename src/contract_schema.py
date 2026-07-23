"""Contract V2 schema and validation for LLM-extracted Skill Contracts."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from typing import Any


SCHEMA_VERSION = "contract_v2"
SOURCE_FIELDS = ("name", "description", "body")
DEPENDENCY_TYPES = {"software", "service", "hardware", "knowledge", "data", "other"}
EXTRACTION_STATUSES = {"validated", "needs_review"}


class ContractValidationError(ValueError):
    """Raised when a Contract V2 record is malformed or lacks source support."""


_TOP_LEVEL_FIELDS = {
    "schema_version",
    "skill_id",
    "source_hash",
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
    "evidence",
    "extraction",
}
_LANGUAGE_PATTERN = re.compile(r"^(?:[a-z]{2,3}|und)(?:-[A-Za-z0-9]{2,8})*$")
_HASH_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def compute_source_hash(skill: Mapping[str, Any]) -> str:
    """Hash the source fields that determine a contract's semantic content."""
    source = {
        "skill_id": _skill_string(skill, "skill_id", required=True),
        "name": _skill_string(skill, "name"),
        "description": _skill_string(skill, "description"),
        "body": _skill_string(skill, "body"),
    }
    payload = json.dumps(
        source,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_contract(
    contract: dict[str, Any], skill: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate Contract V2 structure, provenance, and evidence references."""
    if not isinstance(contract, dict):
        raise ContractValidationError("contract must be a JSON object")
    if not isinstance(skill, Mapping):
        raise ContractValidationError("skill must be a mapping")

    _require_exact_fields("contract", contract, _TOP_LEVEL_FIELDS)
    if contract["schema_version"] != SCHEMA_VERSION:
        raise ContractValidationError(f"schema_version must be {SCHEMA_VERSION}")

    skill_id = _skill_string(skill, "skill_id", required=True)
    if contract["skill_id"] != skill_id:
        raise ContractValidationError("contract.skill_id does not match source skill")

    source_hash = contract["source_hash"]
    if not isinstance(source_hash, str) or not _HASH_PATTERN.fullmatch(source_hash):
        raise ContractValidationError("source_hash must be a lowercase SHA-256 digest")
    if source_hash != compute_source_hash(skill):
        raise ContractValidationError("source_hash does not match source skill")

    _validate_languages("source_languages", contract["source_languages"], allow_many=True)
    _validate_languages(
        "canonical_language", [contract["canonical_language"]], allow_many=False
    )

    evidence_ids = _validate_evidence(contract["evidence"], skill)
    referenced_ids: set[str] = set()

    _validate_capability(contract["capability"], evidence_ids, referenced_ids)
    _validate_operations(contract["operations"], evidence_ids, referenced_ids)
    _validate_artifacts("inputs", contract["inputs"], evidence_ids, referenced_ids)
    _validate_artifacts("outputs", contract["outputs"], evidence_ids, referenced_ids)
    for field in ("preconditions", "constraints", "exclusions", "quality_criteria"):
        _validate_statements(field, contract[field], evidence_ids, referenced_ids)
    _validate_dependencies(contract["dependencies"], evidence_ids, referenced_ids)
    _validate_extraction(contract["extraction"])

    unused = evidence_ids.difference(referenced_ids)
    if unused:
        raise ContractValidationError(f"unreferenced evidence ids: {sorted(unused)}")
    return contract


def _validate_capability(
    capability: Any, evidence_ids: set[str], referenced_ids: set[str]
) -> None:
    _require_object("capability", capability)
    _require_exact_fields("capability", capability, {"summary", "evidence_ids"})
    _require_non_empty_string("capability.summary", capability["summary"])
    _validate_evidence_references(
        "capability.evidence_ids",
        capability["evidence_ids"],
        evidence_ids,
        referenced_ids,
    )


def _validate_operations(
    operations: Any, evidence_ids: set[str], referenced_ids: set[str]
) -> None:
    _require_list("operations", operations)
    required = {"action", "target", "outcome", "qualifiers", "evidence_ids"}
    for index, operation in enumerate(operations):
        field = f"operations[{index}]"
        _require_object(field, operation)
        _require_exact_fields(field, operation, required)
        _require_non_empty_string(f"{field}.action", operation["action"])
        _require_non_empty_string(f"{field}.target", operation["target"])
        _validate_optional_string(f"{field}.outcome", operation["outcome"])
        _validate_string_list(f"{field}.qualifiers", operation["qualifiers"])
        _validate_evidence_references(
            f"{field}.evidence_ids",
            operation["evidence_ids"],
            evidence_ids,
            referenced_ids,
        )


def _validate_artifacts(
    field: str,
    artifacts: Any,
    evidence_ids: set[str],
    referenced_ids: set[str],
) -> None:
    _require_list(field, artifacts)
    required = {"artifact", "format", "required", "constraints", "evidence_ids"}
    for index, artifact in enumerate(artifacts):
        item = f"{field}[{index}]"
        _require_object(item, artifact)
        _require_exact_fields(item, artifact, required)
        _require_non_empty_string(f"{item}.artifact", artifact["artifact"])
        _validate_optional_string(f"{item}.format", artifact["format"])
        if artifact["required"] is not None and not isinstance(artifact["required"], bool):
            raise ContractValidationError(f"{item}.required must be boolean or null")
        _validate_string_list(f"{item}.constraints", artifact["constraints"])
        _validate_evidence_references(
            f"{item}.evidence_ids",
            artifact["evidence_ids"],
            evidence_ids,
            referenced_ids,
        )


def _validate_statements(
    field: str,
    statements: Any,
    evidence_ids: set[str],
    referenced_ids: set[str],
) -> None:
    _require_list(field, statements)
    for index, statement in enumerate(statements):
        item = f"{field}[{index}]"
        _require_object(item, statement)
        _require_exact_fields(item, statement, {"statement", "evidence_ids"})
        _require_non_empty_string(f"{item}.statement", statement["statement"])
        _validate_evidence_references(
            f"{item}.evidence_ids",
            statement["evidence_ids"],
            evidence_ids,
            referenced_ids,
        )


def _validate_dependencies(
    dependencies: Any, evidence_ids: set[str], referenced_ids: set[str]
) -> None:
    _require_list("dependencies", dependencies)
    required = {"name", "type", "required", "evidence_ids"}
    for index, dependency in enumerate(dependencies):
        item = f"dependencies[{index}]"
        _require_object(item, dependency)
        _require_exact_fields(item, dependency, required)
        _require_non_empty_string(f"{item}.name", dependency["name"])
        if dependency["type"] not in DEPENDENCY_TYPES:
            raise ContractValidationError(
                f"{item}.type must be one of {sorted(DEPENDENCY_TYPES)}"
            )
        if dependency["required"] is not None and not isinstance(
            dependency["required"], bool
        ):
            raise ContractValidationError(f"{item}.required must be boolean or null")
        _validate_evidence_references(
            f"{item}.evidence_ids",
            dependency["evidence_ids"],
            evidence_ids,
            referenced_ids,
        )


def _validate_evidence(
    evidence: Any, skill: Mapping[str, Any]
) -> set[str]:
    _require_list("evidence", evidence)
    required = {"evidence_id", "source_field", "quote", "start_char", "end_char"}
    evidence_ids: set[str] = set()

    for index, item in enumerate(evidence):
        field = f"evidence[{index}]"
        _require_object(field, item)
        _require_exact_fields(field, item, required)
        evidence_id = item["evidence_id"]
        _require_non_empty_string(f"{field}.evidence_id", evidence_id)
        if evidence_id in evidence_ids:
            raise ContractValidationError(f"duplicate evidence id: {evidence_id}")
        evidence_ids.add(evidence_id)

        source_field = item["source_field"]
        if source_field not in SOURCE_FIELDS:
            raise ContractValidationError(
                f"{field}.source_field must be one of {list(SOURCE_FIELDS)}"
            )
        quote = item["quote"]
        _require_non_empty_string(f"{field}.quote", quote)
        start = item["start_char"]
        end = item["end_char"]
        if (
            isinstance(start, bool)
            or isinstance(end, bool)
            or not isinstance(start, int)
            or not isinstance(end, int)
            or start < 0
            or end <= start
        ):
            raise ContractValidationError(f"{field} has invalid character offsets")
        source_text = _skill_string(skill, source_field)
        if end > len(source_text) or source_text[start:end] != quote:
            raise ContractValidationError(
                f"{field} quote does not match source text at the declared offset"
            )
    return evidence_ids


def _validate_evidence_references(
    field: str,
    references: Any,
    evidence_ids: set[str],
    referenced_ids: set[str],
) -> None:
    _validate_string_list(field, references, require_non_empty=True)
    duplicates = {item for item in references if references.count(item) > 1}
    if duplicates:
        raise ContractValidationError(
            f"{field} contains duplicate ids: {sorted(duplicates)}"
        )
    for evidence_id in references:
        if evidence_id not in evidence_ids:
            raise ContractValidationError(
                f"{field} references unknown evidence id {evidence_id!r}"
            )
        referenced_ids.add(evidence_id)


def _validate_extraction(extraction: Any) -> None:
    _require_object("extraction", extraction)
    required = {
        "method",
        "provider",
        "model",
        "prompt_version",
        "temperature",
        "status",
        "attempts",
        "warnings",
    }
    _require_exact_fields("extraction", extraction, required)
    if extraction["method"] != "llm":
        raise ContractValidationError("extraction.method must be llm")
    for field in ("provider", "model", "prompt_version"):
        _require_non_empty_string(f"extraction.{field}", extraction[field])
    temperature = extraction["temperature"]
    if (
        isinstance(temperature, bool)
        or not isinstance(temperature, (int, float))
        or not 0 <= temperature <= 2
    ):
        raise ContractValidationError("extraction.temperature must be between 0 and 2")
    if extraction["status"] not in EXTRACTION_STATUSES:
        raise ContractValidationError(
            f"extraction.status must be one of {sorted(EXTRACTION_STATUSES)}"
        )
    attempts = extraction["attempts"]
    if isinstance(attempts, bool) or not isinstance(attempts, int) or attempts < 1:
        raise ContractValidationError("extraction.attempts must be a positive integer")
    _validate_string_list("extraction.warnings", extraction["warnings"])


def _validate_languages(field: str, languages: Any, allow_many: bool) -> None:
    _validate_string_list(field, languages, require_non_empty=True)
    if not allow_many and len(languages) != 1:
        raise ContractValidationError(f"{field} must contain exactly one language")
    if len(set(languages)) != len(languages):
        raise ContractValidationError(f"{field} must not contain duplicates")
    invalid = [language for language in languages if not _LANGUAGE_PATTERN.fullmatch(language)]
    if invalid:
        raise ContractValidationError(f"{field} contains invalid language tags: {invalid}")


def _require_exact_fields(field: str, value: Mapping[str, Any], expected: set[str]) -> None:
    missing = expected.difference(value)
    if missing:
        raise ContractValidationError(f"{field} missing fields: {sorted(missing)}")
    unexpected = set(value).difference(expected)
    if unexpected:
        raise ContractValidationError(f"{field} has unexpected fields: {sorted(unexpected)}")


def _require_object(field: str, value: Any) -> None:
    if not isinstance(value, dict):
        raise ContractValidationError(f"{field} must be an object")


def _require_list(field: str, value: Any) -> None:
    if not isinstance(value, list):
        raise ContractValidationError(f"{field} must be a list")


def _require_non_empty_string(field: str, value: Any) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ContractValidationError(f"{field} must be a non-empty string")


def _validate_optional_string(field: str, value: Any) -> None:
    if value is not None:
        _require_non_empty_string(field, value)


def _validate_string_list(
    field: str, value: Any, require_non_empty: bool = False
) -> None:
    _require_list(field, value)
    if require_non_empty and not value:
        raise ContractValidationError(f"{field} must not be empty")
    for index, item in enumerate(value):
        _require_non_empty_string(f"{field}[{index}]", item)


def _skill_string(
    skill: Mapping[str, Any], field: str, required: bool = False
) -> str:
    value = skill.get(field, "")
    if value is None:
        value = ""
    if not isinstance(value, str):
        raise ContractValidationError(f"skill.{field} must be a string")
    if required and not value.strip():
        raise ContractValidationError(f"skill.{field} must be a non-empty string")
    return value

"""Adapters from FCSR ranking records to executable agent candidates."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from agent.models import SkillCandidate


class CandidateAdapterError(ValueError):
    """Raised when a ranking record cannot form an executable candidate list."""


def adapt_ranked_candidates(
    record: Mapping[str, Any],
    skill_index: Mapping[str, Mapping[str, Any]],
    limit: int = 20,
) -> list[SkillCandidate]:
    """Convert an ordered FCSR ranking into validated executable candidates."""
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise CandidateAdapterError("limit must be a positive integer")

    ranked_skill_ids = record.get("ranked_skill_ids")
    if not isinstance(ranked_skill_ids, Sequence) or isinstance(
        ranked_skill_ids, (str, bytes)
    ):
        raise CandidateAdapterError("ranked_skill_ids must be a sequence")

    selected_ids = list(ranked_skill_ids[:limit])
    if any(not isinstance(skill_id, str) or not skill_id for skill_id in selected_ids):
        raise CandidateAdapterError("ranked_skill_ids must contain non-empty strings")
    if len(selected_ids) != len(set(selected_ids)):
        raise CandidateAdapterError("ranked_skill_ids contains duplicate skill ids")

    candidates: list[SkillCandidate] = []
    for rank, skill_id in enumerate(selected_ids, start=1):
        source = skill_index.get(skill_id)
        if source is None:
            raise CandidateAdapterError(f"missing skill definition: {skill_id}")
        tool_name = source.get("tool_name")
        if tool_name is not None and (
            not isinstance(tool_name, str) or not tool_name
        ):
            raise CandidateAdapterError(f"skill {skill_id} has invalid tool_name")

        candidates.append(
            SkillCandidate(
                skill_id=skill_id,
                name=_string_field(source, "name", default=skill_id),
                description=_string_field(source, "description", default=""),
                body=_string_field(source, "body", default=""),
                rank=rank,
                score=1.0 / rank,
                tool_name=tool_name,
                category_path=_category_path(source),
                metadata=dict(source),
            )
        )
    return candidates


def _string_field(source: Mapping[str, Any], field: str, default: str) -> str:
    value = source.get(field, default)
    if not isinstance(value, str):
        raise CandidateAdapterError(f"skill.{field} must be a string")
    return value


def _category_path(source: Mapping[str, Any]) -> tuple[str, ...]:
    value = source.get("category_path", source.get("category", ()))
    if isinstance(value, str):
        return tuple(part for part in value.split("/") if part)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if any(not isinstance(part, str) or not part for part in value):
            raise CandidateAdapterError("skill.category_path must contain strings")
        return tuple(value)
    raise CandidateAdapterError("skill.category_path must be a string or sequence")

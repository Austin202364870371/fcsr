"""Prepare leakage-controlled hard tasks from frozen FCSR reranker output."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class InsufficientEligibleTasks(ValueError):
    """Raised when runnable tasks cannot satisfy the fixed pilot quotas."""


class PublicInstructionSkill(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    alias: str = Field(pattern=r"^S\d{2}$")
    name: str
    description: str
    body: str
    rank: int = Field(ge=1)
    reranker_score: float


class PublicPilotTask(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: str
    task: str
    domain: str
    skills: tuple[PublicInstructionSkill, ...]


class PrivatePilotEvaluation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: str
    gt_skill_ids: tuple[str, ...]
    alias_to_skill_id: dict[str, str]
    full_coverage: bool


class PreparedPilot(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    public_tasks: tuple[PublicPilotTask, ...]
    evaluations: tuple[PrivatePilotEvaluation, ...]


_QUOTAS = {
    ("single", True): 3,
    ("single", False): 2,
    ("multi", True): 5,
    ("multi", False): 5,
}


def prepare_hard_pilot(
    queries: Iterable[Mapping[str, Any]],
    rankings: Iterable[Mapping[str, Any]],
    skills: Iterable[Mapping[str, Any]],
    *,
    eligible_task_ids: set[str],
    seed: int = 42,
) -> PreparedPilot:
    """Join records and select a deterministic 15-task development pilot."""
    query_index = {_required(record, "query_id"): record for record in queries}
    ranking_index = {_required(record, "query_id"): record for record in rankings}
    buckets: dict[tuple[str, bool], list[str]] = {key: [] for key in _QUOTAS}
    for task_id in sorted(eligible_task_ids & query_index.keys() & ranking_index.keys()):
        query = query_index[task_id]
        gt_ids = _string_list(query.get("gt_skill_ids"), "gt_skill_ids")
        candidate_ids = _candidate_ids(ranking_index[task_id])
        key = ("single" if len(gt_ids) == 1 else "multi", set(gt_ids) <= set(candidate_ids))
        buckets[key].append(task_id)

    selected: list[str] = []
    shortages: list[str] = []
    for key, quota in _QUOTAS.items():
        ordered = sorted(buckets[key], key=lambda value: _sample_key(seed, value))
        if len(ordered) < quota:
            shortages.append(f"{key[0]}/{key[1]}={len(ordered)}/{quota}")
        selected.extend(ordered[:quota])
    if shortages:
        raise InsufficientEligibleTasks(
            "eligible tasks cannot satisfy pilot quotas: " + ", ".join(shortages)
        )

    needed_ids = {
        skill_id
        for task_id in selected
        for skill_id in _candidate_ids(ranking_index[task_id])
    }
    skill_index = {
        _required(record, "skill_id"): record
        for record in skills
        if record.get("skill_id") in needed_ids
    }
    public_tasks: list[PublicPilotTask] = []
    evaluations: list[PrivatePilotEvaluation] = []
    for task_id in selected:
        query = query_index[task_id]
        candidates = ranking_index[task_id].get("reranked_candidates")
        if not isinstance(candidates, list):
            raise ValueError(f"{task_id}: reranked_candidates must be a list")
        public_skills: list[PublicInstructionSkill] = []
        alias_map: dict[str, str] = {}
        for rank, candidate in enumerate(candidates, start=1):
            skill_id = _required(candidate, "skill_id")
            definition = skill_index.get(skill_id)
            if definition is None:
                raise ValueError(f"missing Skill definition: {skill_id}")
            alias = f"S{rank:02d}"
            alias_map[alias] = skill_id
            public_skills.append(
                PublicInstructionSkill(
                    alias=alias,
                    name=_text(definition, "name"),
                    description=_text(definition, "description"),
                    body=_text(definition, "body"),
                    rank=rank,
                    reranker_score=float(candidate.get("reranker_score", 0.0)),
                )
            )
        gt_ids = tuple(_string_list(query.get("gt_skill_ids"), "gt_skill_ids"))
        public_tasks.append(
            PublicPilotTask(
                task_id=task_id,
                task=_required(query, "query"),
                domain=_text(query, "domain"),
                skills=tuple(public_skills),
            )
        )
        evaluations.append(
            PrivatePilotEvaluation(
                task_id=task_id,
                gt_skill_ids=gt_ids,
                alias_to_skill_id=alias_map,
                full_coverage=set(gt_ids) <= set(alias_map.values()),
            )
        )
    return PreparedPilot(
        public_tasks=tuple(public_tasks),
        evaluations=tuple(evaluations),
    )


def _candidate_ids(record: Mapping[str, Any]) -> list[str]:
    candidates = record.get("reranked_candidates")
    if not isinstance(candidates, list):
        raise ValueError("reranked_candidates must be a list")
    return [_required(candidate, "skill_id") for candidate in candidates]


def _sample_key(seed: int, task_id: str) -> str:
    return hashlib.sha256(f"{seed}:{task_id}".encode()).hexdigest()


def _required(record: Mapping[str, Any], field: str) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _text(record: Mapping[str, Any], field: str) -> str:
    value = record.get(field, "")
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    return value


def _string_list(value: Any, field: str) -> list[str]:
    if not isinstance(value, list) or not value or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise ValueError(f"{field} must be a non-empty string list")
    return value

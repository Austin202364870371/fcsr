"""Prepare the frozen Hard-15 pilot without exposing evaluator-only IDs."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from agent.hard15_organizations import normalize_category_path
from agent.task_catalog import PilotCatalog, PilotStratum
from agent.task_packages import audit_planning_environment


class Hard15Skill(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    alias: str = Field(pattern=r"^S\d{2}$")
    skill_id: str = Field(min_length=1, exclude=True)
    name: str = Field(min_length=1)
    description: str = ""
    body: str = ""
    rank: int = Field(ge=1)
    reranker_score: float
    category_path: tuple[str, ...] = Field(exclude=True)


class Hard15Task(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: str
    task: str
    domain: str
    stratum: PilotStratum = Field(exclude=True)
    skills: tuple[Hard15Skill, ...]


class Hard15Evaluation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: str
    gt_skill_ids: tuple[str, ...]
    alias_to_skill_id: dict[str, str]
    full_candidate_coverage: bool


class FixedHard15Pilot(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    tasks: tuple[Hard15Task, ...]
    evaluations: tuple[Hard15Evaluation, ...]


def prepare_fixed_hard15(
    catalog: PilotCatalog,
    queries: Iterable[Mapping[str, Any]],
    rankings: Iterable[Mapping[str, Any]],
    skills: Iterable[Mapping[str, Any]],
    *,
    packages_root: Path | None = None,
    candidate_limit: int = 20,
) -> FixedHard15Pilot:
    """Join the exact catalog tasks to frozen FCSR Top-k candidates."""
    if candidate_limit != 20:
        raise ValueError("the main Hard15 experiment requires exactly Top-20")
    query_index = {_required(row, "query_id"): row for row in queries}
    ranking_index = {_required(row, "query_id"): row for row in rankings}
    if packages_root is not None:
        failed = [
            audit_planning_environment(item.source_task_id, packages_root)
            for item in catalog.tasks
        ]
        failed = [item for item in failed if not item.planning_ready]
        if failed:
            details = ", ".join(f"{item.task_id}:{item.status}" for item in failed)
            raise ValueError(f"Hard15 public task context is not ready: {details}")

    candidate_rows: dict[str, list[Mapping[str, Any]]] = {}
    needed: set[str] = set()
    for item in catalog.tasks:
        if item.task_id not in query_index or item.task_id not in ranking_index:
            raise ValueError(f"missing query or ranking for {item.task_id}")
        rows = ranking_index[item.task_id].get("reranked_candidates")
        if not isinstance(rows, list) or len(rows) != 20:
            raise ValueError(f"{item.task_id}: requires exactly 20 reranked candidates")
        selected = rows[:candidate_limit]
        ids = [_required(row, "skill_id") for row in selected]
        if len(ids) != len(set(ids)):
            raise ValueError(f"{item.task_id}: duplicate reranked Skill IDs")
        ranks = [row.get("reranker_rank") for row in selected]
        if ranks != list(range(1, 21)):
            raise ValueError(
                f"{item.task_id}: reranker_rank must be the ordered sequence 1..20"
            )
        candidate_rows[item.task_id] = selected
        needed.update(ids)

    skill_index = {
        _required(row, "skill_id"): row
        for row in skills
        if row.get("skill_id") in needed
    }
    missing = sorted(needed - skill_index.keys())
    if missing:
        raise ValueError(f"missing {len(missing)} Skill definitions; first={missing[0]}")

    public: list[Hard15Task] = []
    private: list[Hard15Evaluation] = []
    for item in catalog.tasks:
        query = query_index[item.task_id]
        gt_ids = tuple(_string_list(query.get("gt_skill_ids"), "gt_skill_ids"))
        rows = candidate_rows[item.task_id]
        candidate_ids = tuple(_required(row, "skill_id") for row in rows)
        actual_stratum = _stratum(gt_ids, candidate_ids)
        if actual_stratum != item.stratum:
            raise ValueError(
                f"{item.task_id}: catalog stratum {item.stratum} != {actual_stratum}"
            )
        cards: list[Hard15Skill] = []
        alias_map: dict[str, str] = {}
        for rank, row in enumerate(rows, start=1):
            skill_id = _required(row, "skill_id")
            source = skill_index[skill_id]
            alias = f"S{rank:02d}"
            alias_map[alias] = skill_id
            cards.append(
                Hard15Skill(
                    alias=alias,
                    skill_id=skill_id,
                    name=_optional_text(source, "name") or skill_id.rsplit("/", 1)[-1],
                    description=_optional_text(source, "description"),
                    body=_optional_text(source, "body"),
                    rank=rank,
                    reranker_score=float(row.get("reranker_score", 0.0)),
                    category_path=normalize_category_path(source),
                )
            )
        public.append(
            Hard15Task(
                task_id=item.task_id,
                task=_required(query, "query"),
                domain=_optional_text(query, "domain"),
                stratum=item.stratum,
                skills=tuple(cards),
            )
        )
        private.append(
            Hard15Evaluation(
                task_id=item.task_id,
                gt_skill_ids=gt_ids,
                alias_to_skill_id=alias_map,
                full_candidate_coverage=set(gt_ids) <= set(candidate_ids),
            )
        )
    return FixedHard15Pilot(tasks=tuple(public), evaluations=tuple(private))


def _stratum(gt_ids: tuple[str, ...], candidates: tuple[str, ...]) -> PilotStratum:
    prefix = "single" if len(gt_ids) == 1 else "multi"
    suffix = "full" if set(gt_ids) <= set(candidates) else "missing"
    return f"{prefix}_{suffix}"  # type: ignore[return-value]


def _required(row: Mapping[str, Any], field: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _optional_text(row: Mapping[str, Any], field: str) -> str:
    value = row.get(field, "")
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    return value


def _string_list(value: Any, field: str) -> list[str]:
    if not isinstance(value, list) or not value or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise ValueError(f"{field} must be a non-empty string list")
    return value

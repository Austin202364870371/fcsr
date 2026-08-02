"""Deterministic hierarchical organization of retrieved Skill candidates."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from agent.models import SkillBundle, SkillCandidate


class SkillGroup(BaseModel):
    """One disclosed category and its selected member Skills."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    category_path: tuple[str, ...]
    label: str = Field(min_length=1)
    score: float = Field(gt=0)
    best_rank: int = Field(ge=1)
    skills: tuple[SkillCandidate, ...]


class HierarchySkillBundle(SkillBundle):
    """A SkillBundle with explicit category grouping metadata."""

    strategy: Literal["hierarchy"] = "hierarchy"
    groups: tuple[SkillGroup, ...]

    @model_validator(mode="after")
    def groups_match_selected_skills(self) -> "HierarchySkillBundle":
        bundle_ids = [skill.skill_id for skill in self.skills]
        grouped_ids = [
            skill.skill_id
            for group in self.groups
            for skill in group.skills
        ]
        if len(grouped_ids) != len(set(grouped_ids)):
            raise ValueError("duplicate skill ids across hierarchy groups")
        if set(grouped_ids) != set(bundle_ids):
            raise ValueError("hierarchy groups must cover exactly the bundle skills")
        paths = [group.category_path for group in self.groups]
        if len(paths) != len(set(paths)):
            raise ValueError("duplicate hierarchy category paths")
        return self


class HierarchyOrganizer:
    """Select Top-r categories, then retain globally ranked Skills within budget."""

    def __init__(
        self,
        max_groups: int,
        max_skills: int,
        category_depth: int = 1,
    ) -> None:
        _require_positive_int("max_groups", max_groups)
        _require_positive_int("max_skills", max_skills)
        _require_positive_int("category_depth", category_depth)
        self.max_groups = max_groups
        self.max_skills = max_skills
        self.category_depth = category_depth

    def organize(
        self,
        candidates: Iterable[SkillCandidate],
    ) -> HierarchySkillBundle:
        ranked = sorted(candidates, key=lambda candidate: candidate.rank)
        grouped: dict[tuple[str, ...], list[SkillCandidate]] = defaultdict(list)
        for skill in ranked:
            grouped[self._group_path(skill)].append(skill)

        selected_paths = [
            path
            for path, members in sorted(
                grouped.items(),
                key=lambda item: self._group_sort_key(item[0], item[1]),
            )[: self.max_groups]
        ]
        selected_path_set = set(selected_paths)
        selected_skills = [
            skill
            for skill in ranked
            if self._group_path(skill) in selected_path_set
        ][: self.max_skills]

        groups: list[SkillGroup] = []
        for path in selected_paths:
            visible_members = [
                skill
                for skill in selected_skills
                if self._group_path(skill) == path
            ]
            if not visible_members:
                continue
            all_members = grouped[path]
            groups.append(
                SkillGroup(
                    category_path=path,
                    label="/".join(path),
                    score=sum(1.0 / skill.rank for skill in all_members),
                    best_rank=min(skill.rank for skill in all_members),
                    skills=visible_members,
                )
            )

        return HierarchySkillBundle(skills=selected_skills, groups=groups)

    def _group_path(self, skill: SkillCandidate) -> tuple[str, ...]:
        if not skill.category_path:
            return ("uncategorized",)
        return skill.category_path[: self.category_depth]

    @staticmethod
    def _group_sort_key(
        path: tuple[str, ...],
        members: list[SkillCandidate],
    ) -> tuple[float, int, tuple[str, ...]]:
        reciprocal_rank_score = sum(1.0 / skill.rank for skill in members)
        best_rank = min(skill.rank for skill in members)
        return -reciprocal_rank_score, best_rank, path


def _require_positive_int(field: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field} must be a positive integer")

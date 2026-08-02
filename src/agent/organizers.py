"""Skill bundle organization strategies."""

from __future__ import annotations

from collections.abc import Iterable

from agent.models import SkillBundle, SkillCandidate


class FlatOrganizer:
    """Expose the highest-ranked candidates as a flat bounded bundle."""

    def __init__(self, max_skills: int) -> None:
        if (
            isinstance(max_skills, bool)
            or not isinstance(max_skills, int)
            or max_skills < 1
        ):
            raise ValueError("max_skills must be a positive integer")
        self.max_skills = max_skills

    def organize(self, candidates: Iterable[SkillCandidate]) -> SkillBundle:
        ranked = sorted(candidates, key=lambda candidate: candidate.rank)
        return SkillBundle(strategy="flat", skills=ranked[: self.max_skills])

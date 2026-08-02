"""Injectable selection policies for organized skill bundles."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict

from agent.models import SkillBundle, SkillCandidate


class Selection(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    skill_id: str
    tool_name: str
    arguments: dict[str, Any]
    reason: str


class SkillSelector(Protocol):
    def select(self, task: str, bundle: SkillBundle) -> Selection | None:
        """Return one executable selection or None when no skill is usable."""


ArgumentBuilder = Callable[[str, SkillCandidate], Mapping[str, Any]]


class FirstRankedSelector:
    """Deterministic baseline that always selects the first ranked skill."""

    def __init__(self, argument_builder: ArgumentBuilder | None = None) -> None:
        self._argument_builder = argument_builder or _empty_arguments

    def select(self, task: str, bundle: SkillBundle) -> Selection | None:
        if not bundle.skills:
            return None
        skill = bundle.skills[0]
        arguments = self._argument_builder(task, skill)
        if not isinstance(arguments, Mapping):
            raise ValueError("selector arguments must be an object")
        return Selection(
            skill_id=skill.skill_id,
            tool_name=skill.tool_name,
            arguments=dict(arguments),
            reason="selected highest-ranked skill",
        )


def _empty_arguments(task: str, skill: SkillCandidate) -> Mapping[str, Any]:
    del task, skill
    return {}

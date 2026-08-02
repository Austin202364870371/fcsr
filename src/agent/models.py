"""Validated domain models shared by agent runtime components."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class FrozenModel(BaseModel):
    """Forbid accidental schema drift in experiment records."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class SkillCandidate(FrozenModel):
    skill_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    description: str = ""
    body: str = ""
    rank: int = Field(ge=1)
    score: float
    tool_name: str | None = Field(default=None, min_length=1)
    category_path: tuple[str, ...] = ()
    metadata: dict[str, Any] = Field(default_factory=dict)


class SkillBundle(FrozenModel):
    strategy: Literal["flat"]
    skills: tuple[SkillCandidate, ...]

    @field_validator("skills")
    @classmethod
    def reject_duplicate_skill_ids(
        cls, skills: tuple[SkillCandidate, ...]
    ) -> tuple[SkillCandidate, ...]:
        skill_ids = [skill.skill_id for skill in skills]
        if len(skill_ids) != len(set(skill_ids)):
            raise ValueError("duplicate skill ids")
        return skills


class ToolCall(FrozenModel):
    tool_name: str = Field(min_length=1)
    arguments: dict[str, Any]


class ToolResult(FrozenModel):
    tool_name: str
    ok: bool
    output: Any = None
    error: str | None = None


class VerificationResult(FrozenModel):
    passed: bool
    verifier_id: str
    details: str


class AgentRunResult(FrozenModel):
    task_id: str
    selected_skill_id: str | None
    tool_result: ToolResult | None
    verification: VerificationResult
    termination_reason: Literal[
        "verified",
        "verification_failed",
        "selection_failed",
        "execution_failed",
    ]
    trace: tuple[dict[str, Any], ...]

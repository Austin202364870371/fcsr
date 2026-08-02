"""Schema-validated DeepSeek planning over an organized Hard-15 prompt."""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from agent.hard15_organizations import OrganizedTask
from agent.llm import PlanningClient, SkillPlan


class PlanningAttempt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: str
    method: Literal["flat", "hierarchy", "graph"]
    model: str
    fingerprint: str
    valid: bool
    plan: SkillPlan | None = None
    error: str | None = None
    selected_candidate_aliases: tuple[str, ...]
    rendered_characters: int = Field(ge=0)
    omitted_count: int = Field(ge=0)
    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    organization_metadata: dict[str, Any]


def plan_organized_task(
    organized: OrganizedTask,
    *,
    client: PlanningClient,
    model: str,
    fingerprint: str,
) -> PlanningAttempt:
    reply = client.complete(model=model, messages=_messages(organized))
    try:
        plan = SkillPlan.model_validate_json(reply.content)
        allowed = set(organized.selected_aliases)
        used = set(plan.selected_skill_aliases)
        used.update(alias for step in plan.steps for alias in step.skill_aliases)
        unknown = sorted(used - allowed)
        if unknown:
            raise ValueError(f"unknown or omitted Skill aliases: {unknown}")
        dangling = sorted(used - set(plan.selected_skill_aliases))
        if dangling:
            raise ValueError(f"steps use unselected Skill aliases: {dangling}")
    except Exception as exc:
        return _attempt(
            organized,
            model=model,
            fingerprint=fingerprint,
            valid=False,
            error=f"{type(exc).__name__}: {exc}",
            prompt_tokens=reply.prompt_tokens,
            completion_tokens=reply.completion_tokens,
        )
    return _attempt(
        organized,
        model=model,
        fingerprint=fingerprint,
        valid=True,
        plan=plan,
        prompt_tokens=reply.prompt_tokens,
        completion_tokens=reply.completion_tokens,
    )


def failed_attempt(
    organized: OrganizedTask,
    *,
    model: str,
    fingerprint: str,
    error: Exception,
) -> PlanningAttempt:
    return _attempt(
        organized,
        model=model,
        fingerprint=fingerprint,
        valid=False,
        error=f"{type(error).__name__}: {error}",
        prompt_tokens=0,
        completion_tokens=0,
    )


def _attempt(
    organized: OrganizedTask,
    *,
    model: str,
    fingerprint: str,
    valid: bool,
    prompt_tokens: int,
    completion_tokens: int,
    plan: SkillPlan | None = None,
    error: str | None = None,
) -> PlanningAttempt:
    return PlanningAttempt(
        task_id=organized.task_id,
        method=organized.method,
        model=model,
        fingerprint=fingerprint,
        valid=valid,
        plan=plan,
        error=error,
        selected_candidate_aliases=organized.selected_aliases,
        rendered_characters=len(organized.rendered_prompt),
        omitted_count=len(organized.omitted_aliases),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        organization_metadata=organized.metadata,
    )


def _messages(organized: OrganizedTask) -> list[dict[str, str]]:
    schema = {
        "selected_skill_aliases": ["S01"],
        "steps": [
            {
                "id": "step-1",
                "objective": "concrete subgoal",
                "skill_aliases": ["S01"],
                "expected_output": "verifiable artifact or state",
            }
        ],
        "final_output": "requested artifact or answer",
    }
    return [
        {
            "role": "system",
            "content": (
                "You are a planning component. Select only useful aliases from the "
                "provided organized Skill set and return exactly one JSON object. "
                "Never invent aliases. Write concise operational objectives, not "
                "private chain-of-thought."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "task_id": organized.task_id,
                    "task": organized.task,
                    "organized_skills": json.loads(organized.rendered_prompt),
                    "required_schema_example": schema,
                },
                ensure_ascii=False,
            ),
        },
    ]

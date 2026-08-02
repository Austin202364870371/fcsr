"""Validated one-call Skill selection and planning for the hard pilot."""

from __future__ import annotations

import json
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator

from agent.hard_pilot import PublicPilotTask


class LLMReply(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    content: str
    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)


class PlanningClient(Protocol):
    def complete(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
    ) -> LLMReply: ...


class PlanStep(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=1)
    objective: str = Field(min_length=1)
    skill_aliases: tuple[str, ...] = Field(min_length=1)
    expected_output: str = Field(min_length=1)


class SkillPlan(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    selected_skill_aliases: tuple[str, ...] = Field(min_length=1)
    steps: tuple[PlanStep, ...] = Field(min_length=1)
    final_output: str = Field(min_length=1)

    @field_validator("selected_skill_aliases")
    @classmethod
    def unique_selected_aliases(cls, aliases: tuple[str, ...]) -> tuple[str, ...]:
        if len(aliases) != len(set(aliases)):
            raise ValueError("selected Skill aliases must be unique")
        return aliases


class PlanRunRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: str
    model: str
    plan: SkillPlan
    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)


def plan_task(
    task: PublicPilotTask,
    *,
    client: PlanningClient,
    model: str = "deepseek-v4-flash",
    body_char_limit: int = 1600,
) -> PlanRunRecord:
    """Select instruction Skills and produce a schema-validated execution plan."""
    if body_char_limit < 0:
        raise ValueError("body_char_limit must be non-negative")
    reply = client.complete(
        model=model,
        messages=_messages(task, body_char_limit),
    )
    try:
        payload = json.loads(reply.content)
    except json.JSONDecodeError as exc:
        raise ValueError("planner returned invalid JSON") from exc
    plan = SkillPlan.model_validate(payload)
    allowed = {skill.alias for skill in task.skills}
    used = set(plan.selected_skill_aliases)
    used.update(alias for step in plan.steps for alias in step.skill_aliases)
    unknown = sorted(used - allowed)
    if unknown:
        raise ValueError(f"unknown Skill aliases: {unknown}")
    selected = set(plan.selected_skill_aliases)
    unselected = sorted(used - selected)
    if unselected:
        raise ValueError(f"plan steps use unselected Skill aliases: {unselected}")
    return PlanRunRecord(
        task_id=task.task_id,
        model=model,
        plan=plan,
        prompt_tokens=reply.prompt_tokens,
        completion_tokens=reply.completion_tokens,
    )


def _messages(task: PublicPilotTask, body_char_limit: int) -> list[dict[str, str]]:
    cards = [
        {
            "alias": skill.alias,
            "name": skill.name,
            "description": skill.description,
            "instructions": skill.body[:body_char_limit],
            "retrieval_rank": skill.rank,
        }
        for skill in task.skills
    ]
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
        "final_output": "requested final artifact or answer",
    }
    return [
        {
            "role": "system",
            "content": (
                "Select only useful Skill aliases and make a concise executable plan. "
                "Return one JSON object only. Never invent aliases. Do not expose "
                "chain-of-thought; write short operational objectives."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "task_id": task.task_id,
                    "task": task.task,
                    "skills": cards,
                    "required_schema_example": schema,
                },
                ensure_ascii=False,
            ),
        },
    ]


class DeepSeekPlanningClient:
    """Small OpenAI-compatible adapter for the official DeepSeek endpoint."""

    def __init__(self, api_key: str, base_url: str = "https://api.deepseek.com") -> None:
        if not api_key:
            raise ValueError("api_key must not be empty")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("openai is required for DeepSeek planning") from exc
        self._client = OpenAI(api_key=api_key, base_url=base_url)

    def complete(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
    ) -> LLMReply:
        response = self._client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0,
            response_format={"type": "json_object"},
            stream=False,
        )
        content = response.choices[0].message.content
        if not isinstance(content, str) or not content:
            raise ValueError("DeepSeek returned empty plan content")
        usage = response.usage
        return LLMReply(
            content=content,
            prompt_tokens=int(usage.prompt_tokens if usage else 0),
            completion_tokens=int(usage.completion_tokens if usage else 0),
        )

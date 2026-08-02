"""Shared LLM planning contracts and the DeepSeek client."""

from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator


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


class DeepSeekPlanningClient:
    """Small OpenAI-compatible adapter for DeepSeek JSON planning."""

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

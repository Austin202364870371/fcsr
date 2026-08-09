"""Immutable data models shared by the Skill organization pipeline."""

from __future__ import annotations

import hashlib
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class SkillRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    skill_id: str = Field(min_length=1)
    name: str
    description: str
    body: str
    source: Literal["pool", "gt", "distractor"]

    def canonical_hash(self) -> str:
        payload = json.dumps(
            {"body": self.body, "description": self.description, "name": self.name},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


class FrozenSkill(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    alias: str = Field(pattern=r"^S0[1-8]$")
    rank: int = Field(ge=1, le=8)
    record: SkillRecord

    def organizer_view(self) -> dict[str, object]:
        return {
            "alias": self.alias,
            "rank": self.rank,
            "name": self.record.name,
            "description": self.record.description,
            "body": self.record.body,
        }


class TaskInput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    task_key: str = Field(pattern=r"^T\d{3}$")
    task_id: str = Field(min_length=1)
    skills: tuple[FrozenSkill, ...] = Field(min_length=8, max_length=8)

    @model_validator(mode="after")
    def validate_rank_alias_alignment(self) -> "TaskInput":
        expected = tuple((f"S{rank:02d}", rank) for rank in range(1, 9))
        actual = tuple((skill.alias, skill.rank) for skill in self.skills)
        if actual != expected:
            raise ValueError("skills must map S01--S08 to retrieval ranks 1--8")
        return self

    def organizer_view(self) -> list[dict[str, object]]:
        return [skill.organizer_view() for skill in self.skills]


class FrozenInputs(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    predictions_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    skills_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    task_ids_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    task_catalog_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    skillsbench_version: str
    skillsbench_commit: str
    huggingface_revision: str
    tasks: tuple[TaskInput, ...] = Field(min_length=15, max_length=15)

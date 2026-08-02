"""Frozen catalog for the lightweight SkillsBench Hard-15 pilot."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


PilotStratum = Literal[
    "single_full",
    "single_missing",
    "multi_full",
    "multi_missing",
]


class PilotCatalogEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: str = Field(min_length=1)
    source_task_id: str = Field(min_length=1)
    source_path: str = Field(pattern=r"^tasks/[a-z0-9][a-z0-9-]*$")
    stratum: PilotStratum
    estimated_context_bytes: int = Field(gt=0)


class PilotCatalog(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    skillsbench_version: Literal["v1.1"]
    github_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    huggingface_repo: Literal["benchflow/skillsbench"]
    huggingface_revision: str = Field(min_length=7)
    tasks: tuple[PilotCatalogEntry, ...]

    @model_validator(mode="after")
    def validate_fixed_catalog(self) -> "PilotCatalog":
        ids = [task.task_id for task in self.tasks]
        if len(ids) != len(set(ids)):
            raise ValueError("task catalog contains duplicate task IDs")
        if len(ids) != 15:
            raise ValueError("task catalog must contain exactly 15 tasks")
        for task in self.tasks:
            if task.source_task_id != task.task_id:
                raise ValueError("source_task_id must match task_id in this pilot")
            if task.source_path != f"tasks/{task.source_task_id}":
                raise ValueError("source_path must match source_task_id")
        return self

    @property
    def task_ids(self) -> tuple[str, ...]:
        return tuple(task.task_id for task in self.tasks)


def load_pilot_catalog(path: Path) -> PilotCatalog:
    """Load and validate the frozen task catalog."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    return PilotCatalog.model_validate(payload)


def require_catalog_order(
    catalog: PilotCatalog,
    task_ids: Sequence[str],
) -> None:
    """Reject reordered, missing, or additional task lists."""
    if tuple(task_ids) != catalog.task_ids:
        raise ValueError("task IDs do not match the frozen catalog order")


"""Audit whether a benchmark task has a reproducible local execution package."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict


class EnvironmentAudit(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: str
    status: Literal[
        "ready",
        "missing_task_environment",
        "missing_container",
        "missing_verifier",
    ]
    execution_ready: bool
    task_root: str


def audit_task_environment(task_id: str, environments_root: Path) -> EnvironmentAudit:
    """Require the container and deterministic verifier used by SkillsBench."""
    task_root = environments_root / task_id
    container = task_root / "environment" / "Dockerfile"
    verifier_dir = task_root / "verifier"
    has_verifier = any(
        path.is_file()
        for path in (
            verifier_dir / "test.sh",
            verifier_dir / "test_outputs.py",
        )
    )
    if not task_root.is_dir():
        status = "missing_task_environment"
    elif not container.is_file():
        status = "missing_container"
    elif not has_verifier:
        status = "missing_verifier"
    else:
        status = "ready"
    return EnvironmentAudit(
        task_id=task_id,
        status=status,
        execution_ready=status == "ready",
        task_root=str(task_root),
    )

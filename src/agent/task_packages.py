"""Leakage-safe synchronization and audit for planning-only task context."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from agent.task_catalog import PilotCatalog


PlanningStatus = Literal[
    "ready",
    "missing_task",
    "missing_task_md",
    "prohibited_path",
]
Downloader = Callable[..., str]


class PlanningAudit(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: str
    status: PlanningStatus
    planning_ready: bool
    task_root: str
    file_count: int = Field(ge=0)
    downloaded_bytes: int = Field(ge=0)
    prohibited_paths: tuple[str, ...] = ()


class TaskSyncManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    repository: str
    revision: str
    packages_root: str
    ready: bool
    tasks: tuple[PlanningAudit, ...]


def prohibited_relative_path(path: str) -> bool:
    """Return whether a downloaded path exposes evaluator-only information."""
    parts = tuple(part.lower() for part in PurePosixPath(path.replace("\\", "/")).parts)
    for first, second in zip(parts, parts[1:]):
        if (first, second) == ("environment", "skills"):
            return True
    if "oracle" in parts or "verifier" in parts:
        return True
    return any(
        part in {"groundtruth", "ground_truth", "reference_answer"}
        for part in parts
    )


def build_download_patterns(catalog: PilotCatalog) -> tuple[list[str], list[str]]:
    """Build exact allow patterns and global evaluator-only exclusions."""
    allow = [f"{task.source_task_id}/**" for task in catalog.tasks]
    ignore = [
        "*/environment/skills/**",
        "*/oracle/**",
        "*/verifier/**",
        "*/**/groundtruth/**",
        "*/**/ground_truth/**",
        "*/**/reference_answer/**",
    ]
    return allow, ignore


def audit_planning_environment(task_id: str, packages_root: Path) -> PlanningAudit:
    """Require a public task definition and reject evaluator-only files."""
    task_root = packages_root / task_id
    if not task_root.is_dir():
        return _audit(task_id, task_root, "missing_task")
    files = tuple(path for path in task_root.rglob("*") if path.is_file())
    prohibited = tuple(
        path.relative_to(packages_root).as_posix()
        for path in files
        if prohibited_relative_path(path.relative_to(packages_root).as_posix())
    )
    if prohibited:
        return _audit(
            task_id,
            task_root,
            "prohibited_path",
            files=files,
            prohibited=prohibited,
        )
    if not (task_root / "task.md").is_file():
        return _audit(task_id, task_root, "missing_task_md", files=files)
    return _audit(task_id, task_root, "ready", files=files)


def sync_task_packages(
    catalog: PilotCatalog,
    packages_root: Path,
    *,
    downloader: Downloader,
    revision: str | None = None,
) -> TaskSyncManifest:
    """Synchronize the fixed public context and return strict readiness."""
    selected_revision = revision or catalog.huggingface_revision
    allow, ignore = build_download_patterns(catalog)
    packages_root.mkdir(parents=True, exist_ok=True)
    downloader(
        repo_id=catalog.huggingface_repo,
        repo_type="dataset",
        revision=selected_revision,
        allow_patterns=allow,
        ignore_patterns=ignore,
        local_dir=str(packages_root),
    )
    audits = tuple(
        audit_planning_environment(task.source_task_id, packages_root)
        for task in catalog.tasks
    )
    return TaskSyncManifest(
        repository=catalog.huggingface_repo,
        revision=selected_revision,
        packages_root=str(packages_root),
        ready=all(audit.planning_ready for audit in audits),
        tasks=audits,
    )


def default_snapshot_downloader(**kwargs: Any) -> str:
    """Load huggingface_hub only for the network synchronization command."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required; install requirements-hard15.txt"
        ) from exc
    return str(snapshot_download(**kwargs))


def _audit(
    task_id: str,
    task_root: Path,
    status: PlanningStatus,
    *,
    files: tuple[Path, ...] = (),
    prohibited: tuple[str, ...] = (),
) -> PlanningAudit:
    return PlanningAudit(
        task_id=task_id,
        status=status,
        planning_ready=status == "ready",
        task_root=str(task_root),
        file_count=len(files),
        downloaded_bytes=sum(path.stat().st_size for path in files),
        prohibited_paths=prohibited,
    )


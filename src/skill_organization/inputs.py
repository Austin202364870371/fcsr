"""Fail-closed loading of frozen retrieval and Skill inputs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from data_io import stream_jsonl
from skill_organization.models import FrozenInputs, FrozenSkill, SkillRecord, TaskInput


HARD15_TASK_IDS = (
    "jax-computing-basics",
    "dialogue-parser",
    "econ-detrending-correlation",
    "citation-check",
    "enterprise-information-search",
    "flood-risk-analysis",
    "syzkaller-ppdev-syzlang",
    "manufacturing-fjsp-optimization",
    "threejs-to-obj",
    "threejs-structure-parser",
    "setup-fuzzing-py",
    "suricata-custom-exfil",
    "powerlifting-coef-calc",
    "xlsx-recover-data",
    "parallel-tfidf-search",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_object(path: Path, *, label: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _load_task_ids(path: Path) -> tuple[str, ...]:
    task_ids = tuple(
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if len(task_ids) != 15 or len(set(task_ids)) != 15:
        raise ValueError("Hard-15 must contain exactly 15 unique task IDs")
    if task_ids != HARD15_TASK_IDS:
        raise ValueError(
            "Hard-15 task IDs or order differ from the registered experiment"
        )
    return task_ids


def _validate_catalog(catalog: dict[str, Any], task_ids: tuple[str, ...]) -> None:
    tasks = catalog.get("tasks")
    if not isinstance(tasks, list):
        raise ValueError("task catalog must contain a tasks list")
    catalog_ids = tuple(
        item.get("task_id")
        for item in tasks
        if isinstance(item, dict) and item.get("task_id")
    )
    if catalog_ids != task_ids:
        raise ValueError(
            "catalog task IDs and order must exactly match registered Hard-15"
        )


def load_frozen_inputs(
    *,
    predictions_path: Path,
    skills_path: Path,
    task_ids_path: Path,
    task_catalog_path: Path,
    expected_skills_sha256: str | None,
    top_k: int = 8,
) -> FrozenInputs:
    if top_k != 8:
        raise ValueError("Skill organization requires top_k=8")

    task_ids = _load_task_ids(task_ids_path)
    catalog = _load_object(task_catalog_path, label="task catalog")
    _validate_catalog(catalog, task_ids)
    predictions = _load_object(predictions_path, label="predictions")

    selected: dict[str, tuple[str, ...]] = {}
    for task_id in task_ids:
        ranked = predictions.get(task_id)
        if not isinstance(ranked, list) or not all(
            isinstance(item, str) for item in ranked
        ):
            raise ValueError(
                f"missing or invalid frozen predictions for task: {task_id}"
            )
        top_ids = tuple(ranked[:top_k])
        if len(top_ids) != top_k or len(set(top_ids)) != top_k:
            raise ValueError(
                f"task must have eight unique frozen predictions: {task_id}"
            )
        selected[task_id] = top_ids

    skills_hash = sha256_file(skills_path)
    if expected_skills_sha256 and skills_hash.lower() != expected_skills_sha256.lower():
        raise ValueError("Hard Skill pool SHA-256 mismatch")

    wanted = {skill_id for skill_ids in selected.values() for skill_id in skill_ids}
    records: dict[str, SkillRecord] = {}
    for raw in stream_jsonl(skills_path):
        skill_id = raw.get("skill_id")
        if skill_id not in wanted:
            continue
        record = SkillRecord.model_validate(raw)
        previous = records.get(record.skill_id)
        if previous is not None and previous != record:
            raise ValueError(f"conflicting duplicate Skill record: {record.skill_id}")
        records[record.skill_id] = record

    missing = sorted(wanted - records.keys())
    if missing:
        raise ValueError(f"missing frozen Skill records: {missing}")

    tasks = tuple(
        TaskInput(
            task_key=f"T{index:03d}",
            task_id=task_id,
            skills=tuple(
                FrozenSkill(alias=f"S{rank:02d}", rank=rank, record=records[skill_id])
                for rank, skill_id in enumerate(selected[task_id], start=1)
            ),
        )
        for index, task_id in enumerate(task_ids, start=1)
    )
    return FrozenInputs(
        predictions_sha256=sha256_file(predictions_path),
        skills_sha256=skills_hash,
        task_ids_sha256=sha256_file(task_ids_path),
        task_catalog_sha256=sha256_file(task_catalog_path),
        skillsbench_version=str(catalog.get("skillsbench_version", "")),
        skillsbench_commit=str(catalog.get("github_commit", "")),
        huggingface_revision=str(catalog.get("huggingface_revision", "")),
        tasks=tasks,
    )

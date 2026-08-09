"""Byte-stable rendering of the three retrieved-Skill conditions."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Literal

from skill_organization.models import TaskInput
from skill_organization.organizer import (
    HierarchyGroup,
    OrganizationBundle,
    reading_order,
    validate_bundle,
)


SkillCondition = Literal["flat_top8", "hierarchy_top8", "graph_top8"]
SKILL_CONDITIONS: tuple[SkillCondition, ...] = (
    "flat_top8",
    "hierarchy_top8",
    "graph_top8",
)
ATOMIC_MARKER = "## Atomic skill payloads\n"
FRONTMATTER = """---
name: retrieved-skills
description: Retrieved procedural skills organized for the current task.
---
"""


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_json(value: object) -> str:
    canonical = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return sha256_bytes(canonical)


def render_atomic_section(task: TaskInput) -> str:
    blocks: list[str] = []
    for item in task.skills:
        blocks.append(
            f"### {item.alias}\n\n"
            f"Retrieval rank: {item.rank}\n\n"
            "<original_skill>\n"
            f"Name: {item.record.name}\n\n"
            "Description:\n"
            f"{item.record.description}\n\n"
            "Body:\n"
            f"{item.record.body}\n"
            "</original_skill>\n"
        )
    return ATOMIC_MARKER + "\n" + "\n".join(blocks)


def _render_hierarchy_group(group: HierarchyGroup, *, depth: int) -> list[str]:
    lines = [f"{'  ' * (depth - 1)}- {group.label}"]
    for alias in group.skills:
        lines.append(f"{'  ' * depth}- {alias}")
    for child in group.children:
        lines.extend(_render_hierarchy_group(child, depth=depth + 1))
    return lines


def _render_index(
    task: TaskInput, bundle: OrganizationBundle, condition: SkillCondition
) -> str:
    names = {skill.alias: skill.record.name for skill in task.skills}
    if condition == "flat_top8":
        lines = ["## Organization: Flat retrieval order", ""]
        lines.extend(f"{skill.alias}. {skill.record.name}" for skill in task.skills)
        return "\n".join(lines) + "\n"
    if condition == "hierarchy_top8":
        lines = ["## Organization: Hierarchy", ""]
        for root in bundle.hierarchy.roots:
            lines.extend(_render_hierarchy_group(root, depth=1))
        lines.extend(("", "Alias reference:"))
        lines.extend(f"- {alias}: {names[alias]}" for alias in names)
        return "\n".join(lines) + "\n"

    order = reading_order(task, bundle.graph)
    lines = ["## Organization: Evidence graph", "", "Recommended reading order:"]
    lines.append(" -> ".join(order))
    lines.extend(("", "Validated relations:"))
    if bundle.graph.edges:
        lines.extend(
            f"- {edge.source} --{edge.edge_type}--> {edge.target}"
            for edge in bundle.graph.edges
        )
    else:
        lines.append(
            "- No supported cross-Skill relations were found; use retrieval order."
        )
    lines.extend(("", "Alias reference:"))
    lines.extend(f"- {alias}: {names[alias]}" for alias in names)
    return "\n".join(lines) + "\n"


def render_context(
    task: TaskInput, bundle: OrganizationBundle, condition: SkillCondition
) -> str:
    if condition not in SKILL_CONDITIONS:
        raise ValueError(f"unsupported Skill condition: {condition}")
    validate_bundle(task, bundle)
    introduction = (
        "\n# Retrieved procedural skills\n\n"
        "Use these documents as optional procedural guidance. Select and apply only guidance "
        "that is relevant to the task. The organization index does not change retrieval rank "
        "or the underlying Skill text.\n\n"
    )
    return (
        FRONTMATTER
        + introduction
        + _render_index(task, bundle, condition)
        + "\n"
        + render_atomic_section(task)
    )


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_bytes(payload)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _write_json_atomic(path: Path, value: object) -> None:
    payload = (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    _write_bytes_atomic(path, payload)


def write_skill_packages(
    task: TaskInput, bundle: OrganizationBundle, output_root: Path
) -> dict[str, Path]:
    validate_bundle(task, bundle)
    atomic_bytes = render_atomic_section(task).encode("utf-8")
    atomic_hash = sha256_bytes(atomic_bytes)
    hierarchy_hash = sha256_json(bundle.hierarchy.model_dump(mode="json"))
    graph_hash = sha256_json(bundle.graph.model_dump(mode="json"))
    written: dict[str, Path] = {}
    for condition in SKILL_CONDITIONS:
        rendered = render_context(task, bundle, condition).encode("utf-8")
        condition_root = output_root / task.task_id / condition
        skill_path = condition_root / "skills" / "retrieved-skills" / "SKILL.md"
        manifest_path = condition_root / "context_manifest.json"
        _write_bytes_atomic(skill_path, rendered)
        _write_json_atomic(
            manifest_path,
            {
                "schema_version": "skill-context-v1",
                "task_key": task.task_key,
                "task_id": task.task_id,
                "condition": condition,
                "rendered_context_sha256": sha256_bytes(rendered),
                "atomic_payload_sha256": atomic_hash,
                "hierarchy_sha256": hierarchy_hash,
                "graph_sha256": graph_hash,
                "skills": [
                    {
                        "alias": skill.alias,
                        "rank": skill.rank,
                        "skill_id": skill.record.skill_id,
                        "source": skill.record.source,
                        "canonical_record_sha256": skill.record.canonical_hash(),
                    }
                    for skill in task.skills
                ],
            },
        )
        written[condition] = skill_path
    return written

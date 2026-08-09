"""Fairness and leakage gates for rendered Skill packages."""

from __future__ import annotations

import json
from pathlib import Path

from skill_organization.models import TaskInput
from skill_organization.organizer import OrganizationBundle, validate_bundle
from skill_organization.render import (
    ATOMIC_MARKER,
    SKILL_CONDITIONS,
    render_context,
    sha256_bytes,
    sha256_json,
)


MAX_ORGANIZATION_INDEX_BYTES = 16_384


def validate_rendered_task(
    task: TaskInput, bundle: OrganizationBundle, output_root: Path
) -> dict[str, object]:
    validate_bundle(task, bundle)
    marker = ATOMIC_MARKER.encode("utf-8")
    suffixes: set[bytes] = set()
    context_hashes: dict[str, str] = {}
    payloads: dict[str, bytes] = {}
    atomic_payloads: dict[str, bytes] = {}
    manifests: dict[str, dict[str, object]] = {}
    expected_manifest_skills = [
        {
            "alias": skill.alias,
            "rank": skill.rank,
            "skill_id": skill.record.skill_id,
            "source": skill.record.source,
            "canonical_record_sha256": skill.record.canonical_hash(),
        }
        for skill in task.skills
    ]

    for condition in SKILL_CONDITIONS:
        condition_root = output_root / task.task_id / condition
        skill_path = condition_root / "skills" / "retrieved-skills" / "SKILL.md"
        manifest_path = condition_root / "context_manifest.json"
        payload = skill_path.read_bytes()
        expected_payload = render_context(task, bundle, condition).encode("utf-8")
        if payload != expected_payload:
            raise ValueError(
                f"rendered context differs from reviewed bundle for {condition}"
            )
        if payload.count(marker) != 1:
            raise ValueError(f"{condition} must contain one atomic payload marker")
        suffix = marker + payload.split(marker, 1)[1]
        suffixes.add(suffix)
        payloads[condition] = payload
        atomic_payloads[condition] = suffix

        text = payload.decode("utf-8")
        wrapper = text.split(ATOMIC_MARKER, 1)[0]
        if len(wrapper.encode("utf-8")) > MAX_ORGANIZATION_INDEX_BYTES:
            raise ValueError(f"organization index is too large for {condition}")
        for skill in task.skills:
            heading = f"### {skill.alias}\n"
            if text.split(ATOMIC_MARKER, 1)[1].count(heading) != 1:
                raise ValueError(f"{condition} must contain {skill.alias} exactly once")
            if skill.record.skill_id in text:
                raise ValueError(
                    f"hidden Skill ID leaked into {condition}: {skill.alias}"
                )
        lowered_wrapper = wrapper.casefold()
        forbidden_provenance = (
            "ground truth",
            "distractor",
            "source: pool",
            "source: gt",
            "source: distractor",
        )
        if any(value in lowered_wrapper for value in forbidden_provenance):
            raise ValueError(
                f"Skill provenance leaked into {condition} organization index"
            )

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            raise ValueError(f"context manifest must be an object for {condition}")
        manifests[condition] = manifest

    if len(suffixes) != 1:
        raise ValueError("atomic payload differs across Skill conditions")

    for condition in SKILL_CONDITIONS:
        payload = payloads[condition]
        suffix = atomic_payloads[condition]
        manifest = manifests[condition]
        rendered_hash = sha256_bytes(payload)
        atomic_hash = sha256_bytes(suffix)
        if manifest.get("rendered_context_sha256") != rendered_hash:
            raise ValueError(f"rendered context hash mismatch for {condition}")
        if manifest.get("atomic_payload_sha256") != atomic_hash:
            raise ValueError(f"atomic payload hash mismatch for {condition}")
        if manifest.get("hierarchy_sha256") != sha256_json(
            bundle.hierarchy.model_dump(mode="json")
        ):
            raise ValueError(f"hierarchy hash mismatch for {condition}")
        if manifest.get("graph_sha256") != sha256_json(
            bundle.graph.model_dump(mode="json")
        ):
            raise ValueError(f"graph hash mismatch for {condition}")
        if manifest.get("skills") != expected_manifest_skills:
            raise ValueError(f"private Skill manifest mismatch for {condition}")
        context_hashes[condition] = rendered_hash
    return {
        "task_id": task.task_id,
        "conditions": len(SKILL_CONDITIONS),
        "atomic_suffixes": len(suffixes),
        "context_sha256": context_hashes,
    }

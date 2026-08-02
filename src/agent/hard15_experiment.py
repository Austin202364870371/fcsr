"""Fingerprinting, checkpoint validation, and planning-only Hard-15 metrics."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Iterable
from statistics import fmean
from typing import Any

from agent.hard15_pilot import Hard15Evaluation
from agent.hard15_planning import PlanningAttempt


def experiment_fingerprint(
    model: str,
    max_skills: int,
    body_char_budget: int,
    source_revision: str,
    *,
    max_groups: int = 4,
    input_digest: str = "",
) -> str:
    payload = {
        "schema": 2,
        "model": model,
        "max_skills": max_skills,
        "body_char_budget": body_char_budget,
        "max_groups": max_groups,
        "source_revision": source_revision,
        "temperature": 0,
        "input_digest": input_digest,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def compatible_completed(
    records: Iterable[PlanningAttempt], fingerprint: str
) -> dict[tuple[str, str], PlanningAttempt]:
    completed = {}
    for record in records:
        if record.fingerprint != fingerprint:
            raise ValueError("checkpoint fingerprint differs from this experiment")
        if record.valid:
            completed[(record.task_id, record.method)] = record
    return completed


def evaluate_attempts(
    attempts: Iterable[PlanningAttempt],
    evaluations: Iterable[Hard15Evaluation],
) -> dict[str, Any]:
    rows = list(attempts)
    private = {item.task_id: item for item in evaluations}
    per_task = []
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for attempt in rows:
        evaluation = private[attempt.task_id]
        planned_aliases = (
            set(attempt.plan.selected_skill_aliases)
            if attempt.valid and attempt.plan is not None
            else set()
        )
        planned_ids = {
            evaluation.alias_to_skill_id[alias]
            for alias in planned_aliases
            if alias in evaluation.alias_to_skill_id
        }
        gt = set(evaluation.gt_skill_ids)
        coverage = len(gt & planned_ids) / len(gt)
        row = {
            "task_id": attempt.task_id,
            "method": attempt.method,
            "valid_plan": attempt.valid,
            "selected_gt_coverage": coverage,
            "complete_gt_coverage": gt <= planned_ids,
            "planned_skill_count": len(planned_aliases),
            "presented_skill_count": len(attempt.selected_candidate_aliases),
            "prompt_tokens": attempt.prompt_tokens,
            "completion_tokens": attempt.completion_tokens,
            "rendered_characters": attempt.rendered_characters,
            "omitted_count": attempt.omitted_count,
            **{
                key: attempt.organization_metadata.get(key, 0)
                for key in (
                    "group_count",
                    "node_count",
                    "edge_count",
                    "explicit_reference_count",
                    "namespace_edge_count",
                    "connected_component_count",
                )
            },
        }
        per_task.append(row)
        grouped[attempt.method].append(row)
    methods = {}
    for method, values in sorted(grouped.items()):
        methods[method] = {
            "attempts": len(values),
            "valid_plan_rate": _mean(values, "valid_plan"),
            "mean_selected_gt_coverage": _mean(values, "selected_gt_coverage"),
            "complete_gt_coverage_rate": _mean(values, "complete_gt_coverage"),
            "mean_planned_skill_count": _mean(values, "planned_skill_count"),
            "mean_presented_skill_count": _mean(values, "presented_skill_count"),
            "mean_prompt_tokens": _mean(values, "prompt_tokens"),
            "mean_completion_tokens": _mean(values, "completion_tokens"),
            "mean_rendered_characters": _mean(values, "rendered_characters"),
            "mean_omitted_count": _mean(values, "omitted_count"),
        }
        if method == "hierarchy":
            methods[method]["mean_group_count"] = _mean(values, "group_count")
        if method == "graph":
            for key in (
                "node_count",
                "edge_count",
                "explicit_reference_count",
                "namespace_edge_count",
                "connected_component_count",
            ):
                methods[method][f"mean_{key}"] = _mean(values, key)
    return {
        "result_type": "planning_only_not_task_success",
        "attempt_count": len(rows),
        "methods": methods,
        "per_task": per_task,
    }


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    return fmean(float(row[key]) for row in rows) if rows else 0.0

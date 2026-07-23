"""SkillRouter-compatible scoring over consolidated FCSR benchmark records."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from metrics import compute_all_metrics


@dataclass(frozen=True)
class EvaluationResult:
    summary: dict[str, dict[str, float]]
    details: list[dict[str, Any]]
    skipped_generic_only: int
    skipped_missing_prediction: int
    skipped_no_gt_in_pool: int


def evaluate_predictions(
    tasks: list[dict[str, Any]],
    predictions: dict[str, Any],
    pool_ids: set[str],
) -> EvaluationResult:
    by_stratum: dict[str, list[dict[str, float]]] = {
        "all": [],
        "single": [],
        "multi": [],
    }
    details = []
    skipped_generic = 0
    skipped_missing = 0
    skipped_no_gt = 0

    for task in tasks:
        task_id = task.get("task_id") or task.get("query_id")
        if not isinstance(task_id, str) or not task_id:
            raise ValueError("each task must have task_id or query_id")
        if task.get("generic_only") is True or task.get("task_type") == "generic_only":
            skipped_generic += 1
            continue
        gt_ids = _ground_truth_ids(task)
        gt_in_pool = gt_ids & pool_ids
        if not gt_in_pool:
            skipped_no_gt += 1
            continue
        if task_id not in predictions:
            skipped_missing += 1
            continue

        ranked_ids = _prediction_ids(predictions[task_id])
        ranked_in_pool = []
        seen = set()
        for skill_id in ranked_ids:
            if skill_id in pool_ids and skill_id not in seen:
                seen.add(skill_id)
                ranked_in_pool.append(skill_id)
        relevance = task.get("relevance", {})
        tier_relevance = (
            {
                skill_id: float(value)
                for skill_id, value in relevance.items()
                if skill_id in pool_ids
            }
            if isinstance(relevance, dict)
            else {}
        )
        metrics = compute_all_metrics(
            ranked_in_pool,
            gt_in_pool,
            tier_relevance or None,
        )
        stratum = "single" if len(gt_ids) == 1 else "multi"
        by_stratum["all"].append(metrics)
        by_stratum[stratum].append(metrics)
        details.append(
            {
                "task_id": task_id,
                "stratum": stratum,
                "gt_skill_ids": sorted(gt_in_pool),
                "ranked_skill_ids": ranked_in_pool,
                "metrics": metrics,
            }
        )

    summary = {
        stratum: _aggregate(values)
        for stratum, values in by_stratum.items()
        if values
    }
    return EvaluationResult(
        summary=summary,
        details=details,
        skipped_generic_only=skipped_generic,
        skipped_missing_prediction=skipped_missing,
        skipped_no_gt_in_pool=skipped_no_gt,
    )


def _ground_truth_ids(task: dict[str, Any]) -> set[str]:
    for field in (
        "core_gt_ids",
        "core_gold_skill_ids",
        "gt_skill_ids",
        "gold_skill_ids",
    ):
        values = task.get(field)
        if isinstance(values, list):
            return {
                value for value in values if isinstance(value, str) and value
            }
    return set()


def _prediction_ids(value: Any) -> list[str]:
    if isinstance(value, list):
        candidates = value
    elif isinstance(value, dict):
        candidates = (
            value.get("ranked_skill_ids")
            or value.get("skill_ids")
            or value.get("predictions")
            or []
        )
    else:
        raise ValueError("prediction must be a ranked list or prediction object")
    result = []
    for item in candidates:
        if isinstance(item, str):
            result.append(item)
        elif isinstance(item, dict) and isinstance(item.get("skill_id"), str):
            result.append(item["skill_id"])
    return result


def _aggregate(values: list[dict[str, float]]) -> dict[str, float]:
    metrics = {
        key: sum(item[key] for item in values) / len(values)
        for key in values[0]
    }
    metrics["count"] = len(values)
    return metrics

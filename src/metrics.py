"""Retrieval metrics aligned with the public SkillRouter evaluation."""

from __future__ import annotations

import math


def dcg_at_k(relevances: list[float], k: int) -> float:
    return sum(value / math.log2(index + 2) for index, value in enumerate(relevances[:k]))


def ndcg_at_k(relevances: list[float], ideal_relevances: list[float], k: int) -> float:
    ideal = dcg_at_k(sorted(ideal_relevances, reverse=True), k)
    return dcg_at_k(relevances, k) / ideal if ideal > 0 else 0.0


def compute_all_metrics(
    ranked_ids: list[str],
    gt_skill_ids: set[str],
    relevance_map: dict[str, float] | None = None,
) -> dict[str, float]:
    if relevance_map:
        relevances = [float(relevance_map.get(skill_id, 0.0)) for skill_id in ranked_ids]
        ideal_relevances = [float(value) for value in relevance_map.values()]
    else:
        relevances = [1.0 if skill_id in gt_skill_ids else 0.0 for skill_id in ranked_ids]
        ideal_relevances = [1.0] * len(gt_skill_ids)

    return {
        "nDCG@1": ndcg_at_k(relevances, ideal_relevances, 1),
        "nDCG@3": ndcg_at_k(relevances, ideal_relevances, 3),
        "nDCG@10": ndcg_at_k(relevances, ideal_relevances, 10),
        "Hit@1": _hit_at_k(ranked_ids, gt_skill_ids, 1),
        "Precision@3": _precision_at_k(ranked_ids, gt_skill_ids, 3),
        "MRR@10": _mrr_at_k(ranked_ids, gt_skill_ids, 10),
        "Recall@10": _recall_at_k(ranked_ids, gt_skill_ids, 10),
        "Recall@20": _recall_at_k(ranked_ids, gt_skill_ids, 20),
        "Recall@50": _recall_at_k(ranked_ids, gt_skill_ids, 50),
        "FullCoverage@1": _full_coverage_at_k(ranked_ids, gt_skill_ids, 1),
        "FullCoverage@3": _full_coverage_at_k(ranked_ids, gt_skill_ids, 3),
        "FullCoverage@5": _full_coverage_at_k(ranked_ids, gt_skill_ids, 5),
        "FullCoverage@10": _full_coverage_at_k(ranked_ids, gt_skill_ids, 10),
    }


def _hit_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int) -> float:
    return 1.0 if set(ranked_ids[:k]) & relevant_ids else 0.0


def _precision_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int) -> float:
    return sum(skill_id in relevant_ids for skill_id in ranked_ids[:k]) / k if k else 0.0


def _mrr_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int) -> float:
    for index, skill_id in enumerate(ranked_ids[:k], start=1):
        if skill_id in relevant_ids:
            return 1.0 / index
    return 0.0


def _recall_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    return len(set(ranked_ids[:k]) & relevant_ids) / len(relevant_ids)


def _full_coverage_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if not relevant_ids:
        return 1.0
    return 1.0 if relevant_ids.issubset(set(ranked_ids[:k])) else 0.0

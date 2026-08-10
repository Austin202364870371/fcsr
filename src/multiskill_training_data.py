"""Deterministic mixed single- and multi-Skill training data builders."""

from __future__ import annotations

import hashlib
import random
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from itertools import chain
from typing import Any

from modeling import build_reranker_groups
from preprocessing import filter_identity_and_overlap, normalize_category
from retrieval import BM25Index


@dataclass(frozen=True)
class MixedTrainingBuildResult:
    biencoder_records: list[dict[str, Any]]
    reranker_groups: list[dict[str, Any]]
    compositional_query_count: int
    compositional_biencoder_examples: int
    compositional_reranker_groups: int


def build_mixed_training_records(
    single_biencoder_records: Iterable[dict[str, Any]],
    single_reranker_groups: Iterable[dict[str, Any]],
    compositional_records: Iterable[dict[str, Any]],
    skills: Iterable[dict[str, Any]],
    semantic_candidates: Mapping[str, list[dict[str, Any]]],
    *,
    multiplier: int,
    seed: int,
    overlap_threshold: float = 0.85,
    progress: Any | None = None,
) -> MixedTrainingBuildResult:
    """Build mixed data while treating every Skill in a composition as positive."""
    if multiplier <= 0:
        raise ValueError("multiplier must be positive")

    skill_records = _valid_skills(skills)
    lookup = {record["skill_id"]: record for record in skill_records}
    compositional = list(compositional_records)
    mined = _mine_compositional_negatives(
        compositional,
        skill_records,
        semantic_candidates,
        seed=seed,
        overlap_threshold=overlap_threshold,
        progress=progress,
    )

    biencoder_records = list(single_biencoder_records)
    reranker_groups = list(single_reranker_groups)
    compositional_biencoder_examples = 0
    compositional_reranker_groups = 0
    expanded_reranker_records: list[dict[str, Any]] = []

    for record in compositional:
        query_id = _required_query_id(record)
        positive_ids = _positive_ids(record, lookup)
        negatives = mined[query_id]
        for repeat_index in range(1, multiplier + 1):
            replica_id = _replica_query_id(query_id, repeat_index)
            for positive_index, positive_id in enumerate(positive_ids, start=1):
                biencoder_records.append(
                    {
                        "query_id": f"{replica_id}::positive-{positive_index}",
                        "source_query_id": query_id,
                        "repeat_index": repeat_index,
                        "query": record.get("query"),
                        "positive_skill_id": positive_id,
                        "positive_skill_ids": list(positive_ids),
                        "source_hashes": record.get("source_hashes", []),
                        "negative_candidates": negatives,
                    }
                )
                compositional_biencoder_examples += 1
            expanded_reranker_records.append(
                {
                    "query_id": replica_id,
                    "source_query_id": query_id,
                    "repeat_index": repeat_index,
                    "query": record.get("query"),
                    "positive_skill_ids": list(positive_ids),
                    "retrieved_candidates": [
                        {"skill_id": skill_id, "score": 1.0}
                        for skill_id in positive_ids
                    ]
                    + negatives,
                }
            )
            compositional_reranker_groups += 1

    compositional_groups = build_reranker_groups(
        expanded_reranker_records,
        skill_records,
        top_k=20,
    )
    if compositional_groups.dropped_no_positive:
        raise ValueError("compositional reranker group lost all positives")
    reranker_groups.extend(compositional_groups.groups)
    return MixedTrainingBuildResult(
        biencoder_records=biencoder_records,
        reranker_groups=reranker_groups,
        compositional_query_count=len(compositional),
        compositional_biencoder_examples=compositional_biencoder_examples,
        compositional_reranker_groups=compositional_reranker_groups,
    )


def _mine_compositional_negatives(
    queries: list[dict[str, Any]],
    skills: list[dict[str, Any]],
    semantic_candidates: Mapping[str, list[dict[str, Any]]],
    *,
    seed: int,
    overlap_threshold: float,
    progress: Any | None,
) -> dict[str, list[dict[str, Any]]]:
    if not 0 <= overlap_threshold <= 1:
        raise ValueError("overlap_threshold must be between 0 and 1")
    lookup = {record["skill_id"]: record for record in skills}
    documents = [_skill_search_text(record) for record in skills]
    bm25 = BM25Index(documents)
    stable_pool_order = sorted(range(len(skills)), key=lambda index: skills[index]["skill_id"])
    category_members: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in skills:
        category_members[normalize_category(record)].append(record)
    for members in category_members.values():
        members.sort(key=lambda record: record["skill_id"])

    result: dict[str, list[dict[str, Any]]] = {}
    for query in queries:
        query_id = _required_query_id(query)
        if query_id not in semantic_candidates:
            raise ValueError(f"semantic candidates missing for query: {query_id!r}")
        positive_ids = _positive_ids(query, lookup)
        positives = [lookup[skill_id] for skill_id in positive_ids]
        scores, bm25_head = bm25.rank(str(query.get("query", "")), limit=256)
        head_set = set(bm25_head)
        bm25_order = chain(
            bm25_head,
            (index for index in stable_pool_order if index not in head_set),
        )
        same_category = [
            candidate
            for category in sorted({normalize_category(positive) for positive in positives})
            for candidate in category_members[category]
        ]
        random.Random(_stable_seed(seed, f"{query_id}::category")).shuffle(same_category)
        random_order = _lazy_random_order(
            skills,
            _stable_seed(seed, f"{query_id}::random"),
        )
        source_candidates: list[tuple[str, int, Iterable[tuple[dict[str, Any], float]]]] = [
            (
                "semantic",
                4,
                (
                    (lookup[item["skill_id"]], float(item.get("score", 0.0)))
                    for item in semantic_candidates[query_id]
                    if isinstance(item.get("skill_id"), str) and item["skill_id"] in lookup
                ),
            ),
            ("bm25", 3, ((skills[index], float(scores[index])) for index in bm25_order)),
            ("same_category", 2, ((candidate, 0.0) for candidate in same_category)),
            ("random", 1, ((candidate, 0.0) for candidate in random_order)),
        ]

        selected: list[dict[str, Any]] = []
        selected_ids: set[str] = set(positive_ids)
        for source, target, candidates in source_candidates:
            added = 0
            for candidate, score in candidates:
                candidate_id = candidate["skill_id"]
                if candidate_id in selected_ids:
                    continue
                if any(
                    filter_identity_and_overlap(
                        positive,
                        [candidate],
                        threshold=overlap_threshold,
                    ).removed
                    for positive in positives
                ):
                    continue
                selected.append(
                    {"skill_id": candidate_id, "source": source, "score": float(score)}
                )
                selected_ids.add(candidate_id)
                added += 1
                if added >= target:
                    break
        result[query_id] = selected
        if progress is not None:
            progress(1)
    return result


def _valid_skills(skills: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    records = [
        record
        for record in skills
        if isinstance(record.get("skill_id"), str) and record["skill_id"]
    ]
    if not records:
        raise ValueError("skills must contain at least one valid skill_id")
    duplicate_ids = len(records) - len({record["skill_id"] for record in records})
    if duplicate_ids:
        raise ValueError("skills contain duplicate skill_ids")
    return records


def _positive_ids(record: Mapping[str, Any], lookup: Mapping[str, dict[str, Any]]) -> list[str]:
    values = record.get("positive_skill_ids")
    if not isinstance(values, list) or not values:
        raise ValueError("compositional record must contain positive_skill_ids")
    positive_ids = [value for value in values if isinstance(value, str) and value]
    if len(positive_ids) != len(values) or len(set(positive_ids)) != len(positive_ids):
        raise ValueError("positive_skill_ids must be unique non-empty strings")
    for skill_id in positive_ids:
        if skill_id not in lookup:
            raise ValueError(f"positive skill not found in pool: {skill_id!r}")
    return positive_ids


def _required_query_id(record: Mapping[str, Any]) -> str:
    query_id = record.get("query_id")
    if not isinstance(query_id, str) or not query_id:
        raise ValueError("record must contain a non-empty query_id")
    return query_id


def _replica_query_id(query_id: str, repeat_index: int) -> str:
    return f"{query_id}::repeat-{repeat_index}"


def _skill_search_text(skill: Mapping[str, Any]) -> str:
    return "\n".join(str(skill.get(field, "")) for field in ("name", "description", "body"))


def _stable_seed(seed: int, value: str) -> int:
    digest = hashlib.sha256(f"{seed}:{value}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _lazy_random_order(records: list[dict[str, Any]], seed: int) -> Iterable[dict[str, Any]]:
    rng = random.Random(seed)
    visited: set[int] = set()
    while len(visited) < len(records):
        index = rng.randrange(len(records))
        if index not in visited:
            visited.add(index)
            yield records[index]

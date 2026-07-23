"""Tensor-free retrieval and negative-source composition utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np


@dataclass(frozen=True)
class EmbeddingFilterResult:
    kept: list[dict[str, Any]]
    removed: list[dict[str, Any]]


def semantic_topk(
    query_embeddings: np.ndarray,
    skill_embeddings: np.ndarray,
    k: int,
    device: str | None = None,
    query_batch_size: int = 128,
    progress: Callable[[int], None] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    queries = _normalized_matrix(query_embeddings, "query_embeddings")
    skills = _normalized_matrix(skill_embeddings, "skill_embeddings")
    if queries.shape[1] != skills.shape[1]:
        raise ValueError("query and skill embedding dimensions must match")
    if k <= 0 or query_batch_size <= 0:
        raise ValueError("k and query_batch_size must be positive")
    k = min(k, skills.shape[0])
    if device and device.startswith("cuda"):
        return _semantic_topk_torch(queries, skills, k, device, query_batch_size, progress)

    index_batches = []
    score_batches = []
    skill_indices = np.arange(skills.shape[0])
    for start in range(0, queries.shape[0], query_batch_size):
        similarities = queries[start : start + query_batch_size] @ skills.T
        batch_indices = np.empty((similarities.shape[0], k), dtype=np.int64)
        batch_scores = np.empty((similarities.shape[0], k), dtype=np.float32)
        for row, values in enumerate(similarities):
            order = np.lexsort((skill_indices, -values))[:k]
            batch_indices[row] = order
            batch_scores[row] = values[order]
        index_batches.append(batch_indices)
        score_batches.append(batch_scores)
        if progress is not None:
            progress(len(similarities))
    return np.concatenate(index_batches), np.concatenate(score_batches)


def _semantic_topk_torch(
    queries: np.ndarray,
    skills: np.ndarray,
    k: int,
    device: str,
    query_batch_size: int,
    progress: Callable[[int], None] | None,
) -> tuple[np.ndarray, np.ndarray]:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch is required for CUDA semantic search") from exc
    skill_tensor = torch.as_tensor(skills, device=device)
    index_batches = []
    score_batches = []
    with torch.no_grad():
        for start in range(0, queries.shape[0], query_batch_size):
            query_tensor = torch.as_tensor(
                queries[start : start + query_batch_size],
                device=device,
            )
            scores, indices = torch.topk(query_tensor @ skill_tensor.T, k=k, dim=1)
            index_batches.append(indices.cpu().numpy())
            score_batches.append(scores.float().cpu().numpy())
            if progress is not None:
                progress(len(query_tensor))
    return np.concatenate(index_batches), np.concatenate(score_batches)

def merge_negative_sources(
    local_record: dict[str, Any],
    semantic_candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    targets = {
        "semantic": 4,
        "bm25": 3,
        "same_category": 2,
        "random": 1,
    }
    selected: list[dict[str, Any]] = []
    seen = {local_record.get("positive_skill_id")}

    sources = {
        "semantic": [
            {
                "skill_id": item.get("skill_id"),
                "source": "semantic",
                "score": float(item.get("score", 0.0)),
            }
            for item in semantic_candidates
        ]
    }
    for source in ("bm25", "same_category", "random"):
        sources[source] = [
            item
            for item in local_record.get("negative_candidates", [])
            if item.get("source") == source
        ]

    for source in ("semantic", "bm25", "same_category", "random"):
        count = 0
        for item in sources[source]:
            skill_id = item.get("skill_id")
            if not isinstance(skill_id, str) or not skill_id or skill_id in seen:
                continue
            selected.append(
                {
                    "skill_id": skill_id,
                    "source": source,
                    "score": float(item.get("score", 0.0)),
                }
            )
            seen.add(skill_id)
            count += 1
            if count >= targets[source]:
                break

    merged = dict(local_record)
    merged["negative_candidates"] = selected
    return merged


def embedding_false_negative_filter(
    positive_embedding: np.ndarray,
    candidates: list[dict[str, Any]],
    candidate_embeddings: dict[str, np.ndarray],
    threshold: float,
) -> EmbeddingFilterResult:
    if not -1 <= threshold <= 1:
        raise ValueError("threshold must be between -1 and 1")
    positive = _normalized_vector(positive_embedding, "positive_embedding")
    kept: list[dict[str, Any]] = []
    removed: list[dict[str, Any]] = []
    for candidate in candidates:
        skill_id = candidate.get("skill_id")
        embedding = candidate_embeddings.get(skill_id)
        if embedding is None:
            raise KeyError(f"missing embedding for candidate {skill_id!r}")
        score = float(positive @ _normalized_vector(embedding, str(skill_id)))
        if score >= threshold:
            removed.append(
                {
                    "skill_id": skill_id,
                    "reason": "high_embedding_similarity",
                    "score": score,
                }
            )
        else:
            kept.append(candidate)
    return EmbeddingFilterResult(kept=kept, removed=removed)


def _normalized_matrix(value: np.ndarray, name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError(f"{name} must be a non-empty 2D array")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError(f"{name} contains a zero vector")
    return matrix / norms


def _normalized_vector(value: np.ndarray, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float32)
    if vector.ndim != 1 or vector.size == 0:
        raise ValueError(f"{name} must be a non-empty 1D array")
    norm = np.linalg.norm(vector)
    if norm == 0:
        raise ValueError(f"{name} must not be a zero vector")
    return vector / norm

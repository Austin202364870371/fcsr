"""Tensor-free retrieval and negative-source composition utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import math
from collections import defaultdict

import numpy as np


@dataclass(frozen=True)
class EmbeddingFilterResult:
    kept: list[dict[str, Any]]
    removed: list[dict[str, Any]]


class BM25Index:
    """Deterministic Unicode-aware BM25 index shared by mining and evaluation."""

    def __init__(
        self,
        documents: list[str],
        k1: float = 1.5,
        b: float = 0.75,
        progress: Callable[[int], None] | None = None,
    ) -> None:
        self.k1 = k1
        self.b = b
        tokenized = []
        for document in documents:
            tokenized.append(unicode_tokens(document))
            if progress is not None:
                progress(1)
        self.lengths = np.asarray([len(document) for document in tokenized], dtype=np.float32)
        self.average_length = float(self.lengths.mean()) if len(self.lengths) else 0.0
        raw_postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for document_index, document in enumerate(tokenized):
            frequencies: dict[str, int] = defaultdict(int)
            for token in document:
                frequencies[token] += 1
            for token, frequency in frequencies.items():
                raw_postings[token].append((document_index, frequency))
        count = len(tokenized)
        self.postings: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.idf: dict[str, float] = {}
        for token, values in raw_postings.items():
            self.postings[token] = (
                np.asarray([item[0] for item in values], dtype=np.int64),
                np.asarray([item[1] for item in values], dtype=np.float32),
            )
            frequency = len(values)
            self.idf[token] = math.log(
                1 + (count - frequency + 0.5) / (frequency + 0.5)
            )

    def rank(self, query: str, limit: int) -> tuple[np.ndarray, list[int]]:
        if limit <= 0:
            raise ValueError("limit must be positive")
        scores = np.zeros(len(self.lengths), dtype=np.float32)
        query_frequencies: dict[str, int] = defaultdict(int)
        for token in unicode_tokens(query):
            query_frequencies[token] += 1
        for token, query_frequency in query_frequencies.items():
            posting = self.postings.get(token)
            if posting is None:
                continue
            indices, frequencies = posting
            normalization = (
                1 - self.b + self.b * self.lengths[indices] / self.average_length
                if self.average_length
                else 1.0
            )
            denominator = frequencies + self.k1 * normalization
            scores[indices] += (
                query_frequency
                * self.idf[token]
                * frequencies
                * (self.k1 + 1)
                / denominator
            )
        positive = np.flatnonzero(scores > 0)
        if len(positive) > limit:
            local = np.argpartition(-scores[positive], limit - 1)[:limit]
            positive = positive[local]
        ordered = sorted(positive.tolist(), key=lambda index: (-scores[index], index))
        return scores, ordered

    def topk(self, query: str, k: int) -> tuple[np.ndarray, list[int]]:
        if k <= 0:
            raise ValueError("k must be positive")
        k = min(k, len(self.lengths))
        scores, ordered = self.rank(query, limit=k)
        seen = set(ordered)
        ordered.extend(index for index in range(len(self.lengths)) if index not in seen)
        return scores, ordered[:k]


def reciprocal_rank_fusion(
    rankings: list[list[str]],
    top_k: int,
    rrf_k: int = 60,
) -> list[dict[str, float | str]]:
    if top_k <= 0 or rrf_k < 0:
        raise ValueError("top_k must be positive and rrf_k must be non-negative")
    scores: dict[str, float] = {}
    for ranking in rankings:
        seen: set[str] = set()
        for rank, skill_id in enumerate(ranking, start=1):
            if skill_id in seen:
                continue
            seen.add(skill_id)
            scores[skill_id] = scores.get(skill_id, 0.0) + 1.0 / (rrf_k + rank)
    ordered = sorted(scores, key=lambda skill_id: (-scores[skill_id], skill_id))
    return [
        {"skill_id": skill_id, "rrf_score": scores[skill_id]}
        for skill_id in ordered[:top_k]
    ]


def unicode_tokens(text: str) -> list[str]:
    normalized = text.casefold()
    tokens: list[str] = []
    word: list[str] = []
    cjk_run: list[str] = []

    def flush_word() -> None:
        if word:
            tokens.append("".join(word))
            word.clear()

    def flush_cjk() -> None:
        if cjk_run:
            tokens.extend(cjk_run)
            tokens.extend(
                "".join(cjk_run[index : index + 2])
                for index in range(len(cjk_run) - 1)
            )
            cjk_run.clear()

    for character in normalized:
        if _is_cjk(character):
            flush_word()
            cjk_run.append(character)
        elif character.isalnum():
            flush_cjk()
            word.append(character)
        else:
            flush_word()
            flush_cjk()
    flush_word()
    flush_cjk()
    return tokens


def _is_cjk(character: str) -> bool:
    return (
        "\u3400" <= character <= "\u4dbf"
        or "\u4e00" <= character <= "\u9fff"
        or "\u3040" <= character <= "\u30ff"
        or "\uac00" <= character <= "\ud7af"
    )

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

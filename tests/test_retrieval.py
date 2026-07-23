import unittest

import numpy as np

from retrieval import (
    embedding_false_negative_filter,
    merge_negative_sources,
    semantic_topk,
)


class RetrievalTests(unittest.TestCase):
    def test_semantic_topk_returns_cosine_ranked_indices(self) -> None:
        queries = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        skills = np.array(
            [[1.0, 0.0], [0.8, 0.2], [0.0, 1.0], [-1.0, 0.0]],
            dtype=np.float32,
        )

        indices, scores = semantic_topk(queries, skills, k=2)

        self.assertEqual(indices.tolist(), [[0, 1], [2, 1]])
        self.assertGreater(scores[0, 0], scores[0, 1])

    def test_semantic_topk_reports_processed_queries(self) -> None:
        queries = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        skills = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        processed: list[int] = []

        semantic_topk(
            queries,
            skills,
            k=1,
            query_batch_size=1,
            progress=processed.append,
        )

        self.assertEqual(processed, [1, 1])

    def test_source_merge_preserves_targets_and_uniqueness(self) -> None:
        local = {
            "query_id": "q",
            "query": "task",
            "positive_skill_id": "p",
            "negative_candidates": [
                *[
                    {"skill_id": f"b{index}", "source": "bm25", "score": 1.0}
                    for index in range(3)
                ],
                *[
                    {
                        "skill_id": f"c{index}",
                        "source": "same_category",
                        "score": 0.0,
                    }
                    for index in range(2)
                ],
                {"skill_id": "r0", "source": "random", "score": 0.0},
            ],
            "filtered": [],
        }
        semantic = [
            {"skill_id": f"s{index}", "score": 1.0 - index / 10}
            for index in range(6)
        ]

        merged = merge_negative_sources(local, semantic)

        counts = {
            source: sum(
                item["source"] == source
                for item in merged["negative_candidates"]
            )
            for source in ("semantic", "bm25", "same_category", "random")
        }
        self.assertEqual(
            counts,
            {"semantic": 4, "bm25": 3, "same_category": 2, "random": 1},
        )
        ids = [item["skill_id"] for item in merged["negative_candidates"]]
        self.assertEqual(len(ids), len(set(ids)))

    def test_embedding_filter_records_similarity_reason(self) -> None:
        positive = np.array([1.0, 0.0], dtype=np.float32)
        candidates = [
            {"skill_id": "near", "source": "semantic", "score": 0.9},
            {"skill_id": "far", "source": "bm25", "score": 0.1},
        ]
        embeddings = {
            "near": np.array([0.99, 0.01], dtype=np.float32),
            "far": np.array([0.0, 1.0], dtype=np.float32),
        }

        result = embedding_false_negative_filter(
            positive,
            candidates,
            embeddings,
            threshold=0.95,
        )

        self.assertEqual([item["skill_id"] for item in result.kept], ["far"])
        self.assertEqual(result.removed[0]["skill_id"], "near")
        self.assertEqual(result.removed[0]["reason"], "high_embedding_similarity")
        self.assertGreater(result.removed[0]["score"], 0.95)


if __name__ == "__main__":
    unittest.main()

import unittest

from modeling import build_biencoder_examples
from rq1_training_data import build_mixed_training_records


def skill(skill_id: str, category: str = "development") -> dict:
    return {
        "skill_id": skill_id,
        "name": skill_id,
        "description": f"Description for {skill_id}",
        "body": f"Workflow body for {skill_id}.",
        "category": category,
    }


class MultiPositiveBiEncoderTests(unittest.TestCase):
    def test_secondary_positive_is_excluded_from_negative_documents(self) -> None:
        skills = [skill("dev/a"), skill("dev/b"), skill("dev/c")]
        record = {
            "query_id": "compq::1",
            "query": "Complete both workflows.",
            "positive_skill_id": "dev/a",
            "positive_skill_ids": ["dev/a", "dev/b"],
            "negative_candidates": [
                {"skill_id": "dev/b", "source": "semantic"},
                {"skill_id": "dev/c", "source": "semantic"},
            ],
        }

        example = build_biencoder_examples([record], skills)[0]

        self.assertEqual(example["negative_skill_ids"], ["dev/c"])


class Rq1MixedTrainingDataTests(unittest.TestCase):
    def setUp(self) -> None:
        self.skills = [
            skill("dev/a"),
            skill("dev/b"),
            skill("dev/c"),
            skill("dev/d"),
            skill("design/e", "design"),
            skill("design/f", "design"),
            skill("other/g", "other"),
            skill("other/h", "other"),
        ]
        self.single_biencoder = [
            {
                "query_id": "syn::single",
                "query": "Single workflow.",
                "positive_skill_id": "dev/a",
                "negative_candidates": [{"skill_id": "dev/c", "source": "semantic"}],
            }
        ]
        self.single_reranker = [
            {
                "query_id": "syn::single",
                "query": "Single workflow.",
                "candidates": [
                    {"skill_id": "dev/a", "rank": 1, "retrieval_score": 1.0, "label": 1},
                    {"skill_id": "dev/c", "rank": 2, "retrieval_score": 0.5, "label": 0},
                ],
                "positive_mask": [True, False],
            }
        ]
        self.compositional = [
            {
                "query_id": "compq::pair",
                "query": "Complete workflow A and then workflow B.",
                "positive_skill_ids": ["dev/a", "dev/b"],
                "source_hashes": ["a", "b"],
            }
        ]
        self.semantic = {
            "compq::pair": [
                {"skill_id": "dev/b", "score": 0.99},
                {"skill_id": "dev/c", "score": 0.80},
                {"skill_id": "dev/d", "score": 0.70},
                {"skill_id": "design/e", "score": 0.60},
                {"skill_id": "design/f", "score": 0.50},
                {"skill_id": "other/g", "score": 0.40},
                {"skill_id": "other/h", "score": 0.30},
            ]
        }

    def test_expands_multi_positive_examples_and_preserves_group_labels(self) -> None:
        result = build_mixed_training_records(
            self.single_biencoder,
            self.single_reranker,
            self.compositional,
            self.skills,
            self.semantic,
            multiplier=3,
            seed=42,
        )

        self.assertEqual(len(result.biencoder_records), 7)
        self.assertEqual(len(result.reranker_groups), 4)
        expanded = result.biencoder_records[1:]
        self.assertEqual({record["positive_skill_id"] for record in expanded}, {"dev/a", "dev/b"})
        for record in expanded:
            self.assertEqual(record["positive_skill_ids"], ["dev/a", "dev/b"])
            negative_ids = {item["skill_id"] for item in record["negative_candidates"]}
            self.assertFalse(negative_ids & {"dev/a", "dev/b"})

        groups = result.reranker_groups[1:]
        for group in groups:
            labels = {
                item["skill_id"]: item["label"]
                for item in group["candidates"]
            }
            self.assertEqual(labels["dev/a"], 1)
            self.assertEqual(labels["dev/b"], 1)
            self.assertEqual(sum(group["positive_mask"]), 2)

    def test_is_deterministic(self) -> None:
        first = build_mixed_training_records(
            self.single_biencoder, self.single_reranker, self.compositional,
            self.skills, self.semantic, multiplier=3, seed=42,
        )
        second = build_mixed_training_records(
            self.single_biencoder, self.single_reranker, self.compositional,
            self.skills, self.semantic, multiplier=3, seed=42,
        )

        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
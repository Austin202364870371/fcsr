import unittest

from evaluation import evaluate_predictions


class EvaluationTests(unittest.TestCase):
    def test_matches_core_protocol_and_aggregates_single_multi(self) -> None:
        tasks = [
            {
                "query_id": "generic",
                "task_type": "generic_only",
                "core_gold_skill_ids": ["g"],
                "relevance": {"g": 3},
            },
            {
                "query_id": "single",
                "core_gold_skill_ids": ["p"],
                "relevance": {"p": 3, "d": 1},
            },
            {
                "query_id": "multi",
                "core_gold_skill_ids": ["p1", "p2"],
                "relevance": {"p1": 3, "p2": 3},
            },
        ]
        predictions = {
            "generic": ["g"],
            "single": ["outside", "d", "p"],
            "multi": ["p1", "p2"],
        }
        pool_ids = {"g", "p", "d", "p1", "p2"}

        result = evaluate_predictions(tasks, predictions, pool_ids)

        self.assertEqual(result.summary["all"]["count"], 2)
        self.assertEqual(result.summary["single"]["count"], 1)
        self.assertEqual(result.summary["multi"]["count"], 1)
        self.assertLess(result.summary["single"]["nDCG@3"], 1.0)
        self.assertEqual(result.summary["multi"]["FullCoverage@3"], 1.0)
        self.assertEqual(result.skipped_generic_only, 1)

    def test_intersects_ground_truth_and_relevance_with_tier_pool(self) -> None:
        tasks = [
            {
                "task_id": "multi",
                "core_gt_ids": ["p1", "p2"],
                "relevance": {"p1": 3, "p2": 3, "degraded": 1},
            }
        ]

        result = evaluate_predictions(
            tasks,
            {"multi": ["p2", "degraded", "p1"]},
            {"p1", "degraded"},
        )

        self.assertEqual(result.summary["multi"]["count"], 1)
        self.assertEqual(result.details[0]["gt_skill_ids"], ["p1"])
        self.assertEqual(result.details[0]["ranked_skill_ids"], ["degraded", "p1"])


if __name__ == "__main__":
    unittest.main()

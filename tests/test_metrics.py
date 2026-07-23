import unittest

from metrics import compute_all_metrics


class MetricsTests(unittest.TestCase):
    def test_single_skill_metrics(self) -> None:
        metrics = compute_all_metrics(["x", "gold"], {"gold"})

        self.assertEqual(metrics["Hit@1"], 0.0)
        self.assertEqual(metrics["MRR@10"], 0.5)
        self.assertEqual(metrics["Recall@10"], 1.0)

    def test_multi_skill_full_coverage(self) -> None:
        metrics = compute_all_metrics(["a", "x", "b"], {"a", "b"})

        self.assertEqual(metrics["Hit@1"], 1.0)
        self.assertEqual(metrics["FullCoverage@3"], 1.0)
        self.assertEqual(metrics["FullCoverage@1"], 0.0)

    def test_uses_graded_relevance_for_ndcg(self) -> None:
        better = compute_all_metrics(
            ["high", "low"],
            {"high", "low"},
            {"high": 3.0, "low": 1.0},
        )
        worse = compute_all_metrics(
            ["low", "high"],
            {"high", "low"},
            {"high": 3.0, "low": 1.0},
        )

        self.assertGreater(better["nDCG@1"], worse["nDCG@1"])


if __name__ == "__main__":
    unittest.main()

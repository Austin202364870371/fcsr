import unittest

from scripts.evaluate import create_rerank_progress, format_retrieval_skill


class _RecordingProgress:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs


class EvaluateProgressTests(unittest.TestCase):
    def test_retrieval_uses_training_skill_caps(self) -> None:
        skill = {"name": "n", "description": "d" * 400, "body": "b" * 3000}

        self.assertEqual(
            format_retrieval_skill(skill),
            f"n | {'d' * 300} | {'b' * 2500}",
        )
    def test_creates_query_progress_for_reranking(self) -> None:
        progress = create_rerank_progress(
            total_queries=75,
            progress_factory=_RecordingProgress,
        )

        self.assertEqual(progress.kwargs["total"], 75)
        self.assertEqual(progress.kwargs["desc"], "Rerank: scoring queries")
        self.assertEqual(progress.kwargs["unit"], "query")
        self.assertTrue(progress.kwargs["dynamic_ncols"])


if __name__ == "__main__":
    unittest.main()
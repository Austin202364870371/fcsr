import importlib.util
import subprocess
import sys
import unittest

from modeling import build_reranker_groups


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


class RerankerTests(unittest.TestCase):
    def test_builds_ordered_group_and_drops_no_positive(self) -> None:
        skills = [
            {"skill_id": "p", "name": "p", "description": "p", "body": "p"},
            {"skill_id": "n", "name": "n", "description": "n", "body": "n"},
        ]
        records = [
            {
                "query_id": "keep",
                "query": "task",
                "positive_skill_id": "p",
                "retrieved_candidates": [
                    {"skill_id": "n", "score": 0.8},
                    {"skill_id": "p", "score": 0.7},
                ],
            },
            {
                "query_id": "drop",
                "query": "task",
                "positive_skill_id": "p",
                "retrieved_candidates": [{"skill_id": "n", "score": 0.8}],
            },
        ]

        processed: list[int] = []
        result = build_reranker_groups(
            records,
            skills,
            top_k=20,
            progress=processed.append,
        )

        self.assertEqual(result.dropped_no_positive, 1)
        self.assertEqual(len(result.groups), 1)
        group = result.groups[0]
        self.assertEqual(group["positive_mask"], [False, True])
        self.assertEqual([item["rank"] for item in group["candidates"]], [1, 2])
        self.assertEqual(processed, [1, 1])

    @unittest.skipUnless(
        TORCH_AVAILABLE,
        "requires PyTorch; install requirements-train.txt",
    )
    def test_listwise_loss_rewards_positive_scores_and_supports_multi_positive(self) -> None:
        code = """
import torch
from modeling import listwise_cross_entropy
mask = torch.tensor([False, True, True])
low = listwise_cross_entropy(torch.tensor([2., 0., 0.]), mask)
high = listwise_cross_entropy(torch.tensor([0., 2., 1.]), mask)
assert high.item() < low.item()
try:
    listwise_cross_entropy(torch.tensor([1., 2.]), torch.tensor([False, False]))
except ValueError:
    pass
else:
    raise AssertionError("missing no-positive validation")
"""
        result = subprocess.run(
            [sys.executable, "-B", "-c", code],
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()

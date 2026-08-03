import importlib.util
import subprocess
import sys
import unittest

from modeling import build_biencoder_examples


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


class BiEncoderTests(unittest.TestCase):
    def test_builds_formatted_examples_from_ids(self) -> None:
        skills = [
            {
                "skill_id": "p",
                "name": "positive",
                "description": "useful",
                "body": "do the work",
            },
            {
                "skill_id": "n",
                "name": "negative",
                "description": "unrelated",
                "body": "other work",
            },
        ]
        records = [
            {
                "query_id": "q",
                "query": "complete the task",
                "positive_skill_id": "p",
                "negative_candidates": [
                    {"skill_id": "n", "source": "semantic", "score": 0.5}
                ],
            }
        ]

        example = build_biencoder_examples(records, skills)[0]

        self.assertTrue(example["query_text"].startswith("Instruct:"))
        self.assertEqual(example["positive_skill_id"], "p")
        self.assertEqual(example["negative_skill_ids"], ["n"])
        self.assertEqual(len(example["negative_texts"]), 1)

    @unittest.skipUnless(
        TORCH_AVAILABLE,
        "requires PyTorch; install requirements.txt",
    )
    def test_info_nce_prefers_aligned_positives_and_validates_temperature(self) -> None:
        code = """
import torch
from modeling import info_nce_loss
q = torch.tensor([[1., 0.], [0., 1.]])
d = torch.tensor([[1., 0.], [0., 1.]])
aligned = info_nce_loss(q, d, torch.tensor([0, 1]), 0.1)
permuted = info_nce_loss(q, d, torch.tensor([1, 0]), 0.1)
assert aligned.item() < permuted.item()
try:
    info_nce_loss(q, d, torch.tensor([0, 1]), 0.0)
except ValueError:
    pass
else:
    raise AssertionError("temperature validation missing")
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

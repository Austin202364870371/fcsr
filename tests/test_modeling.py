import importlib.util
import subprocess
import sys
import unittest

from modeling import (
    format_query,
    format_rerank_prompt,
    format_skill,
    get_reranker_template_tokens,
    tokenize_reranker_text,
)


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


class ModelingFormatTests(unittest.TestCase):
    @unittest.skipUnless(
        TORCH_AVAILABLE,
        "requires PyTorch; install requirements.txt",
    )
    def test_last_token_pool_supports_right_padding(self) -> None:
        self._assert_pooling(
            "[[[1.0], [2.0], [99.0]], [[3.0], [4.0], [5.0]]]",
            "[[1, 1, 0], [1, 1, 1]]",
        )

    @unittest.skipUnless(
        TORCH_AVAILABLE,
        "requires PyTorch; install requirements.txt",
    )
    def test_last_token_pool_supports_left_padding(self) -> None:
        self._assert_pooling(
            "[[[99.0], [1.0], [2.0]], [[3.0], [4.0], [5.0]]]",
            "[[0, 1, 1], [1, 1, 1]]",
        )

    def _assert_pooling(self, hidden: str, mask: str) -> None:
        code = (
            "import torch; from modeling import last_token_pool; "
            f"h=torch.tensor({hidden}); m=torch.tensor({mask}); "
            "assert last_token_pool(h,m).squeeze(-1).tolist()==[2.0,5.0]"
        )
        result = subprocess.run(
            [sys.executable, "-B", "-c", code],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_reranker_template_reserves_prefix_and_suffix(self) -> None:
        class FakeTokenizer:
            def encode(self, text, add_special_tokens=False):
                return [1, 2] if "system" in text else [8, 9, 10]

            def __call__(self, text, **kwargs):
                return {"input_ids": [7] * kwargs["max_length"]}

        tokenizer = FakeTokenizer()
        prefix, suffix = get_reranker_template_tokens(tokenizer)
        token_ids = tokenize_reranker_text(
            "candidate prompt", tokenizer, prefix, suffix, max_length=12
        )

        self.assertEqual(len(token_ids), 12)
        self.assertEqual(token_ids[: len(prefix)], prefix)
        self.assertEqual(token_ids[-len(suffix) :], suffix)

    def test_uses_skillrouter_query_instruction(self) -> None:
        formatted = format_query("convert a report")

        self.assertEqual(
            formatted,
            "Instruct: Given a coding task description, retrieve the most relevant "
            "skill document that would help an agent complete the task\n"
            "Query:convert a report",
        )

    def test_applies_skillrouter_skill_caps(self) -> None:
        skill = {"name": "n", "description": "d" * 400, "body": "b" * 3000}

        self.assertEqual(
            format_skill(skill),
            f"n | {'d' * 300} | {'b' * 2500}",
        )

    def test_formats_flat_full_reranker_prompt_with_caps(self) -> None:
        skill = {"name": "n", "description": "d" * 600, "body": "b" * 2200}

        prompt = format_rerank_prompt("query", skill)

        self.assertIn("<Query>: query", prompt)
        self.assertIn(f"<Document>: n | {'d' * 500} | {'b' * 2000}", prompt)


if __name__ == "__main__":
    unittest.main()
import unittest
from unittest.mock import Mock, patch

from scripts.build_single_skill_data import build_llm_client, build_parser


class BuildSingleSkillDataTests(unittest.TestCase):
    def test_uses_local_qwen3_8b_by_default(self) -> None:
        parser = build_parser()

        contracts = parser.parse_args(["contracts"])
        queries = parser.parse_args(["queries"])
        self.assertEqual(contracts.model, "models/Qwen3-8B")
        self.assertEqual(queries.model, "models/Qwen3-8B")
        self.assertEqual(contracts.backend, "transformers")
        self.assertEqual(queries.backend, "transformers")
        self.assertEqual(contracts.batch_size, 8)
        self.assertEqual(queries.batch_size, 8)

    def test_progress_can_be_disabled_for_supported_commands(self) -> None:
        parser = build_parser()

        self.assertFalse(parser.parse_args(["contracts"]).no_progress)
        self.assertFalse(parser.parse_args(["queries"]).no_progress)
        self.assertFalse(parser.parse_args(["local-negatives"]).no_progress)
        self.assertTrue(
            parser.parse_args(["contracts", "--no-progress"]).no_progress
        )
        self.assertTrue(
            parser.parse_args(["queries", "--no-progress"]).no_progress
        )
        self.assertTrue(
            parser.parse_args(["local-negatives", "--no-progress"]).no_progress
        )

    @patch("scripts.build_single_skill_data.VllmJsonClient")
    def test_builds_offline_vllm_client_with_requested_capacity(self, factory: Mock) -> None:
        backend = factory.return_value
        backend.complete_many.return_value = ["{}"]
        args = build_parser().parse_args(
            [
                "contracts",
                "--backend",
                "vllm",
                "--batch-size",
                "32",
                "--max-new-tokens",
                "2048",
            ]
        )

        client = build_llm_client(args)
        result = client.complete_many(
            messages_batch=[[{"role": "user", "content": "extract"}]],
            temperature=0.0,
        )

        factory.assert_called_once_with(
            "models/Qwen3-8B",
            max_model_len=16384,
            max_num_seqs=32,
            gpu_memory_utilization=0.9,
        )
        backend.complete_many.assert_called_once_with(
            [[{"role": "user", "content": "extract"}]],
            temperature=0.0,
            max_new_tokens=2048,
        )
        self.assertEqual(result, ["{}"])

if __name__ == "__main__":
    unittest.main()

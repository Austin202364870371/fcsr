import unittest
from unittest.mock import Mock, patch

from scripts.build_single_skill_data import build_llm_client, build_parser


class BuildSingleSkillDataTests(unittest.TestCase):
    def test_uses_deepseek_v4_flash_with_concurrency_16_by_default(self) -> None:
        parser = build_parser()

        contracts = parser.parse_args(["contracts"])
        queries = parser.parse_args(["queries"])
        self.assertEqual(contracts.model, "deepseek-v4-flash")
        self.assertEqual(queries.model, "deepseek-v4-flash")
        self.assertEqual(contracts.concurrency, 16)
        self.assertEqual(queries.concurrency, 16)
        self.assertEqual(contracts.max_new_tokens, 6144)
        sample = parser.parse_args(["sample"])
        self.assertEqual(sample.sample_size, 32000)
        self.assertEqual(sample.output_dir, "data/contracts_32k")
        self.assertEqual(
            contracts.sample, "data/contracts_32k/sample_skills.jsonl.gz"
        )
        self.assertEqual(
            contracts.output,
            "data/contracts_32k_prompt007/contracts.jsonl.gz",
        )

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

    @patch("scripts.build_single_skill_data.load_dotenv")
    @patch("scripts.build_single_skill_data.DeepSeekJsonClient")
    def test_builds_deepseek_client_with_requested_concurrency(
        self, factory: Mock, load_env: Mock
    ) -> None:
        args = build_parser().parse_args(
            [
                "contracts",
                "--concurrency",
                "16",
                "--max-new-tokens",
                "2048",
            ]
        )

        client = build_llm_client(args)

        load_env.assert_called_once()
        factory.assert_called_once_with(
            model="deepseek-v4-flash",
            concurrency=16,
            max_tokens=2048,
            timeout=180.0,
        )
        self.assertIs(client, factory.return_value)

if __name__ == "__main__":
    unittest.main()

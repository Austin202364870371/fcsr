import unittest

from scripts.build_single_skill_data import build_parser


class BuildSingleSkillDataTests(unittest.TestCase):
    def test_uses_local_qwen3_8b_by_default(self) -> None:
        parser = build_parser()

        contracts = parser.parse_args(["contracts"])
        queries = parser.parse_args(["queries"])
        self.assertEqual(contracts.model, "models/Qwen3-8B")
        self.assertEqual(queries.model, "models/Qwen3-8B")
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

if __name__ == "__main__":
    unittest.main()

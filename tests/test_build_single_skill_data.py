import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.build_single_skill_data import build_parser, load_project_env


class BuildSingleSkillDataTests(unittest.TestCase):
    def test_uses_local_qwen3_8b_by_default(self) -> None:
        parser = build_parser()

        contracts = parser.parse_args(["contracts"])
        queries = parser.parse_args(["queries"])
        self.assertEqual(contracts.provider, "local")
        self.assertEqual(contracts.model, "models/Qwen3-8B")
        self.assertEqual(queries.provider, "local")
        self.assertEqual(queries.model, "models/Qwen3-8B")

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

    def test_disables_thinking_by_default_and_allows_opt_in(self) -> None:
        parser = build_parser()

        self.assertEqual(parser.parse_args(["contracts"]).thinking, "disabled")
        self.assertEqual(parser.parse_args(["queries"]).thinking, "disabled")
        self.assertEqual(
            parser.parse_args(["contracts", "--thinking", "enabled"]).thinking,
            "enabled",
        )

    def test_loads_dotenv_without_overriding_environment(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            env_path = Path(directory) / ".env"
            env_path.write_text(
                "DEEPSEEK_API_KEY=from-dotenv\n",
                encoding="utf-8",
            )

            with patch.dict(os.environ, {}, clear=True):
                self.assertTrue(load_project_env(env_path))
                self.assertEqual(os.environ["DEEPSEEK_API_KEY"], "from-dotenv")

            with patch.dict(
                os.environ,
                {"DEEPSEEK_API_KEY": "from-environment"},
                clear=True,
            ):
                self.assertTrue(load_project_env(env_path))
                self.assertEqual(
                    os.environ["DEEPSEEK_API_KEY"],
                    "from-environment",
                )


if __name__ == "__main__":
    unittest.main()

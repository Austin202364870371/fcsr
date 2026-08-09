import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.skill_organization import parse_args
from skill_organization.workflow import (
    organizer_credentials,
    validate_skillsbench_tasks,
)


class SkillOrganizationCliTests(unittest.TestCase):
    def test_audit_defaults_to_frozen_top8(self):
        args = parse_args(["audit", "--output", "run"])
        self.assertEqual(args.command, "audit")
        self.assertEqual(args.top_k, 8)

    def test_organizer_refuses_missing_credentials(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "LLM_API_KEY"):
                organizer_credentials(
                    api_key_env="LLM_API_KEY",
                    base_url_env="LLM_BASE_URL",
                    model_env="LLM_MODEL",
                    default_model="deepseek-v4-flash",
                )

    def test_skillsbench_validation_requires_all_package_parts(self):
        with tempfile.TemporaryDirectory() as raw:
            tasks_root = Path(raw) / "tasks"
            task = tasks_root / "task-a"
            task.mkdir(parents=True)
            (task / "task.md").write_text("task", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "environment/Dockerfile"):
                validate_skillsbench_tasks(tasks_root, ("task-a",))

    def test_plan_runs_exposes_server_runtime_options(self):
        args = parse_args(
            [
                "plan-runs",
                "--run-dir",
                "run",
                "--tasks-root",
                "/skillsbench/tasks",
                "--stage",
                "pilot",
            ]
        )
        self.assertEqual(args.bench_bin, "bench")
        self.assertFalse(hasattr(args, "repeats"))
        self.assertFalse(hasattr(args, "agent"))


if __name__ == "__main__":
    unittest.main()

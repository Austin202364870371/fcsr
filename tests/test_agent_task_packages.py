import tempfile
import unittest
from pathlib import Path

from agent.task_catalog import load_pilot_catalog
from agent.task_packages import (
    audit_planning_environment,
    build_download_patterns,
    prohibited_relative_path,
    sync_task_packages,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = PROJECT_ROOT / "data" / "agent" / "hard15" / "task_catalog.json"


class TaskPackageTests(unittest.TestCase):
    def test_private_and_ground_truth_paths_are_prohibited(self) -> None:
        prohibited = (
            "x/environment/skills/a/SKILL.md",
            "x/oracle/solve.sh",
            "x/verifier/test_outputs.py",
            "x/environment/groundtruth/answer.json",
            "x/environment/ground_truth/answer.json",
            "x/environment/reference_answer/result.json",
        )
        for path in prohibited:
            with self.subTest(path=path):
                self.assertTrue(prohibited_relative_path(path))
        self.assertFalse(prohibited_relative_path("x/environment/data/input.csv"))
        self.assertFalse(prohibited_relative_path("x/task.md"))

    def test_patterns_are_exactly_scoped_to_catalog_tasks(self) -> None:
        catalog = load_pilot_catalog(CATALOG_PATH)

        allow, ignore = build_download_patterns(catalog)

        self.assertEqual(len(allow), 15)
        self.assertEqual(allow[0], "jax-computing-basics/**")
        self.assertIn("enterprise-information-search/**", allow)
        self.assertIn("*/environment/skills/**", ignore)
        self.assertIn("*/verifier/**", ignore)
        self.assertIn("*/**/groundtruth/**", ignore)

    def test_sync_uses_pinned_revision_and_audits_downloaded_tree(self) -> None:
        catalog = load_pilot_catalog(CATALOG_PATH)
        calls = []

        def fake_downloader(**kwargs):
            calls.append(kwargs)
            root = Path(kwargs["local_dir"])
            for task in catalog.tasks:
                task_root = root / task.source_task_id
                task_root.mkdir(parents=True, exist_ok=True)
                (task_root / "task.md").write_text("task", encoding="utf-8")
            return str(root)

        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as directory:
            root = Path(directory)
            manifest = sync_task_packages(catalog, root, downloader=fake_downloader)

            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0]["revision"], catalog.huggingface_revision)
            self.assertTrue(manifest.ready)
            self.assertEqual(len(manifest.tasks), 15)
            self.assertTrue(all(task.planning_ready for task in manifest.tasks))

    def test_planning_audit_rejects_private_files(self) -> None:
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as directory:
            root = Path(directory)
            task = root / "x"
            (task / "verifier").mkdir(parents=True)
            (task / "task.md").write_text("task", encoding="utf-8")
            (task / "verifier" / "test.py").write_text("private", encoding="utf-8")

            audit = audit_planning_environment("x", root)

            self.assertFalse(audit.planning_ready)
            self.assertEqual(audit.status, "prohibited_path")


if __name__ == "__main__":
    unittest.main()

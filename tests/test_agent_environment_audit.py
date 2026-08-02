import tempfile
import unittest
from pathlib import Path

from agent.environment_audit import audit_task_environment


class EnvironmentAuditTests(unittest.TestCase):
    def test_accepts_containerized_task_with_deterministic_verifier(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            task_root = Path(directory) / "task-a"
            (task_root / "environment").mkdir(parents=True)
            (task_root / "verifier").mkdir()
            (task_root / "environment" / "Dockerfile").write_text(
                "FROM python:3.10\n", encoding="utf-8"
            )
            (task_root / "verifier" / "test.sh").write_text(
                "python -m pytest\n", encoding="utf-8"
            )

            audit = audit_task_environment("task-a", Path(directory))

            self.assertEqual(audit.status, "ready")
            self.assertTrue(audit.execution_ready)

    def test_rejects_missing_task_environment(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            audit = audit_task_environment("task-a", Path(directory))

            self.assertEqual(audit.status, "missing_task_environment")
            self.assertFalse(audit.execution_ready)

    def test_rejects_task_without_verifier(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            task_root = Path(directory) / "task-a" / "environment"
            task_root.mkdir(parents=True)
            (task_root / "Dockerfile").write_text(
                "FROM python:3.10\n", encoding="utf-8"
            )

            audit = audit_task_environment("task-a", Path(directory))

            self.assertEqual(audit.status, "missing_verifier")
            self.assertFalse(audit.execution_ready)


if __name__ == "__main__":
    unittest.main()

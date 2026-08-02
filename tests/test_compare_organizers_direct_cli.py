import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class CompareOrganizersDirectCliTests(unittest.TestCase):
    def test_script_runs_directly_with_documented_pythonpath(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            environment = dict(os.environ)
            environment["PYTHONPATH"] = "src"
            completed = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    "scripts/compare_organizers.py",
                    "--tasks",
                    "data/agent/examples/tasks.json",
                    "--skills",
                    "data/agent/examples/skills.json",
                    "--output-dir",
                    str(Path(directory) / "comparison"),
                    "--max-groups",
                    "2",
                ],
                cwd=project_root,
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(
                completed.returncode,
                0,
                msg=completed.stdout + completed.stderr,
            )


if __name__ == "__main__":
    unittest.main()

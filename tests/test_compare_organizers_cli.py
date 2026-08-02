import json
import tempfile
import unittest
from pathlib import Path

from scripts.compare_organizers import main


class CompareOrganizersCliTests(unittest.TestCase):
    def test_runs_paired_methods_with_identical_candidate_input(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            output_dir = Path(directory) / "comparison"

            exit_code = main(
                [
                    "--tasks",
                    str(project_root / "data" / "agent" / "examples" / "tasks.json"),
                    "--skills",
                    str(project_root / "data" / "agent" / "examples" / "skills.json"),
                    "--output-dir",
                    str(output_dir),
                    "--max-groups",
                    "2",
                ]
            )

            self.assertEqual(exit_code, 0)
            summary = json.loads(
                (output_dir / "summary.json").read_text(encoding="utf-8")
            )
            flat_rows = self.read_jsonl(output_dir / "flat.jsonl")
            hierarchy_rows = self.read_jsonl(output_dir / "hierarchy.jsonl")
            self.assertEqual(summary["paired_task_ids"], ["uppercase-1", "json-key-1"])
            self.assertEqual(summary["methods"]["flat"]["verified"], 2)
            self.assertEqual(summary["methods"]["hierarchy"]["verified"], 2)
            self.assertEqual(flat_rows[0]["organization"]["strategy"], "flat")
            self.assertEqual(
                hierarchy_rows[0]["organization"]["strategy"],
                "hierarchy",
            )
            self.assertEqual(
                flat_rows[0]["candidate_skill_ids"],
                hierarchy_rows[0]["candidate_skill_ids"],
            )

    @staticmethod
    def read_jsonl(path: Path) -> list[dict]:
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
        ]


if __name__ == "__main__":
    unittest.main()

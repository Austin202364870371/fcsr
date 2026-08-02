import json
import tempfile
import unittest
from pathlib import Path

from scripts.run_agent import main


class RunAgentCliTests(unittest.TestCase):
    def test_runs_task_and_writes_replayable_jsonl(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            root = Path(directory)
            skills_path = root / "skills.json"
            tasks_path = root / "tasks.json"
            output_path = root / "results.jsonl"
            skills_path.write_text(
                json.dumps(
                    {
                        "skills": [
                            {
                                "skill_id": "text/uppercase",
                                "name": "uppercase",
                                "description": "Uppercase text.",
                                "tool_name": "uppercase",
                                "category_path": ["text"],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            tasks_path.write_text(
                json.dumps(
                    {
                        "tasks": [
                            {
                                "task_id": "uppercase-1",
                                "task": "hello",
                                "ranked_skill_ids": ["text/uppercase"],
                                "selector_arguments": {"text": "hello"},
                                "verifier_id": "exact",
                                "expected": "HELLO",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            exit_code = main(
                [
                    "--tasks",
                    str(tasks_path),
                    "--skills",
                    str(skills_path),
                    "--output",
                    str(output_path),
                ]
            )

            self.assertEqual(exit_code, 0)
            rows = [
                json.loads(line)
                for line in output_path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["termination_reason"], "verified")
            self.assertEqual(rows[0]["trace"][0]["event"], "organized")


if __name__ == "__main__":
    unittest.main()

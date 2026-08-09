import gzip
import json
import tempfile
import unittest
from pathlib import Path

from skill_organization.inputs import HARD15_TASK_IDS, load_frozen_inputs


class FrozenInputTests(unittest.TestCase):
    def _fixture(self, root: Path, *, record_count: int = 8) -> dict[str, Path]:
        task_ids = list(HARD15_TASK_IDS)
        skill_ids = [f"gt/s{index}" for index in range(1, 9)]
        paths = {
            "task_ids": root / "task_ids.txt",
            "catalog": root / "task_catalog.json",
            "predictions": root / "predictions.json",
            "skills": root / "skills.jsonl.gz",
        }
        paths["task_ids"].write_text("\n".join(task_ids) + "\n", encoding="utf-8")
        paths["catalog"].write_text(
            json.dumps(
                {
                    "skillsbench_version": "v1.1",
                    "github_commit": "b63b7b2850226b6aa4fb5929a8c1ac7bc4d9a6af",
                    "huggingface_revision": "be2a6ce2cb1f4ff67ce937307cade0c5a0477a13",
                    "tasks": [{"task_id": task_id} for task_id in task_ids],
                }
            ),
            encoding="utf-8",
        )
        paths["predictions"].write_text(
            json.dumps({task_id: skill_ids + ["other/s9"] for task_id in task_ids}),
            encoding="utf-8",
        )
        with gzip.open(paths["skills"], "wt", encoding="utf-8", newline="\n") as handle:
            for index in range(1, record_count + 1):
                handle.write(
                    json.dumps(
                        {
                            "skill_id": f"gt/s{index}",
                            "name": f"Skill {index}",
                            "description": f"Description {index}",
                            "body": f"Body {index}",
                            "source": "gt",
                        }
                    )
                    + "\n"
                )
        return paths

    def _load(self, paths: dict[str, Path]):
        return load_frozen_inputs(
            predictions_path=paths["predictions"],
            skills_path=paths["skills"],
            task_ids_path=paths["task_ids"],
            task_catalog_path=paths["catalog"],
            expected_skills_sha256=None,
            top_k=8,
        )

    def test_loads_exact_top8_and_hides_provenance_from_organizer_view(self):
        with tempfile.TemporaryDirectory() as raw:
            paths = self._fixture(Path(raw))
            frozen = self._load(paths)

        self.assertEqual(len(frozen.tasks), 15)
        task = frozen.tasks[0]
        self.assertEqual(tuple(item.rank for item in task.skills), tuple(range(1, 9)))
        self.assertEqual(task.skills[-1].record.skill_id, "gt/s8")
        view = task.skills[0].organizer_view()
        self.assertEqual(set(view), {"alias", "rank", "name", "description", "body"})
        self.assertNotIn("gt/", json.dumps(view))
        self.assertEqual(frozen.skillsbench_version, "v1.1")

    def test_fails_when_a_top8_record_is_missing(self):
        with tempfile.TemporaryDirectory() as raw:
            paths = self._fixture(Path(raw), record_count=7)
            with self.assertRaisesRegex(ValueError, "missing frozen Skill records"):
                self._load(paths)

    def test_fails_when_catalog_and_task_ids_disagree(self):
        with tempfile.TemporaryDirectory() as raw:
            paths = self._fixture(Path(raw))
            catalog = json.loads(paths["catalog"].read_text(encoding="utf-8"))
            catalog["tasks"][-1]["task_id"] = "different-task"
            paths["catalog"].write_text(json.dumps(catalog), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "catalog task IDs"):
                self._load(paths)

    def test_rejects_non_top8_configuration(self):
        with tempfile.TemporaryDirectory() as raw:
            paths = self._fixture(Path(raw))
            with self.assertRaisesRegex(ValueError, "top_k=8"):
                load_frozen_inputs(
                    predictions_path=paths["predictions"],
                    skills_path=paths["skills"],
                    task_ids_path=paths["task_ids"],
                    task_catalog_path=paths["catalog"],
                    expected_skills_sha256=None,
                    top_k=7,
                )


if __name__ == "__main__":
    unittest.main()

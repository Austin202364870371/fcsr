import gzip
import json
import tempfile
import unittest
from pathlib import Path

from skill_organization.organizer import OrganizerReply
from skill_organization.inputs import HARD15_TASK_IDS
from skill_organization.workflow import (
    audit_run,
    EXPECTED_BENCHFLOW_VERSION,
    EXPECTED_SANDBOX_PACKAGES,
    organize_run,
    record_review,
    record_reviews,
    require_oracle_preflight,
    render_reviewed,
)


class FakeOrganizer:
    def organize(
        self, *, task_key: str, skills: list[dict[str, object]]
    ) -> OrganizerReply:
        aliases = [str(skill["alias"]) for skill in skills]
        content = json.dumps(
            {
                "hierarchy": {
                    "schema_version": "skill-hierarchy-v1",
                    "roots": [
                        {
                            "label": "Procedures",
                            "children": [{"label": "Operations", "skills": aliases}],
                        }
                    ],
                },
                "graph": {
                    "schema_version": "skill-graph-v1",
                    "nodes": aliases,
                    "edges": [],
                },
            }
        )
        return OrganizerReply(content=content, usage={"total_tokens": 10})


class WorkflowTests(unittest.TestCase):
    def _fixture(self, root: Path) -> dict[str, Path]:
        task_ids = list(HARD15_TASK_IDS)
        report = root / "report"
        report.mkdir()
        paths = {
            "predictions": report / "predictions.json",
            "skills": root / "skills.jsonl.gz",
            "task_ids": root / "task_ids.txt",
            "catalog": root / "task_catalog.json",
            "run": root / "run",
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
        skills = [f"gt/s{index}" for index in range(1, 9)]
        paths["predictions"].write_text(
            json.dumps({task_id: skills for task_id in task_ids}), encoding="utf-8"
        )
        for name in ("details.jsonl", "records.jsonl"):
            (report / name).write_text("{}\n", encoding="utf-8")
        (report / "summary.json").write_text("{}", encoding="utf-8")
        with gzip.open(paths["skills"], "wt", encoding="utf-8") as handle:
            for index in range(1, 9):
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

    def test_audit_organize_review_and_render_end_to_end(self):
        with tempfile.TemporaryDirectory() as raw:
            paths = self._fixture(Path(raw))
            manifest = audit_run(
                run_dir=paths["run"],
                predictions_path=paths["predictions"],
                skills_path=paths["skills"],
                task_ids_path=paths["task_ids"],
                task_catalog_path=paths["catalog"],
                expected_skills_sha256=None,
                expected_predictions_sha256=None,
                expected_task_ids_sha256=None,
                expected_task_catalog_sha256=None,
                expected_report_sha256=None,
            )
            self.assertEqual(manifest["counts"]["skill_instances"], 120)
            self.assertEqual(manifest["task_keys"][0], "T001")
            self.assertNotIsInstance(manifest["task_keys"], dict)
            organized = organize_run(
                run_dir=paths["run"],
                client=FakeOrganizer(),
                model="fake-model",
                endpoint="https://example.invalid",
            )
            self.assertEqual(organized, {"created": 15, "reused": 0})
            request = (
                paths["run"] / "preprocessing/organizer_requests/T001.json"
            ).read_text(encoding="utf-8")
            self.assertNotIn("gt/s1", request)
            self.assertNotIn('"source"', request)

            with self.assertRaisesRegex(ValueError, "require approval"):
                render_reviewed(paths["run"])
            reviews = record_reviews(
                run_dir=paths["run"],
                task_keys=tuple(f"T{index:03d}" for index in range(1, 16)),
                decision="approve",
                reviewer="test-reviewer",
                notes="fixture reviewed",
            )
            self.assertEqual(len(reviews), 15)
            record_review(
                run_dir=paths["run"],
                task_key="T001",
                decision="reject",
                reviewer="second-reviewer",
                notes="append-only rejection",
            )
            with self.assertRaisesRegex(ValueError, "require approval"):
                render_reviewed(paths["run"])
            record_review(
                run_dir=paths["run"],
                task_key="T001",
                decision="approve",
                reviewer="second-reviewer",
                notes="append-only re-approval",
            )
            events = (
                (paths["run"] / "preprocessing" / "review_events.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            )
            self.assertEqual(len(events), 17)
            rendered = render_reviewed(paths["run"])
            self.assertEqual(rendered, {"tasks": 15, "packages": 45})
            packages = list((paths["run"] / "generated").rglob("SKILL.md"))
            self.assertEqual(len(packages), 45)

    def test_audit_rejects_a_non_v11_skillsbench_catalog(self):
        with tempfile.TemporaryDirectory() as raw:
            paths = self._fixture(Path(raw))
            catalog = json.loads(paths["catalog"].read_text(encoding="utf-8"))
            catalog["github_commit"] = "9a1f4dd5f7659f75707435da3ce854b6e48321d1"
            paths["catalog"].write_text(json.dumps(catalog), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "v1.1 commit"):
                audit_run(
                    run_dir=paths["run"],
                    predictions_path=paths["predictions"],
                    skills_path=paths["skills"],
                    task_ids_path=paths["task_ids"],
                    task_catalog_path=paths["catalog"],
                    expected_skills_sha256=None,
                    expected_predictions_sha256=None,
                    expected_task_ids_sha256=None,
                    expected_task_catalog_sha256=None,
                    expected_report_sha256=None,
                )

    def test_oracle_preflight_gate_requires_all_registered_tasks(self):
        with tempfile.TemporaryDirectory() as raw:
            paths = self._fixture(Path(raw))
            audit_run(
                run_dir=paths["run"],
                predictions_path=paths["predictions"],
                skills_path=paths["skills"],
                task_ids_path=paths["task_ids"],
                task_catalog_path=paths["catalog"],
                expected_skills_sha256=None,
                expected_predictions_sha256=None,
                expected_task_ids_sha256=None,
                expected_task_catalog_sha256=None,
                expected_report_sha256=None,
            )
            report_path = paths["run"] / "preflight" / "oracle_preflight.json"
            report_path.parent.mkdir(parents=True)
            report_path.write_text(
                json.dumps(
                    {
                        "completed": True,
                        "benchflow_version": EXPECTED_BENCHFLOW_VERSION,
                        "skillsbench_commit": "b63b7b2850226b6aa4fb5929a8c1ac7bc4d9a6af",
                        "sandbox_package_versions": EXPECTED_SANDBOX_PACKAGES,
                        "tasks_root": str(Path(raw).resolve()),
                        "tasks": {
                            task_id: {"passed": True} for task_id in HARD15_TASK_IDS
                        },
                    }
                ),
                encoding="utf-8",
            )
            report = require_oracle_preflight(paths["run"])
        self.assertTrue(report["completed"])


if __name__ == "__main__":
    unittest.main()

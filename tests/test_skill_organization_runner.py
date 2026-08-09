import json
import tempfile
import unittest
from pathlib import Path

from skill_organization.runner import (
    CONDITIONS,
    EXPERIMENT_BENCHFLOW_VERSION,
    EXPERIMENT_SANDBOX_PACKAGES,
    RunSpec,
    bench_command,
    build_run_matrix,
    execute_matrix,
    validate_matrix_protocol,
    write_run_matrix,
)


HARD15_IDS = tuple(f"task-{index:02d}" for index in range(1, 16))


def make_spec(condition: str) -> RunSpec:
    return RunSpec(
        run_key=f"task-a__{condition}__r01",
        task_id="task-a",
        condition=condition,
        repeat_id=1,
        order_index=0,
        task_dir=Path("/skillsbench/tasks/task-a"),
        skills_dir=(
            None
            if condition == "no_skill"
            else Path(f"/generated/task-a/{condition}/skills")
        ),
        jobs_dir=Path(f"/jobs/task-a__{condition}__r01"),
        stage="pilot",
        agent="openhands",
        model="deepseek/deepseek-v4-flash",
        sandbox="daytona",
        bench_bin="bench",
    )


class RunnerTests(unittest.TestCase):
    def test_hard15_matrix_has_sixty_unique_runs(self):
        specs = build_run_matrix(
            task_ids=HARD15_IDS,
            tasks_root=Path("/skillsbench/tasks"),
            generated_root=Path("/generated"),
            jobs_root=Path("/jobs"),
            repeats=1,
        )
        self.assertEqual(len(specs), 60)
        self.assertEqual(len({spec.run_key for spec in specs}), 60)
        self.assertEqual(
            {
                condition: sum(spec.condition == condition for spec in specs)
                for condition in CONDITIONS
            },
            {condition: 15 for condition in CONDITIONS},
        )

    def test_condition_order_rotates_by_task(self):
        specs = build_run_matrix(
            task_ids=HARD15_IDS[:4],
            tasks_root=Path("/tasks"),
            generated_root=Path("/generated"),
            jobs_root=Path("/jobs"),
            repeats=1,
        )
        first_condition = [
            spec.condition for spec in specs if spec.order_index % 4 == 0
        ]
        self.assertEqual(first_condition, list(CONDITIONS))

    def test_no_skill_command_has_no_skills_dir(self):
        command = bench_command(make_spec("no_skill"), bench_bin="bench")
        self.assertIn("no-skill", command)
        self.assertNotIn("--skills-dir", command)

    def test_flat_command_uses_generated_package(self):
        command = bench_command(make_spec("flat_top8"), bench_bin="bench")
        self.assertIn("with-skill", command)
        self.assertIn("--skills-dir", command)
        self.assertIn(str(Path("/generated/task-a/flat_top8/skills")), command)

    def test_matrix_round_trips_as_jsonl(self):
        specs = build_run_matrix(
            task_ids=("task-a",),
            tasks_root=Path("/tasks"),
            generated_root=Path("/generated"),
            jobs_root=Path("/jobs"),
            repeats=1,
        )
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "run_matrix.jsonl"
            write_run_matrix(path, specs)
            lines = path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 4)

    def test_matrix_freezes_runtime_configuration(self):
        specs = build_run_matrix(
            task_ids=("task-a",),
            tasks_root=Path("/tasks"),
            generated_root=Path("/generated"),
            jobs_root=Path("/jobs"),
            repeats=1,
            agent="openhands",
            model="deepseek/deepseek-v4-flash",
            sandbox="daytona",
            bench_bin="/opt/bench",
        )
        self.assertTrue(all(spec.bench_bin == "/opt/bench" for spec in specs))
        self.assertEqual(bench_command(specs[0])[0], "/opt/bench")

    def test_existing_process_error_is_not_retried_implicitly(self):
        spec = make_spec("no_skill")
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            state_root = root / "states"
            state_root.mkdir()
            existing = {
                "run_key": spec.run_key,
                "status": "process_error",
                "returncode": 2,
            }
            (state_root / f"{spec.run_key}.json").write_text(
                json.dumps(existing), encoding="utf-8"
            )
            states = execute_matrix(
                (spec,), state_root=state_root, log_root=root / "logs"
            )
        self.assertEqual(states, (existing,))

    def test_missing_bench_binary_is_recorded_as_process_error(self):
        spec = make_spec("no_skill").model_copy(
            update={"bench_bin": "definitely-not-a-bench-binary"}
        )
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            states = execute_matrix(
                (spec,), state_root=root / "states", log_root=root / "logs"
            )
            saved = json.loads(
                (root / "states" / f"{spec.run_key}.json").read_text(encoding="utf-8")
            )
        self.assertEqual(states[0]["status"], "process_error")
        self.assertEqual(saved["status"], "process_error")
        self.assertEqual(saved["failure_type"], "process_start_error")

    def test_run_matrix_refuses_different_overwrite(self):
        specs = build_run_matrix(
            task_ids=("task-a",),
            tasks_root=Path("/tasks"),
            generated_root=Path("/generated"),
            jobs_root=Path("/jobs"),
            repeats=1,
        )
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "run_matrix.jsonl"
            write_run_matrix(path, specs)
            changed = tuple(
                spec.model_copy(update={"model": "different-model"}) for spec in specs
            )
            with self.assertRaisesRegex(ValueError, "immutable"):
                write_run_matrix(path, changed)

    def test_two_task_smoke_has_eight_stage_isolated_runs(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            generated = root / "generated"
            smoke_tasks = ("jax-computing-basics", "citation-check")
            for task_id in smoke_tasks:
                for condition in CONDITIONS[1:]:
                    manifest = generated / task_id / condition / "context_manifest.json"
                    manifest.parent.mkdir(parents=True)
                    manifest.write_text(
                        json.dumps(
                            {
                                "rendered_context_sha256": "a" * 64,
                                "atomic_payload_sha256": "b" * 64,
                            }
                        ),
                        encoding="utf-8",
                    )
            specs = build_run_matrix(
                task_ids=smoke_tasks,
                tasks_root=root / "tasks",
                generated_root=generated,
                jobs_root=root / "jobs/smoke",
                repeats=1,
                stage="smoke",
                require_context_manifests=True,
                benchflow_version=EXPERIMENT_BENCHFLOW_VERSION,
                sandbox_package_versions=EXPERIMENT_SANDBOX_PACKAGES,
            )
            validate_matrix_protocol(
                specs,
                "smoke",
                tasks_root=root / "tasks",
                generated_root=generated,
                jobs_root=root / "jobs/smoke",
            )
            tampered = (
                specs[0].model_copy(update={"task_dir": root / "other-task"}),
                *specs[1:],
            )
            with self.assertRaisesRegex(ValueError, "task directory mismatch"):
                validate_matrix_protocol(
                    tampered,
                    "smoke",
                    tasks_root=root / "tasks",
                    generated_root=generated,
                    jobs_root=root / "jobs/smoke",
                )
            self.assertEqual(len(specs), 8)
            self.assertTrue(all(spec.stage == "smoke" for spec in specs))
            self.assertTrue(all(spec.run_key.startswith("smoke__") for spec in specs))

    def test_matrix_rejects_non_frozen_repeats_or_runtime(self):
        common = {
            "task_ids": ("task-a",),
            "tasks_root": Path("/tasks"),
            "generated_root": Path("/generated"),
            "jobs_root": Path("/jobs"),
        }
        with self.assertRaisesRegex(ValueError, "exactly one repeat"):
            build_run_matrix(**common, repeats=2)
        with self.assertRaisesRegex(ValueError, "runtime is frozen"):
            build_run_matrix(**common, agent="different-agent")


if __name__ == "__main__":
    unittest.main()

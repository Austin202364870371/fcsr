import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from skill_organization.results import (
    classify_result,
    collect_one,
    collect_results,
    validate_smoke_gate,
)
from skill_organization.runner import RunSpec


def make_spec(condition: str = "flat_top8") -> RunSpec:
    return RunSpec(
        run_key=f"task-a__{condition}__r01",
        task_id="task-a",
        condition=condition,
        repeat_id=1,
        order_index=0,
        task_dir=Path("/tasks/task-a"),
        skills_dir=(
            None
            if condition == "no_skill"
            else Path("/generated/task-a/flat_top8/skills")
        ),
        jobs_dir=Path("/jobs/run"),
    )


def result_fixture(*, reward=1.0, **overrides):
    value = {
        "rewards": {"reward": reward},
        "requested_skills_dir": "/generated/task-a/flat_top8/skills",
        "effective_skills_dir": "/sandbox/skills",
        "skill_mode": "with-skill",
        "n_skill_invocations": 1,
        "n_tool_calls": 6,
        "agent_result": {
            "n_input_tokens": 100,
            "n_output_tokens": 20,
            "total_tokens": 120,
            "cost_usd": 0.01,
        },
        "trajectory_summary": {"steps": 14},
        "timing": {
            "environment_setup": 5.0,
            "agent_execution": 60.0,
            "verifier": 10.0,
            "total": 80.0,
        },
        "source": {"resolved_sha": "b63b7b2850226b6aa4fb5929a8c1ac7bc4d9a6af"},
        "error": None,
        "error_category": None,
        "verifier_error": None,
        "verifier_error_category": None,
        "idle_timeout_info": None,
        "agent_timeout_info": None,
        "sandbox_startup_info": None,
        "transport_error_info": None,
        "api_error_info": None,
    }
    value.update(overrides)
    return value


class ResultTests(unittest.TestCase):
    def test_reward_one_is_passed_with_metrics(self):
        row = classify_result(make_spec(), result_fixture())
        self.assertEqual(row.status, "passed")
        self.assertEqual(row.reward, 1.0)
        self.assertEqual(row.total_tokens, 120)
        self.assertTrue(row.injection_verified)

    def test_reward_zero_is_valid_failure(self):
        row = classify_result(make_spec(), result_fixture(reward=0.0))
        self.assertEqual(row.status, "failed")
        self.assertEqual(row.reward, 0.0)

    def test_verifier_error_is_not_converted_to_reward_zero(self):
        row = classify_result(
            make_spec(),
            result_fixture(
                reward=None,
                verifier_error="verifier crashed",
                verifier_error_category="verifier_exception",
            ),
        )
        self.assertEqual(row.status, "verifier_error")
        self.assertIsNone(row.reward)

    def test_verifier_timeout_is_a_verifier_error(self):
        row = classify_result(
            make_spec(),
            result_fixture(reward=None, verifier_timeout_info={"timeout_sec": 900}),
        )
        self.assertEqual(row.status, "verifier_error")
        self.assertEqual(row.failure_type, "verifier_timeout")
        self.assertIsNone(row.reward)

    def test_agent_timeout_has_distinct_status(self):
        row = classify_result(
            make_spec(),
            result_fixture(reward=None, agent_timeout_info={"timeout_sec": 600}),
        )
        self.assertEqual(row.status, "timeout")
        self.assertIsNone(row.reward)

    def test_sandbox_startup_error_is_infrastructure_error(self):
        row = classify_result(
            make_spec(),
            result_fixture(
                reward=None, sandbox_startup_info={"message": "Daytona failed"}
            ),
        )
        self.assertEqual(row.status, "infrastructure_error")
        self.assertIsNone(row.reward)

    def test_no_skill_injection_is_verified_only_when_skill_dirs_are_absent(self):
        result = result_fixture(
            requested_skills_dir=None,
            effective_skills_dir=None,
            skill_mode="no-skill",
            n_skill_invocations=0,
        )
        row = classify_result(make_spec("no_skill"), result)
        self.assertTrue(row.injection_verified)

    def test_wrong_skillsbench_source_commit_is_infrastructure_error(self):
        row = classify_result(
            make_spec(),
            result_fixture(
                source={"resolved_sha": "9a1f4dd5f7659f75707435da3ce854b6e48321d1"}
            ),
        )
        self.assertEqual(row.status, "infrastructure_error")
        self.assertEqual(row.failure_type, "skillsbench_source_mismatch")
        self.assertIsNone(row.reward)

    def test_missing_optional_artifact_source_commit_does_not_invalidate_result(self):
        row = classify_result(make_spec(), result_fixture(source=None))
        self.assertEqual(row.status, "passed")
        self.assertIsNone(row.skillsbench_source_commit)

    def test_injection_verification_rejects_wrong_skill_mode(self):
        row = classify_result(make_spec(), result_fixture(skill_mode="no-skill"))
        self.assertFalse(row.injection_verified)

    def test_result_artifact_takes_priority_over_nonzero_bench_exit(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            spec = make_spec().model_copy(update={"jobs_dir": root / "jobs"})
            state_root = root / "states"
            state_root.mkdir()
            (state_root / f"{spec.run_key}.json").write_text(
                json.dumps({"status": "process_error", "returncode": 1}),
                encoding="utf-8",
            )
            result_dir = spec.jobs_dir / "timestamp" / "rollout"
            result_dir.mkdir(parents=True)
            (result_dir / "result.json").write_text(
                json.dumps(
                    result_fixture(
                        reward=None,
                        verifier_error="verifier crashed",
                        verifier_error_category="verifier_exception",
                    )
                ),
                encoding="utf-8",
            )

            row = collect_one(spec, state_root)

        self.assertEqual(row.status, "verifier_error")
        self.assertEqual(row.failure_type, "verifier_exception")

    def test_collection_verifies_frozen_context_and_prompt_excerpt(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            condition_root = root / "generated" / "task-a" / "flat_top8"
            skills_dir = condition_root / "skills"
            skill_path = skills_dir / "retrieved-skills" / "SKILL.md"
            skill_path.parent.mkdir(parents=True)
            context = (
                "---\nname: retrieved-skills\n---\n"
                "## Atomic skill payloads\n\n### S01\n\nName: Frozen skill\n\n"
                + "x" * 300
            )
            skill_path.write_bytes(context.encode("utf-8"))
            context_hash = hashlib.sha256(context.encode("utf-8")).hexdigest()
            atomic_hash = "b" * 64
            (condition_root / "context_manifest.json").write_text(
                json.dumps(
                    {
                        "rendered_context_sha256": context_hash,
                        "atomic_payload_sha256": atomic_hash,
                    }
                ),
                encoding="utf-8",
            )
            jobs_dir = root / "jobs"
            rollout = jobs_dir / "timestamp" / "rollout"
            rollout.mkdir(parents=True)
            spec = make_spec().model_copy(
                update={
                    "skills_dir": skills_dir,
                    "jobs_dir": jobs_dir,
                    "rendered_context_sha256": context_hash,
                    "atomic_payload_sha256": atomic_hash,
                }
            )
            (rollout / "result.json").write_text(
                json.dumps(result_fixture(requested_skills_dir=str(skills_dir))),
                encoding="utf-8",
            )
            (rollout / "prompts.json").write_text(
                json.dumps({"system": context}), encoding="utf-8"
            )

            row = collect_one(spec, root / "states")

        self.assertTrue(row.context_artifact_verified)
        self.assertTrue(row.skill_prompt_evidence)

    def test_collection_writes_matrix_aggregate_and_failure_files(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            specs = []
            for index, condition in enumerate(
                ("no_skill", "flat_top8", "hierarchy_top8", "graph_top8")
            ):
                base = make_spec(condition)
                spec = base.model_copy(
                    update={
                        "run_key": f"pilot__task-a__{condition}__r01",
                        "order_index": index,
                        "jobs_dir": root / "jobs" / condition,
                    }
                )
                specs.append(spec)
                result_dir = spec.jobs_dir / "timestamp" / "rollout"
                result_dir.mkdir(parents=True)
                if condition == "no_skill":
                    payload = result_fixture(
                        reward=0.0,
                        skill_mode="no-skill",
                        requested_skills_dir=None,
                        effective_skills_dir=None,
                    )
                else:
                    payload = result_fixture(
                        requested_skills_dir=str(spec.skills_dir),
                        effective_skills_dir="/sandbox/skills",
                    )
                (result_dir / "result.json").write_text(
                    json.dumps(payload), encoding="utf-8"
                )

            output = root / "results"
            rows = collect_results(
                specs=tuple(specs), state_root=root / "states", output_root=output
            )
            aggregate = json.loads(
                (output / "aggregate.json").read_text(encoding="utf-8")
            )

            self.assertEqual(len(rows), 4)
            self.assertEqual(
                aggregate["conditions"]["no_skill"]["fixed_denominator_pass_rate"], 0.0
            )
            self.assertEqual(aggregate["transitions"]["flat_top8-no_skill"]["0->1"], 1)
            self.assertEqual(
                len(
                    (output / "task_matrix.csv")
                    .read_text(encoding="utf-8")
                    .splitlines()
                ),
                2,
            )
            self.assertEqual(
                (output / "failures.jsonl").read_text(encoding="utf-8"), ""
            )

    def test_smoke_gate_requires_prompt_and_context_evidence(self):
        rows = []
        specs = []
        for task_id in ("task-a", "task-b"):
            for condition in ("no_skill", "flat_top8", "hierarchy_top8", "graph_top8"):
                spec = make_spec(condition).model_copy(
                    update={
                        "run_key": f"smoke__{task_id}__{condition}__r01",
                        "stage": "smoke",
                        "task_id": task_id,
                    }
                )
                specs.append(spec)
                payload = result_fixture(
                    skill_mode="no-skill" if condition == "no_skill" else "with-skill",
                    requested_skills_dir=(
                        None if condition == "no_skill" else str(spec.skills_dir)
                    ),
                    effective_skills_dir=(
                        None if condition == "no_skill" else "/sandbox/skills"
                    ),
                )
                row = classify_result(spec, payload).model_copy(
                    update={
                        "context_artifact_verified": (
                            None if condition == "no_skill" else True
                        ),
                        "skill_prompt_evidence": condition != "no_skill",
                    }
                )
                rows.append(row)
        with tempfile.TemporaryDirectory() as raw:
            output = Path(raw)
            (output / "trajectories.jsonl").write_text(
                "".join(row.model_dump_json() + "\n" for row in rows), encoding="utf-8"
            )
            validated = validate_smoke_gate(output, specs)
            rows[0] = rows[0].model_copy(update={"model": "substituted-model"})
            (output / "trajectories.jsonl").write_text(
                "".join(row.model_dump_json() + "\n" for row in rows), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "differs from matrix"):
                validate_smoke_gate(output, specs)
        self.assertEqual(len(validated), 8)


if __name__ == "__main__":
    unittest.main()

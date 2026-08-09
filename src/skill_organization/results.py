"""Normalize BenchFlow artifacts and summarize paired Skill conditions."""

from __future__ import annotations

import csv
import hashlib
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Literal, Sequence

from pydantic import BaseModel, ConfigDict

from data_io import stream_jsonl, write_jsonl_atomic
from skill_organization.runner import CONDITIONS, RunSpec
from skill_organization.workflow import write_json_atomic


ResultStatus = Literal[
    "passed",
    "failed",
    "agent_error",
    "verifier_error",
    "timeout",
    "infrastructure_error",
    "missing_artifact",
]


class TrajectoryResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    run_key: str
    stage: str
    task_id: str
    condition: str
    repeat_id: int
    stratum: str | None
    status: ResultStatus
    reward: float | None
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost_usd: float
    tool_calls: int
    skill_invocations: int
    trajectory_steps: int
    environment_setup_time_s: float | None
    agent_execution_time_s: float | None
    verifier_time_s: float | None
    wall_time_s: float | None
    injection_verified: bool | None
    context_artifact_verified: bool | None
    skill_prompt_evidence: bool | None
    skillsbench_source_commit: str | None
    expected_skillsbench_commit: str
    agent: str
    model: str
    sandbox: str
    rendered_context_sha256: str | None
    atomic_payload_sha256: str | None
    predictions_sha256: str | None
    skills_sha256: str | None
    task_catalog_sha256: str | None
    benchflow_version: str | None
    benchflow_agents_commit: str | None
    sandbox_package_versions: dict[str, str] | None
    agent_idle_timeout_sec: int
    loop_strategy: str
    reasoning_mode: str
    failure_type: str | None


def _number(value: Any) -> float | None:
    return (
        float(value)
        if isinstance(value, (int, float)) and not isinstance(value, bool)
        else None
    )


def _integer(value: Any) -> int:
    return (
        int(value)
        if isinstance(value, (int, float)) and not isinstance(value, bool)
        else 0
    )


def _normalized_path(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    return value.replace("\\", "/").rstrip("/")


def _injection_verified(spec: RunSpec, result: dict[str, Any]) -> bool:
    requested = _normalized_path(result.get("requested_skills_dir"))
    effective = _normalized_path(result.get("effective_skills_dir"))
    skill_mode = result.get("skill_mode")
    if spec.condition == "no_skill":
        return skill_mode == "no-skill" and requested is None and effective is None
    expected = _normalized_path(str(spec.skills_dir))
    return (
        skill_mode == "with-skill" and requested == expected and effective is not None
    )


def classify_result(spec: RunSpec, result: dict[str, Any]) -> TrajectoryResult:
    rewards = result.get("rewards")
    reward = _number(rewards.get("reward")) if isinstance(rewards, dict) else None
    failure_type: str | None = None
    source = result.get("source") if isinstance(result.get("source"), dict) else {}
    source_commit = (
        source.get("resolved_sha")
        if isinstance(source.get("resolved_sha"), str)
        else None
    )

    verifier_timeout = result.get("verifier_timeout_info")
    verifier_error = (
        result.get("verifier_error")
        or result.get("verifier_error_category")
        or verifier_timeout
    )
    timeout_info = result.get("agent_timeout_info") or result.get("idle_timeout_info")
    infrastructure = result.get("sandbox_startup_info") or result.get(
        "transport_error_info"
    )
    agent_error = (
        result.get("api_error_info")
        or result.get("error")
        or result.get("error_category")
    )
    # BenchFlow's local ``--tasks-dir`` source does not always emit a resolved
    # Git SHA.  The checkout is pinned and verified before execution; treat an
    # artifact SHA as an additional cross-check only when it is present.
    if source_commit is not None and source_commit != spec.skillsbench_commit:
        status: ResultStatus = "infrastructure_error"
        failure_type = "skillsbench_source_mismatch"
        reward = None
    elif verifier_error:
        status = "verifier_error"
        failure_type = (
            "verifier_timeout"
            if verifier_timeout
            else str(result.get("verifier_error_category") or verifier_error)
        )
        reward = None
    elif timeout_info:
        status = "timeout"
        failure_type = (
            "agent_timeout" if result.get("agent_timeout_info") else "idle_timeout"
        )
        reward = None
    elif infrastructure:
        status = "infrastructure_error"
        failure_type = (
            "sandbox_startup" if result.get("sandbox_startup_info") else "transport"
        )
        reward = None
    elif agent_error:
        status = "agent_error"
        failure_type = str(result.get("error_category") or "agent_error")
        reward = None
    elif reward is None:
        status = "missing_artifact"
        failure_type = "missing_reward"
    elif reward >= 1.0:
        status = "passed"
    else:
        status = "failed"

    agent = (
        result.get("agent_result")
        if isinstance(result.get("agent_result"), dict)
        else {}
    )
    trajectory = (
        result.get("trajectory_summary")
        if isinstance(result.get("trajectory_summary"), dict)
        else {}
    )
    timing = result.get("timing") if isinstance(result.get("timing"), dict) else {}
    return TrajectoryResult(
        run_key=spec.run_key,
        stage=spec.stage,
        task_id=spec.task_id,
        condition=spec.condition,
        repeat_id=spec.repeat_id,
        stratum=spec.stratum,
        status=status,
        reward=reward,
        input_tokens=_integer(agent.get("n_input_tokens")),
        output_tokens=_integer(agent.get("n_output_tokens")),
        total_tokens=_integer(agent.get("total_tokens")),
        cost_usd=float(_number(agent.get("cost_usd")) or 0.0),
        tool_calls=_integer(result.get("n_tool_calls")),
        skill_invocations=_integer(result.get("n_skill_invocations")),
        trajectory_steps=_integer(trajectory.get("steps")),
        environment_setup_time_s=_number(timing.get("environment_setup")),
        agent_execution_time_s=_number(timing.get("agent_execution")),
        verifier_time_s=_number(timing.get("verifier")),
        wall_time_s=_number(timing.get("total")),
        injection_verified=_injection_verified(spec, result),
        context_artifact_verified=None,
        skill_prompt_evidence=None,
        skillsbench_source_commit=source_commit,
        expected_skillsbench_commit=spec.skillsbench_commit,
        agent=spec.agent,
        model=spec.model,
        sandbox=spec.sandbox,
        rendered_context_sha256=spec.rendered_context_sha256,
        atomic_payload_sha256=spec.atomic_payload_sha256,
        predictions_sha256=spec.predictions_sha256,
        skills_sha256=spec.skills_sha256,
        task_catalog_sha256=spec.task_catalog_sha256,
        benchflow_version=spec.benchflow_version,
        benchflow_agents_commit=spec.benchflow_agents_commit,
        sandbox_package_versions=spec.sandbox_package_versions,
        agent_idle_timeout_sec=spec.agent_idle_timeout_sec,
        loop_strategy=spec.loop_strategy,
        reasoning_mode=spec.reasoning_mode,
        failure_type=failure_type,
    )


def _error_row(
    spec: RunSpec, status: ResultStatus, failure_type: str
) -> TrajectoryResult:
    return TrajectoryResult(
        run_key=spec.run_key,
        stage=spec.stage,
        task_id=spec.task_id,
        condition=spec.condition,
        repeat_id=spec.repeat_id,
        stratum=spec.stratum,
        status=status,
        reward=None,
        input_tokens=0,
        output_tokens=0,
        total_tokens=0,
        cost_usd=0.0,
        tool_calls=0,
        skill_invocations=0,
        trajectory_steps=0,
        environment_setup_time_s=None,
        agent_execution_time_s=None,
        verifier_time_s=None,
        wall_time_s=None,
        injection_verified=None,
        context_artifact_verified=None,
        skill_prompt_evidence=None,
        skillsbench_source_commit=None,
        expected_skillsbench_commit=spec.skillsbench_commit,
        agent=spec.agent,
        model=spec.model,
        sandbox=spec.sandbox,
        rendered_context_sha256=spec.rendered_context_sha256,
        atomic_payload_sha256=spec.atomic_payload_sha256,
        predictions_sha256=spec.predictions_sha256,
        skills_sha256=spec.skills_sha256,
        task_catalog_sha256=spec.task_catalog_sha256,
        benchflow_version=spec.benchflow_version,
        benchflow_agents_commit=spec.benchflow_agents_commit,
        sandbox_package_versions=spec.sandbox_package_versions,
        agent_idle_timeout_sec=spec.agent_idle_timeout_sec,
        loop_strategy=spec.loop_strategy,
        reasoning_mode=spec.reasoning_mode,
        failure_type=failure_type,
    )


def _context_artifact_verified(spec: RunSpec) -> bool | None:
    if spec.condition == "no_skill":
        return None
    if spec.skills_dir is None or spec.rendered_context_sha256 is None:
        return None
    skill_path = spec.skills_dir / "retrieved-skills" / "SKILL.md"
    manifest_path = spec.skills_dir.parent / "context_manifest.json"
    if not skill_path.is_file() or not manifest_path.is_file():
        return False
    payload = skill_path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != spec.rendered_context_sha256:
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return (
        isinstance(manifest, dict)
        and manifest.get("rendered_context_sha256") == spec.rendered_context_sha256
        and manifest.get("atomic_payload_sha256") == spec.atomic_payload_sha256
    )


def _skill_prompt_evidence(spec: RunSpec) -> bool | None:
    evidence_names = {"prompts.json", "trajectory.json", "trajectory.jsonl"}
    needles = ["retrieved-skills", "Retrieved procedural skills"]
    if spec.condition != "no_skill" and spec.skills_dir is not None:
        skill_path = spec.skills_dir / "retrieved-skills" / "SKILL.md"
        if skill_path.is_file():
            context = skill_path.read_text(encoding="utf-8")
            marker = "## Atomic skill payloads\n"
            if marker in context:
                excerpt = context.split(marker, 1)[1][:200]
                needles.extend((excerpt, json.dumps(excerpt, ensure_ascii=False)[1:-1]))
    if not spec.jobs_dir.is_dir():
        return None
    inspected = False
    for path in spec.jobs_dir.rglob("*"):
        is_trajectory = "trajectory" in path.name and path.suffix in {".json", ".jsonl"}
        if not path.is_file() or (
            path.name not in evidence_names and not is_trajectory
        ):
            continue
        try:
            if path.stat().st_size > 20_000_000:
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        inspected = True
        if spec.condition == "no_skill":
            if any(needle in text for needle in needles[:2]):
                return True
        elif len(needles) >= 4 and any(needle in text for needle in needles[2:]):
            return True
    return False if inspected else None


def _add_timing_fallback(spec: RunSpec, result: dict[str, Any]) -> None:
    if isinstance(result.get("timing"), dict):
        return
    candidates = (
        list(spec.jobs_dir.rglob("timing.json")) if spec.jobs_dir.is_dir() else []
    )
    if len(candidates) != 1:
        return
    try:
        timing = json.loads(candidates[0].read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    if isinstance(timing, dict):
        result["timing"] = timing


def collect_one(spec: RunSpec, state_root: Path) -> TrajectoryResult:
    state_path = state_root / f"{spec.run_key}.json"
    state: dict[str, Any] = {}
    if state_path.is_file():
        loaded_state = json.loads(state_path.read_text(encoding="utf-8"))
        if isinstance(loaded_state, dict):
            state = loaded_state
        recorded_commit = state.get("skillsbench_commit")
        if (
            isinstance(recorded_commit, str)
            and recorded_commit != spec.skillsbench_commit
        ):
            return _error_row(
                spec, "infrastructure_error", "skillsbench_preflight_mismatch"
            )
        if state.get("failure_type") == "benchflow_agents_commit_mismatch":
            return _error_row(
                spec, "infrastructure_error", "benchflow_agents_commit_mismatch"
            )
    candidates = (
        list(spec.jobs_dir.rglob("result.json")) if spec.jobs_dir.is_dir() else []
    )
    if len(candidates) != 1:
        if state.get("status") == "process_error":
            failure_type = str(state.get("failure_type") or "bench_process_error")
            return _error_row(spec, "infrastructure_error", failure_type)
        kind = "missing_result_json" if not candidates else "multiple_result_json"
        return _error_row(spec, "missing_artifact", kind)
    try:
        value = json.loads(candidates[0].read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return _error_row(spec, "missing_artifact", "invalid_result_json")
    if not isinstance(value, dict):
        return _error_row(spec, "missing_artifact", "invalid_result_json")
    _add_timing_fallback(spec, value)
    row = classify_result(spec, value)
    context_verified = _context_artifact_verified(spec)
    prompt_evidence = _skill_prompt_evidence(spec)
    row = row.model_copy(
        update={
            "context_artifact_verified": context_verified,
            "skill_prompt_evidence": prompt_evidence,
        }
    )
    if context_verified is False:
        row = row.model_copy(
            update={
                "status": "infrastructure_error",
                "reward": None,
                "failure_type": "generated_context_mismatch",
            }
        )
    return row


def _mean(rows: Sequence[TrajectoryResult], field: str) -> float | None:
    values = [getattr(row, field) for row in rows]
    numeric = [float(value) for value in values if value is not None]
    return statistics.fmean(numeric) if numeric else None


def aggregate_results(rows: Sequence[TrajectoryResult]) -> dict[str, object]:
    by_condition: dict[str, list[TrajectoryResult]] = defaultdict(list)
    for row in rows:
        by_condition[row.condition].append(row)
    conditions: dict[str, object] = {}
    for condition in CONDITIONS:
        group = by_condition[condition]
        status_counts = Counter(row.status for row in group)
        passed = status_counts["passed"]
        valid = passed + status_counts["failed"]
        conditions[condition] = {
            "total": len(group),
            "status_counts": dict(sorted(status_counts.items())),
            "fixed_denominator_pass_rate": passed / len(group) if group else None,
            "valid_run_pass_rate": passed / valid if valid else None,
            "mean_reward_valid": _mean(
                [row for row in group if row.status in {"passed", "failed"}], "reward"
            ),
            "mean_total_tokens": _mean(group, "total_tokens"),
            "mean_input_tokens": _mean(group, "input_tokens"),
            "mean_output_tokens": _mean(group, "output_tokens"),
            "mean_wall_time_s": _mean(group, "wall_time_s"),
            "mean_agent_execution_time_s": _mean(group, "agent_execution_time_s"),
            "mean_verifier_time_s": _mean(group, "verifier_time_s"),
            "mean_tool_calls": _mean(group, "tool_calls"),
            "mean_skill_invocations": _mean(group, "skill_invocations"),
            "mean_trajectory_steps": _mean(group, "trajectory_steps"),
            "total_cost_usd": sum(row.cost_usd for row in group),
            "injection_verified": sum(row.injection_verified is True for row in group),
            "context_artifact_verified": sum(
                row.context_artifact_verified is True for row in group
            ),
            "skill_prompt_evidence": sum(
                row.skill_prompt_evidence is True for row in group
            ),
        }

    keyed = {(row.task_id, row.repeat_id, row.condition): row for row in rows}
    transitions: dict[str, dict[str, int]] = {}
    for treatment in ("flat_top8", "hierarchy_top8", "graph_top8"):
        counts: Counter[str] = Counter()
        for task_id, repeat_id, condition in keyed:
            if condition != "no_skill":
                continue
            baseline = keyed[(task_id, repeat_id, "no_skill")]
            treated = keyed.get((task_id, repeat_id, treatment))
            if treated is None:
                continue
            if baseline.status not in {"passed", "failed"} or treated.status not in {
                "passed",
                "failed",
            }:
                counts["unavailable"] += 1
                continue
            before = 1 if baseline.status == "passed" else 0
            after = 1 if treated.status == "passed" else 0
            counts[f"{before}->{after}"] += 1
        transitions[f"{treatment}-no_skill"] = dict(sorted(counts.items()))
    strata: dict[str, dict[str, object]] = {}
    for stratum in sorted({row.stratum for row in rows if row.stratum is not None}):
        stratum_rows = [row for row in rows if row.stratum == stratum]
        by_stratum_condition: dict[str, object] = {}
        for condition in CONDITIONS:
            group = [row for row in stratum_rows if row.condition == condition]
            passed = sum(row.status == "passed" for row in group)
            valid = sum(row.status in {"passed", "failed"} for row in group)
            by_stratum_condition[condition] = {
                "total": len(group),
                "fixed_denominator_pass_rate": passed / len(group) if group else None,
                "valid_run_pass_rate": passed / valid if valid else None,
                "mean_total_tokens": _mean(group, "total_tokens"),
                "mean_input_tokens": _mean(group, "input_tokens"),
                "mean_output_tokens": _mean(group, "output_tokens"),
                "mean_wall_time_s": _mean(group, "wall_time_s"),
                "mean_agent_execution_time_s": _mean(group, "agent_execution_time_s"),
                "mean_verifier_time_s": _mean(group, "verifier_time_s"),
                "mean_trajectory_steps": _mean(group, "trajectory_steps"),
            }
        strata[stratum] = by_stratum_condition
    return {"conditions": conditions, "strata": strata, "transitions": transitions}


def _write_task_matrix(path: Path, rows: Sequence[TrajectoryResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    grouped: dict[tuple[str, int], dict[str, TrajectoryResult]] = defaultdict(dict)
    for row in rows:
        grouped[(row.task_id, row.repeat_id)][row.condition] = row
    fields = ["task_id", "repeat_id"]
    for condition in CONDITIONS:
        fields.extend((f"{condition}_status", f"{condition}_reward"))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for key in sorted(grouped):
            task_id, repeat_id = key
            output: dict[str, object] = {"task_id": task_id, "repeat_id": repeat_id}
            for condition in CONDITIONS:
                row = grouped[key].get(condition)
                output[f"{condition}_status"] = row.status if row else "missing"
                output[f"{condition}_reward"] = (
                    "" if row is None or row.reward is None else row.reward
                )
            writer.writerow(output)


def collect_results(
    *, specs: Sequence[RunSpec], state_root: Path, output_root: Path
) -> tuple[TrajectoryResult, ...]:
    rows = tuple(collect_one(spec, state_root) for spec in specs)
    output_root.mkdir(parents=True, exist_ok=True)
    write_jsonl_atomic(
        output_root / "trajectories.jsonl",
        (row.model_dump(mode="json") for row in rows),
    )
    write_jsonl_atomic(
        output_root / "failures.jsonl",
        (
            row.model_dump(mode="json")
            for row in rows
            if row.status not in {"passed", "failed"}
        ),
    )
    _write_task_matrix(output_root / "task_matrix.csv", rows)
    write_json_atomic(output_root / "aggregate.json", aggregate_results(rows))
    return rows


def validate_smoke_gate(
    output_root: Path, specs: Sequence[RunSpec]
) -> tuple[TrajectoryResult, ...]:
    path = output_root / "trajectories.jsonl"
    if not path.is_file():
        raise ValueError("pilot requires collected smoke trajectories")
    rows = tuple(TrajectoryResult.model_validate(row) for row in stream_jsonl(path))
    if len(rows) != 8 or len({row.run_key for row in rows}) != 8:
        raise ValueError("pilot requires exactly eight unique smoke trajectories")
    expected = {spec.run_key: spec for spec in specs}
    if len(expected) != 8 or set(expected) != {row.run_key for row in rows}:
        raise ValueError("smoke trajectories do not match the registered smoke matrix")
    for row in rows:
        spec = expected[row.run_key]
        bindings = {
            "stage": spec.stage,
            "task_id": spec.task_id,
            "condition": spec.condition,
            "repeat_id": spec.repeat_id,
            "stratum": spec.stratum,
            "expected_skillsbench_commit": spec.skillsbench_commit,
            "agent": spec.agent,
            "model": spec.model,
            "sandbox": spec.sandbox,
            "rendered_context_sha256": spec.rendered_context_sha256,
            "atomic_payload_sha256": spec.atomic_payload_sha256,
            "predictions_sha256": spec.predictions_sha256,
            "skills_sha256": spec.skills_sha256,
            "task_catalog_sha256": spec.task_catalog_sha256,
            "benchflow_version": spec.benchflow_version,
            "benchflow_agents_commit": spec.benchflow_agents_commit,
            "sandbox_package_versions": spec.sandbox_package_versions,
            "agent_idle_timeout_sec": spec.agent_idle_timeout_sec,
            "loop_strategy": spec.loop_strategy,
            "reasoning_mode": spec.reasoning_mode,
        }
        mismatched = [
            field for field, value in bindings.items() if getattr(row, field) != value
        ]
        if mismatched:
            raise ValueError(
                f"smoke trajectory metadata differs from matrix for {row.run_key}: "
                f"{mismatched}"
            )
    if any(row.stage != "smoke" for row in rows):
        raise ValueError("smoke gate contains a non-smoke trajectory")
    counts = Counter(row.condition for row in rows)
    if counts != Counter({condition: 2 for condition in CONDITIONS}):
        raise ValueError("smoke gate must contain two runs per condition")
    uninterpretable = {"infrastructure_error", "missing_artifact", "verifier_error"}
    if any(row.status in uninterpretable for row in rows):
        raise ValueError(
            "smoke gate contains infrastructure, artifact, or verifier errors"
        )
    if any(row.injection_verified is not True for row in rows):
        raise ValueError(
            "smoke gate requires verified Skill mode and mount configuration"
        )
    for row in rows:
        if row.condition == "no_skill":
            if row.skill_prompt_evidence is not False:
                raise ValueError(
                    "no_skill smoke prompts must prove retrieved Skills are absent"
                )
        elif (
            row.context_artifact_verified is not True
            or row.skill_prompt_evidence is not True
        ):
            raise ValueError(
                "Skill smoke runs require matching context and prompt evidence"
            )
    return rows

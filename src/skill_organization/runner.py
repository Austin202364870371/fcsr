"""Deterministic, resumable BenchFlow run planning and execution."""

from __future__ import annotations

import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field

from data_io import stream_jsonl, write_jsonl_atomic
from skill_organization.inputs import HARD15_TASK_IDS


Condition = Literal["no_skill", "flat_top8", "hierarchy_top8", "graph_top8"]
RunStage = Literal["smoke", "pilot"]
CONDITIONS: tuple[Condition, ...] = (
    "no_skill",
    "flat_top8",
    "hierarchy_top8",
    "graph_top8",
)
EXPERIMENT_AGENT = "openhands"
EXPERIMENT_MODEL = "deepseek/deepseek-v4-flash"
EXPERIMENT_SANDBOX = "daytona"
EXPERIMENT_BENCHFLOW_VERSION = "benchflow 0.6.6"
EXPERIMENT_SANDBOX_PACKAGES = {
    "daytona": "0.203.0",
    "python-socketio": "5.16.4",
    "python-engineio": "4.13.4",
    "aiohttp": "3.14.3",
}


class RunSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    run_key: str = Field(min_length=1)
    stage: RunStage = "pilot"
    task_id: str = Field(min_length=1)
    condition: Condition
    repeat_id: int = Field(ge=1)
    order_index: int = Field(ge=0)
    task_dir: Path
    skills_dir: Path | None
    jobs_dir: Path
    agent: str = "openhands"
    model: str = "deepseek/deepseek-v4-flash"
    sandbox: str = "daytona"
    bench_bin: str = "bench"
    skillsbench_commit: str = "b63b7b2850226b6aa4fb5929a8c1ac7bc4d9a6af"
    rendered_context_sha256: str | None = None
    atomic_payload_sha256: str | None = None
    stratum: str | None = None
    predictions_sha256: str | None = None
    skills_sha256: str | None = None
    task_catalog_sha256: str | None = None
    benchflow_version: str | None = None
    benchflow_agents_commit: str | None = None
    sandbox_package_versions: dict[str, str] | None = None
    agent_idle_timeout_sec: int = 600
    loop_strategy: str = "single-shot"
    reasoning_mode: str = "provider-default"


def build_run_matrix(
    *,
    task_ids: Sequence[str],
    tasks_root: Path,
    generated_root: Path,
    jobs_root: Path,
    repeats: int = 1,
    agent: str = "openhands",
    model: str = "deepseek/deepseek-v4-flash",
    sandbox: str = "daytona",
    bench_bin: str = "bench",
    stage: RunStage = "pilot",
    skillsbench_commit: str = "b63b7b2850226b6aa4fb5929a8c1ac7bc4d9a6af",
    require_context_manifests: bool = False,
    task_strata: Mapping[str, str] | None = None,
    input_sha256: Mapping[str, str] | None = None,
    benchflow_version: str | None = None,
    benchflow_agents_commit: str | None = None,
    sandbox_package_versions: Mapping[str, str] | None = None,
) -> tuple[RunSpec, ...]:
    if repeats != 1:
        raise ValueError("the frozen experiment requires exactly one repeat")
    expected_runtime = (EXPERIMENT_AGENT, EXPERIMENT_MODEL, EXPERIMENT_SANDBOX)
    if (agent, model, sandbox) != expected_runtime:
        raise ValueError(
            "runtime is frozen to "
            f"agent={EXPERIMENT_AGENT}, model={EXPERIMENT_MODEL}, "
            f"sandbox={EXPERIMENT_SANDBOX}"
        )
    if not task_ids or len(set(task_ids)) != len(task_ids):
        raise ValueError("task IDs must be non-empty and unique")
    specs: list[RunSpec] = []
    order_index = 0
    for repeat_id in range(1, repeats + 1):
        for task_index, task_id in enumerate(task_ids):
            shift = (task_index + repeat_id - 1) % len(CONDITIONS)
            rotated = CONDITIONS[shift:] + CONDITIONS[:shift]
            for condition in rotated:
                run_key = f"{stage}__{task_id}__{condition}__r{repeat_id:02d}"
                skills_dir = (
                    None
                    if condition == "no_skill"
                    else generated_root / task_id / condition / "skills"
                )
                rendered_context_sha256: str | None = None
                atomic_payload_sha256: str | None = None
                if condition != "no_skill":
                    manifest_path = (
                        generated_root / task_id / condition / "context_manifest.json"
                    )
                    if manifest_path.is_file():
                        context_manifest = json.loads(
                            manifest_path.read_text(encoding="utf-8")
                        )
                        rendered_context_sha256 = context_manifest.get(
                            "rendered_context_sha256"
                        )
                        atomic_payload_sha256 = context_manifest.get(
                            "atomic_payload_sha256"
                        )
                        if not isinstance(
                            rendered_context_sha256, str
                        ) or not isinstance(atomic_payload_sha256, str):
                            raise ValueError(
                                f"invalid context manifest: {manifest_path}"
                            )
                    elif require_context_manifests:
                        raise ValueError(f"missing context manifest: {manifest_path}")
                specs.append(
                    RunSpec(
                        run_key=run_key,
                        stage=stage,
                        task_id=task_id,
                        condition=condition,
                        repeat_id=repeat_id,
                        order_index=order_index,
                        task_dir=tasks_root / task_id,
                        skills_dir=skills_dir,
                        jobs_dir=jobs_root / run_key,
                        agent=agent,
                        model=model,
                        sandbox=sandbox,
                        bench_bin=bench_bin,
                        skillsbench_commit=skillsbench_commit,
                        rendered_context_sha256=rendered_context_sha256,
                        atomic_payload_sha256=atomic_payload_sha256,
                        stratum=task_strata.get(task_id) if task_strata else None,
                        predictions_sha256=(
                            input_sha256.get("predictions") if input_sha256 else None
                        ),
                        skills_sha256=(
                            input_sha256.get("skills") if input_sha256 else None
                        ),
                        task_catalog_sha256=(
                            input_sha256.get("task_catalog") if input_sha256 else None
                        ),
                        benchflow_version=benchflow_version,
                        benchflow_agents_commit=benchflow_agents_commit,
                        sandbox_package_versions=(
                            dict(sandbox_package_versions)
                            if sandbox_package_versions is not None
                            else None
                        ),
                    )
                )
                order_index += 1
    return tuple(specs)


def write_run_matrix(path: Path, specs: Sequence[RunSpec]) -> int:
    payload = "".join(
        json.dumps(
            spec.model_dump(mode="json"), ensure_ascii=False, separators=(",", ":")
        )
        + "\n"
        for spec in specs
    )
    if path.is_file():
        if path.read_text(encoding="utf-8") == payload:
            return len(specs)
        raise ValueError(f"run matrix is immutable once written: {path}")
    return write_jsonl_atomic(path, (spec.model_dump(mode="json") for spec in specs))


def load_run_matrix(path: Path) -> tuple[RunSpec, ...]:
    specs = tuple(RunSpec.model_validate(row) for row in stream_jsonl(path))
    if len({spec.run_key for spec in specs}) != len(specs):
        raise ValueError("run matrix contains duplicate run keys")
    return specs


def validate_matrix_protocol(
    specs: Sequence[RunSpec],
    stage: RunStage,
    *,
    tasks_root: Path,
    generated_root: Path,
    jobs_root: Path,
) -> None:
    expected_tasks = 2 if stage == "smoke" else len(HARD15_TASK_IDS)
    expected_runs = expected_tasks * len(CONDITIONS)
    if len(specs) != expected_runs:
        raise ValueError(f"{stage} matrix requires exactly {expected_runs} runs")
    if any(spec.stage != stage or spec.repeat_id != 1 for spec in specs):
        raise ValueError(f"{stage} matrix has a stage or repeat mismatch")
    if any(
        (spec.agent, spec.model, spec.sandbox)
        != (EXPERIMENT_AGENT, EXPERIMENT_MODEL, EXPERIMENT_SANDBOX)
        for spec in specs
    ):
        raise ValueError("run matrix runtime differs from the frozen experiment")
    if any(
        spec.benchflow_version != EXPERIMENT_BENCHFLOW_VERSION
        or spec.sandbox_package_versions != EXPERIMENT_SANDBOX_PACKAGES
        for spec in specs
    ):
        raise ValueError("run matrix BenchFlow or Daytona versions are not frozen")

    task_ids = tuple(dict.fromkeys(spec.task_id for spec in specs))
    if len(task_ids) != expected_tasks:
        raise ValueError(
            f"{stage} matrix requires exactly {expected_tasks} unique tasks"
        )
    if stage == "pilot" and task_ids != HARD15_TASK_IDS:
        raise ValueError(
            "pilot matrix task IDs or order differ from registered Hard-15"
        )
    if stage == "smoke" and task_ids != (
        "jax-computing-basics",
        "citation-check",
    ):
        raise ValueError("smoke matrix must use the registered fixed task pair")

    expected_order: list[tuple[str, Condition, int]] = []
    for task_index, task_id in enumerate(task_ids):
        shift = task_index % len(CONDITIONS)
        for condition in CONDITIONS[shift:] + CONDITIONS[:shift]:
            expected_order.append((task_id, condition, len(expected_order)))
    actual_order = [(spec.task_id, spec.condition, spec.order_index) for spec in specs]
    if actual_order != expected_order:
        raise ValueError(f"{stage} matrix run order or order_index was modified")

    for spec in specs:
        expected_run_key = f"{stage}__{spec.task_id}__{spec.condition}__r01"
        if spec.run_key != expected_run_key:
            raise ValueError(
                f"derived run key mismatch for {spec.task_id}/{spec.condition}"
            )
        if spec.task_dir != tasks_root / spec.task_id:
            raise ValueError(f"task directory mismatch for {spec.run_key}")
        if spec.jobs_dir != jobs_root / expected_run_key:
            raise ValueError(f"jobs directory mismatch for {spec.run_key}")
        if spec.condition == "no_skill":
            if (
                spec.skills_dir is not None
                or spec.rendered_context_sha256 is not None
                or spec.atomic_payload_sha256 is not None
            ):
                raise ValueError("no_skill matrix rows cannot mount a Skill directory")
            continue

        expected_skills_dir = generated_root / spec.task_id / spec.condition / "skills"
        if spec.skills_dir != expected_skills_dir:
            raise ValueError(f"Skill directory mismatch for {spec.run_key}")
        manifest_path = expected_skills_dir.parent / "context_manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            raise ValueError(
                f"cannot read context manifest for {spec.run_key}"
            ) from None
        if spec.rendered_context_sha256 != manifest.get(
            "rendered_context_sha256"
        ) or spec.atomic_payload_sha256 != manifest.get("atomic_payload_sha256"):
            raise ValueError(f"frozen Skill context hash mismatch for {spec.run_key}")


def bench_command(
    spec: RunSpec,
    *,
    bench_bin: str | None = None,
    agent: str | None = None,
    model: str | None = None,
    sandbox: str | None = None,
) -> list[str]:
    selected_bench = bench_bin or spec.bench_bin
    selected_agent = agent or spec.agent
    selected_model = model or spec.model
    selected_sandbox = sandbox or spec.sandbox
    command = [
        selected_bench,
        "eval",
        "run",
        "--tasks-dir",
        str(spec.task_dir),
        "--agent",
        selected_agent,
        "--model",
        selected_model,
        "--sandbox",
        selected_sandbox,
        "--jobs-dir",
        str(spec.jobs_dir),
    ]
    if spec.condition == "no_skill":
        command.extend(("--skill-mode", "no-skill"))
    else:
        if spec.skills_dir is None:
            raise ValueError(f"skills_dir is required for {spec.condition}")
        command.extend(
            ("--skill-mode", "with-skill", "--skills-dir", str(spec.skills_dir))
        )
    return command


def _write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _credential_presence() -> dict[str, bool]:
    return {
        name: bool(os.environ.get(name))
        for name in (
            "DAYTONA_API_KEY",
            "LLM_API_KEY",
            "LLM_BASE_URL",
            "DEEPSEEK_API_KEY",
            "DEEPSEEK_BASE_URL",
            "MODAL_TOKEN_ID",
            "MODAL_TOKEN_SECRET",
        )
    }


def _agent_manifest_head(spec: RunSpec) -> str | None:
    repository = (
        spec.task_dir.parent.parent
        / ".cache"
        / "datasets"
        / "benchflow-ai"
        / "_agents_clone"
    )
    if not repository.is_dir():
        return None
    completed = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def _execute_one(
    spec: RunSpec,
    *,
    state_root: Path,
    log_root: Path,
) -> dict[str, object]:
    state_path = state_root / f"{spec.run_key}.json"
    if state_path.is_file():
        existing = json.loads(state_path.read_text(encoding="utf-8"))
        return existing

    command = bench_command(spec)
    log_path = log_root / f"{spec.run_key}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    running = {
        "run_key": spec.run_key,
        "status": "running",
        "command": command,
        "log_path": str(log_path),
        "skillsbench_commit": spec.skillsbench_commit,
        "credential_presence": _credential_presence(),
        "benchflow_version": spec.benchflow_version,
        "benchflow_agents_commit": spec.benchflow_agents_commit,
        "sandbox_package_versions": spec.sandbox_package_versions,
        "agent_idle_timeout_sec": spec.agent_idle_timeout_sec,
        "loop_strategy": spec.loop_strategy,
        "reasoning_mode": spec.reasoning_mode,
    }
    _write_json_atomic(state_path, running)
    before_agent_head = _agent_manifest_head(spec)
    if (
        spec.benchflow_agents_commit
        and before_agent_head != spec.benchflow_agents_commit
    ):
        final = {
            **running,
            "status": "process_error",
            "returncode": None,
            "failure_type": "benchflow_agents_commit_mismatch",
            "actual_benchflow_agents_commit": before_agent_head,
        }
        _write_json_atomic(state_path, final)
        return final
    try:
        with log_path.open("w", encoding="utf-8", newline="\n") as log:
            completed = subprocess.run(
                command,
                text=True,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
                cwd=spec.task_dir.parent.parent,
            )
    except OSError as exc:
        final = {
            **running,
            "status": "process_error",
            "returncode": None,
            "failure_type": "process_start_error",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        _write_json_atomic(state_path, final)
        return final
    after_agent_head = _agent_manifest_head(spec)
    manifest_changed = bool(
        spec.benchflow_agents_commit
        and after_agent_head != spec.benchflow_agents_commit
    )
    final = {
        **running,
        "status": (
            "completed"
            if completed.returncode == 0 and not manifest_changed
            else "process_error"
        ),
        "returncode": completed.returncode,
        "actual_benchflow_agents_commit": after_agent_head,
    }
    if manifest_changed:
        final["failure_type"] = "benchflow_agents_commit_mismatch"
    _write_json_atomic(state_path, final)
    return final


def execute_matrix(
    specs: Sequence[RunSpec],
    *,
    state_root: Path,
    log_root: Path,
    workers: int = 1,
) -> tuple[dict[str, object], ...]:
    if workers < 1:
        raise ValueError("workers must be at least one")
    state_root.mkdir(parents=True, exist_ok=True)
    kwargs = {
        "state_root": state_root,
        "log_root": log_root,
    }
    if workers == 1:
        return tuple(_execute_one(spec, **kwargs) for spec in specs)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_execute_one, spec, **kwargs) for spec in specs]
        return tuple(future.result() for future in futures)

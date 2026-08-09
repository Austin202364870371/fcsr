"""Preparation workflow for audited, reviewed Skill organization packages."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

from data_io import stream_jsonl, write_jsonl_atomic
from skill_organization.inputs import load_frozen_inputs, sha256_file
from skill_organization.models import FrozenInputs, TaskInput
from skill_organization.organizer import (
    OrganizerClient,
    OrganizationBundle,
    build_organizer_messages,
    validate_bundle,
)
from skill_organization.render import write_skill_packages
from skill_organization.validate import validate_rendered_task


EXPECTED_SKILLS_SHA256 = (
    "492bd8e7958434deeae97c91fbd6921aecefb19ea16d4605f100b645bec5af31"
)
EXPECTED_PREDICTIONS_SHA256 = (
    "82a03c683c7387028944faf37829a87563ef820a978029eaf786bd39c4bc800a"
)
EXPECTED_TASK_IDS_SHA256 = (
    "305390b815ce0460fbf9636f61beb8240449e8b0789abaa5f484f1f2115f1613"
)
EXPECTED_TASK_CATALOG_SHA256 = (
    "982c6e3d15103b94fc7cc16e55dac82438037601ac1b2f7fe795d024d08a15b0"
)
EXPECTED_SKILLSBENCH_COMMIT = "b63b7b2850226b6aa4fb5929a8c1ac7bc4d9a6af"
EXPECTED_HUGGINGFACE_REVISION = "be2a6ce2cb1f4ff67ce937307cade0c5a0477a13"
REPORT_FILES = ("predictions.json", "details.jsonl", "records.jsonl", "summary.json")
EXPECTED_REPORT_SHA256 = {
    "details.jsonl": "705ee76d58b57b8b12c177683bad14dce2f2f1fc45026be68b9be181c10b07c6",
    "predictions.json": EXPECTED_PREDICTIONS_SHA256,
    "records.jsonl": "fc4271b46441db052931008e166db9091926cde1155add4b7b8dd69b5b60f122",
    "summary.json": "d93bffe9c26e096d3fa91eb64fb4388dbd5788904ecdfc4ef38e90669b66ac8c",
}
EXPECTED_BENCHFLOW_VERSION = "benchflow 0.6.6"
EXPECTED_SANDBOX_PACKAGES = {
    "daytona": "0.203.0",
    "python-socketio": "5.16.4",
    "python-engineio": "4.13.4",
    "aiohttp": "3.14.3",
}


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_bytes(_json_bytes(value))
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def hash_json(value: object) -> str:
    canonical = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def organizer_credentials(
    *,
    api_key_env: str,
    base_url_env: str,
    model_env: str,
    default_model: str,
) -> tuple[str, str, str]:
    api_key = os.environ.get(api_key_env, "").strip()
    if not api_key:
        raise ValueError(
            f"missing organizer API key environment variable: {api_key_env}"
        )
    base_url = os.environ.get(base_url_env, "").strip()
    if not base_url:
        raise ValueError(
            f"missing organizer base URL environment variable: {base_url_env}"
        )
    model = os.environ.get(model_env, "").strip() or default_model
    return api_key, base_url, model


def audit_run(
    *,
    run_dir: Path,
    predictions_path: Path,
    skills_path: Path,
    task_ids_path: Path,
    task_catalog_path: Path,
    top_k: int = 8,
    expected_skills_sha256: str = EXPECTED_SKILLS_SHA256,
    expected_predictions_sha256: str | None = EXPECTED_PREDICTIONS_SHA256,
    expected_task_ids_sha256: str | None = EXPECTED_TASK_IDS_SHA256,
    expected_task_catalog_sha256: str | None = EXPECTED_TASK_CATALOG_SHA256,
    expected_report_sha256: Mapping[str, str] | None = EXPECTED_REPORT_SHA256,
) -> dict[str, object]:
    if run_dir.exists() and any(run_dir.iterdir()):
        raise ValueError(f"run directory is not empty: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    frozen = load_frozen_inputs(
        predictions_path=predictions_path,
        skills_path=skills_path,
        task_ids_path=task_ids_path,
        task_catalog_path=task_catalog_path,
        expected_skills_sha256=expected_skills_sha256,
        top_k=top_k,
    )
    if frozen.skillsbench_version != "v1.1":
        raise ValueError(
            f"expected SkillsBench v1.1, got {frozen.skillsbench_version!r}"
        )
    if frozen.skillsbench_commit != EXPECTED_SKILLSBENCH_COMMIT:
        raise ValueError(
            "expected SkillsBench v1.1 commit "
            f"{EXPECTED_SKILLSBENCH_COMMIT}, got {frozen.skillsbench_commit!r}"
        )
    if frozen.huggingface_revision != EXPECTED_HUGGINGFACE_REVISION:
        raise ValueError(
            "unexpected SkillsBench Hugging Face revision: "
            f"{frozen.huggingface_revision!r}"
        )
    expected_inputs = {
        "predictions": expected_predictions_sha256,
        "task_ids": expected_task_ids_sha256,
        "task_catalog": expected_task_catalog_sha256,
    }
    actual_inputs = {
        "predictions": frozen.predictions_sha256,
        "task_ids": frozen.task_ids_sha256,
        "task_catalog": frozen.task_catalog_sha256,
    }
    mismatched = [
        name
        for name, expected in expected_inputs.items()
        if expected is not None and actual_inputs[name].lower() != expected.lower()
    ]
    if mismatched:
        raise ValueError(f"registered experiment input SHA-256 mismatch: {mismatched}")
    report_dir = predictions_path.parent
    report_hashes = {
        name: sha256_file(report_dir / name)
        for name in REPORT_FILES
        if (report_dir / name).is_file()
    }
    if set(report_hashes) != set(REPORT_FILES):
        missing = sorted(set(REPORT_FILES) - set(report_hashes))
        raise ValueError(f"frozen report is incomplete: {missing}")
    if expected_report_sha256 is not None and report_hashes != dict(
        expected_report_sha256
    ):
        raise ValueError(
            "frozen report SHA-256 fingerprints differ from registered values"
        )

    inventory = [
        {
            "task_key": task.task_key,
            "task_id": task.task_id,
            "alias": skill.alias,
            "rank": skill.rank,
            "skill_id": skill.record.skill_id,
            "source": skill.record.source,
            "canonical_record_sha256": skill.record.canonical_hash(),
            "name_chars": len(skill.record.name),
            "description_chars": len(skill.record.description),
            "body_chars": len(skill.record.body),
        }
        for task in frozen.tasks
        for skill in task.skills
    ]
    unique_ids = {row["skill_id"] for row in inventory}
    sources = {
        source: len({row["skill_id"] for row in inventory if row["source"] == source})
        for source in ("pool", "gt", "distractor")
    }
    manifest: dict[str, object] = {
        "schema_version": "skill-organization-experiment-v1",
        "top_k": top_k,
        "skillsbench_version": frozen.skillsbench_version,
        "skillsbench_commit": frozen.skillsbench_commit,
        "huggingface_revision": frozen.huggingface_revision,
        "source_paths": {
            "predictions": str(predictions_path.resolve()),
            "skills": str(skills_path.resolve()),
            "task_ids": str(task_ids_path.resolve()),
            "task_catalog": str(task_catalog_path.resolve()),
        },
        "input_sha256": {
            "predictions": frozen.predictions_sha256,
            "skills": frozen.skills_sha256,
            "task_ids": frozen.task_ids_sha256,
            "task_catalog": frozen.task_catalog_sha256,
        },
        "report_sha256": report_hashes,
        "counts": {
            "tasks": len(frozen.tasks),
            "skill_instances": len(inventory),
            "unique_skills": len(unique_ids),
            "unique_skills_by_source": sources,
        },
        "task_keys": [task.task_key for task in frozen.tasks],
    }
    write_json_atomic(run_dir / "experiment_manifest.json", manifest)
    write_json_atomic(
        run_dir / "private" / "task_map.json",
        {task.task_key: task.task_id for task in frozen.tasks},
    )
    write_jsonl_atomic(
        run_dir / "preprocessing" / "frozen_skill_inventory.jsonl", inventory
    )
    return manifest


def load_audited_inputs(run_dir: Path) -> FrozenInputs:
    manifest_path = run_dir / "experiment_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    paths = manifest["source_paths"]
    hashes = manifest["input_sha256"]
    frozen = load_frozen_inputs(
        predictions_path=Path(paths["predictions"]),
        skills_path=Path(paths["skills"]),
        task_ids_path=Path(paths["task_ids"]),
        task_catalog_path=Path(paths["task_catalog"]),
        expected_skills_sha256=hashes["skills"],
        top_k=int(manifest["top_k"]),
    )
    current = {
        "predictions": frozen.predictions_sha256,
        "skills": frozen.skills_sha256,
        "task_ids": frozen.task_ids_sha256,
        "task_catalog": frozen.task_catalog_sha256,
    }
    if current != hashes:
        raise ValueError("audited input fingerprints changed")
    report_dir = Path(paths["predictions"]).parent
    report_hashes = {name: sha256_file(report_dir / name) for name in REPORT_FILES}
    if report_hashes != manifest["report_sha256"]:
        raise ValueError("frozen report fingerprints changed")
    return frozen


def _task_by_key(frozen: FrozenInputs, task_key: str) -> TaskInput:
    for task in frozen.tasks:
        if task.task_key == task_key:
            return task
    raise ValueError(f"unknown anonymous task key: {task_key}")


def _bundle_paths(run_dir: Path, task_key: str) -> tuple[Path, Path]:
    base = run_dir / "preprocessing"
    return base / "hierarchy" / f"{task_key}.json", base / "graph" / f"{task_key}.json"


def _write_reviewer_packet(
    run_dir: Path, task: TaskInput, bundle: OrganizationBundle
) -> None:
    root = run_dir / "reviewer_packet" / task.task_key
    write_json_atomic(
        root / "context.json",
        {"task_key": task.task_key, "skills": task.organizer_view()},
    )
    write_json_atomic(root / "hierarchy.json", bundle.hierarchy.model_dump(mode="json"))
    write_json_atomic(root / "graph.json", bundle.graph.model_dump(mode="json"))


def read_bundle(run_dir: Path, task_key: str) -> OrganizationBundle:
    hierarchy_path, graph_path = _bundle_paths(run_dir, task_key)
    return OrganizationBundle.model_validate(
        {
            "hierarchy": json.loads(hierarchy_path.read_text(encoding="utf-8")),
            "graph": json.loads(graph_path.read_text(encoding="utf-8")),
        }
    )


def organize_run(
    *,
    run_dir: Path,
    client: OrganizerClient,
    model: str,
    endpoint: str,
) -> dict[str, int]:
    frozen = load_audited_inputs(run_dir)
    created = 0
    reused = 0
    for task in frozen.tasks:
        hierarchy_path, graph_path = _bundle_paths(run_dir, task.task_key)
        response_path = (
            run_dir / "preprocessing" / "organizer_responses" / f"{task.task_key}.json"
        )
        if hierarchy_path.is_file() or graph_path.is_file() or response_path.is_file():
            if not (
                hierarchy_path.is_file()
                and graph_path.is_file()
                and response_path.is_file()
            ):
                raise ValueError(f"partial organizer checkpoint for {task.task_key}")
            bundle = read_bundle(run_dir, task.task_key)
            validate_bundle(task, bundle)
            _write_reviewer_packet(run_dir, task, bundle)
            reused += 1
            continue

        skills = task.organizer_view()
        messages = build_organizer_messages(task_key=task.task_key, skills=skills)
        request = {
            "task_key": task.task_key,
            "model": model,
            "endpoint": endpoint,
            "temperature": 0,
            "messages": messages,
            "prompt_sha256": hash_json(messages),
        }
        request_path = (
            run_dir / "preprocessing" / "organizer_requests" / f"{task.task_key}.json"
        )
        write_json_atomic(request_path, request)
        reply = client.organize(task_key=task.task_key, skills=skills)
        bundle = reply.parse_bundle()
        validate_bundle(task, bundle)
        write_json_atomic(hierarchy_path, bundle.hierarchy.model_dump(mode="json"))
        write_json_atomic(graph_path, bundle.graph.model_dump(mode="json"))
        write_json_atomic(
            response_path,
            {
                "task_key": task.task_key,
                "model": model,
                "endpoint": endpoint,
                "usage": reply.usage,
                "content": reply.content,
                "response_sha256": hashlib.sha256(
                    reply.content.encode("utf-8")
                ).hexdigest(),
            },
        )
        _write_reviewer_packet(run_dir, task, bundle)
        created += 1
    return {"created": created, "reused": reused}


def record_review(
    *,
    run_dir: Path,
    task_key: str,
    decision: str,
    reviewer: str,
    notes: str,
) -> dict[str, object]:
    return record_reviews(
        run_dir=run_dir,
        task_keys=(task_key,),
        decision=decision,
        reviewer=reviewer,
        notes=notes,
    )[0]


def record_reviews(
    *,
    run_dir: Path,
    task_keys: Sequence[str],
    decision: str,
    reviewer: str,
    notes: str,
) -> tuple[dict[str, object], ...]:
    if decision not in {"approve", "reject"}:
        raise ValueError("review decision must be approve or reject")
    if not task_keys or len(set(task_keys)) != len(task_keys):
        raise ValueError("review task keys must be non-empty and unique")
    frozen = load_audited_inputs(run_dir)
    rows: list[dict[str, object]] = []
    for task_key in task_keys:
        task = _task_by_key(frozen, task_key)
        bundle = read_bundle(run_dir, task_key)
        validate_bundle(task, bundle)
        hierarchy_path, graph_path = _bundle_paths(run_dir, task_key)
        rows.append(
            {
                "task_key": task_key,
                "decision": decision,
                "reviewer": reviewer,
                "notes": notes,
                "hierarchy_sha256": sha256_file(hierarchy_path),
                "graph_sha256": sha256_file(graph_path),
            }
        )
    review_path = run_dir / "preprocessing" / "review_events.jsonl"
    existing = list(stream_jsonl(review_path)) if review_path.is_file() else []
    previous_by_key = {str(item["task_key"]): item for item in existing}
    events: list[dict[str, object]] = []
    for row in rows:
        task_key = str(row["task_key"])
        previous = previous_by_key.get(task_key)
        event = {
            **row,
            "event_index": len(existing) + len(events) + 1,
            "reviewed_at_utc": datetime.now(timezone.utc).isoformat(),
            "previous_event_sha256": (
                hash_json(previous) if previous is not None else None
            ),
        }
        events.append(event)
        previous_by_key[task_key] = event
    write_jsonl_atomic(review_path, [*existing, *events])
    return tuple(events)


def _approved_reviews(run_dir: Path) -> dict[str, dict[str, object]]:
    path = run_dir / "preprocessing" / "review_events.jsonl"
    if not path.is_file():
        return {}
    latest: dict[str, dict[str, object]] = {}
    for row in stream_jsonl(path):
        latest[str(row["task_key"])] = row
    return {key: row for key, row in latest.items() if row.get("decision") == "approve"}


def render_reviewed(run_dir: Path) -> dict[str, int]:
    frozen = load_audited_inputs(run_dir)
    reviews = _approved_reviews(run_dir)
    expected = {task.task_key for task in frozen.tasks}
    if set(reviews) != expected:
        missing = sorted(expected - set(reviews))
        raise ValueError(
            f"all organization bundles require approval; missing={missing}"
        )
    generated_root = run_dir / "generated"
    for task in frozen.tasks:
        hierarchy_path, graph_path = _bundle_paths(run_dir, task.task_key)
        review = reviews[task.task_key]
        if review.get("hierarchy_sha256") != sha256_file(hierarchy_path):
            raise ValueError(f"reviewed hierarchy changed for {task.task_key}")
        if review.get("graph_sha256") != sha256_file(graph_path):
            raise ValueError(f"reviewed graph changed for {task.task_key}")
        bundle = read_bundle(run_dir, task.task_key)
        validate_bundle(task, bundle)
        write_skill_packages(task, bundle, generated_root)
    return {"tasks": len(frozen.tasks), "packages": len(frozen.tasks) * 3}


def validate_skillsbench_tasks(tasks_root: Path, task_ids: Sequence[str]) -> None:
    required_files = (Path("task.md"), Path("environment") / "Dockerfile")
    required_dirs = (Path("oracle"), Path("verifier"))
    for task_id in task_ids:
        task_root = tasks_root / task_id
        for relative in required_files:
            if not (task_root / relative).is_file():
                raise ValueError(
                    f"missing {relative.as_posix()} for SkillsBench task {task_id}"
                )
        for relative in required_dirs:
            if not (task_root / relative).is_dir():
                raise ValueError(
                    f"missing {relative.as_posix()}/ for SkillsBench task {task_id}"
                )


def validate_skillsbench_commit(tasks_root: Path, expected_commit: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(tasks_root.parent), "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise ValueError(
            f"cannot read SkillsBench git commit: {completed.stderr.strip()}"
        )
    actual = completed.stdout.strip()
    if actual != expected_commit:
        raise ValueError(
            f"SkillsBench commit mismatch: expected {expected_commit}, got {actual}"
        )
    return actual


def validate_run(run_dir: Path, tasks_root: Path) -> dict[str, object]:
    frozen = load_audited_inputs(run_dir)
    reviews = _approved_reviews(run_dir)
    expected_keys = {task.task_key for task in frozen.tasks}
    if set(reviews) != expected_keys:
        raise ValueError("validated packages require current approval for every task")
    validate_skillsbench_tasks(tasks_root, tuple(task.task_id for task in frozen.tasks))
    commit = validate_skillsbench_commit(tasks_root, frozen.skillsbench_commit)
    reports = []
    for task in frozen.tasks:
        hierarchy_path, graph_path = _bundle_paths(run_dir, task.task_key)
        review = reviews[task.task_key]
        if review.get("hierarchy_sha256") != sha256_file(hierarchy_path):
            raise ValueError(f"reviewed hierarchy changed for {task.task_key}")
        if review.get("graph_sha256") != sha256_file(graph_path):
            raise ValueError(f"reviewed graph changed for {task.task_key}")
        reports.append(
            validate_rendered_task(
                task, read_bundle(run_dir, task.task_key), run_dir / "generated"
            )
        )
    report = {
        "tasks": len(reports),
        "packages": len(reports) * 3,
        "skillsbench_commit": commit,
        "valid": True,
    }
    write_json_atomic(run_dir / "preprocessing" / "validation_report.json", report)
    return report


def _latest_summary(jobs_dir: Path) -> dict[str, object] | None:
    candidates = list(jobs_dir.rglob("summary.json")) if jobs_dir.is_dir() else []
    if not candidates:
        return None
    path = max(candidates, key=lambda candidate: candidate.stat().st_mtime_ns)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _bench_tool_python(bench_bin: str) -> Path:
    located = (
        shutil.which(bench_bin) if not Path(bench_bin).is_absolute() else bench_bin
    )
    if not located:
        raise ValueError(f"cannot locate BenchFlow executable: {bench_bin}")
    executable = Path(located).resolve()
    for name in ("python3", "python"):
        candidate = executable.parent / name
        if candidate.is_file():
            return candidate
    raise ValueError(f"cannot locate BenchFlow tool Python beside {executable}")


def _sandbox_package_versions(bench_bin: str) -> dict[str, str]:
    python = _bench_tool_python(bench_bin)
    versions: dict[str, str] = {}
    for package in EXPECTED_SANDBOX_PACKAGES:
        completed = subprocess.run(
            [
                str(python),
                "-c",
                "import importlib.metadata,sys;print(importlib.metadata.version(sys.argv[1]))",
                package,
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.returncode != 0:
            raise ValueError(
                f"cannot read {package} version from BenchFlow tool environment"
            )
        versions[package] = completed.stdout.strip()
    return versions


def run_oracle_preflight(
    *, run_dir: Path, tasks_root: Path, bench_bin: str = "bench"
) -> dict[str, object]:
    validation = validate_run(run_dir, tasks_root)
    version = subprocess.run(
        [bench_bin, "--version"], text=True, capture_output=True, check=False
    )
    benchflow_version = (version.stdout or version.stderr).strip()
    if version.returncode != 0 or benchflow_version != EXPECTED_BENCHFLOW_VERSION:
        raise ValueError(
            f"BenchFlow version mismatch: expected {EXPECTED_BENCHFLOW_VERSION}, "
            f"got {benchflow_version!r}"
        )
    sandbox_versions = _sandbox_package_versions(bench_bin)
    if sandbox_versions != EXPECTED_SANDBOX_PACKAGES:
        raise ValueError(
            f"Daytona runtime package mismatch: expected {EXPECTED_SANDBOX_PACKAGES}, "
            f"got {sandbox_versions}"
        )
    frozen = load_audited_inputs(run_dir)
    report_path = run_dir / "preflight" / "oracle_preflight.json"
    if report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if (
            report.get("benchflow_version") != benchflow_version
            or report.get("skillsbench_commit") != validation["skillsbench_commit"]
            or report.get("sandbox_package_versions") != sandbox_versions
            or report.get("tasks_root") != str(tasks_root.resolve())
        ):
            raise ValueError("existing oracle preflight uses a different runtime")
    else:
        report = {
            "schema_version": "oracle-preflight-v1",
            "benchflow_version": benchflow_version,
            "skillsbench_commit": validation["skillsbench_commit"],
            "sandbox": "daytona",
            "sandbox_package_versions": sandbox_versions,
            "tasks_root": str(tasks_root.resolve()),
            "tasks": {},
        }
    task_rows = report.get("tasks")
    if not isinstance(task_rows, dict):
        raise ValueError("invalid oracle preflight report")

    for task in frozen.tasks:
        existing = task_rows.get(task.task_id)
        if isinstance(existing, dict) and existing.get("passed") is True:
            continue
        jobs_dir = run_dir / "preflight" / "oracle_jobs" / task.task_id
        log_path = run_dir / "preflight" / "oracle_logs" / f"{task.task_id}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        command = [
            bench_bin,
            "eval",
            "run",
            "--tasks-dir",
            str(tasks_root / task.task_id),
            "--agent",
            "oracle",
            "--sandbox",
            "daytona",
            "--jobs-dir",
            str(jobs_dir),
        ]
        with log_path.open("w", encoding="utf-8", newline="\n") as log:
            check_completed = subprocess.run(
                [bench_bin, "tasks", "check", str(tasks_root / task.task_id)],
                cwd=tasks_root.parent,
                text=True,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
            if check_completed.returncode != 0:
                task_rows[task.task_id] = {
                    "passed": False,
                    "returncode": check_completed.returncode,
                    "failure_type": "bench_tasks_check",
                    "log_path": str(log_path),
                }
                write_json_atomic(report_path, report)
                raise ValueError(
                    f"BenchFlow task check failed for {task.task_id}; see {log_path}"
                )
            completed = subprocess.run(
                command,
                cwd=tasks_root.parent,
                text=True,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        summary = _latest_summary(jobs_dir)
        passed = bool(
            completed.returncode == 0
            and summary
            and summary.get("total") == 1
            and summary.get("passed") == 1
            and summary.get("errored") == 0
            and summary.get("verifier_errored") == 0
        )
        task_rows[task.task_id] = {
            "passed": passed,
            "returncode": completed.returncode,
            "summary": summary,
            "log_path": str(log_path),
        }
        write_json_atomic(report_path, report)
        if not passed:
            raise ValueError(
                f"oracle preflight failed for {task.task_id}; see {log_path}"
            )

    agents_repo = (
        tasks_root.parent / ".cache" / "datasets" / "benchflow-ai" / "_agents_clone"
    )
    if agents_repo.is_dir():
        agent_head = subprocess.run(
            ["git", "-C", str(agents_repo), "rev-parse", "HEAD"],
            text=True,
            capture_output=True,
            check=False,
        )
        report["benchflow_agents_commit"] = (
            agent_head.stdout.strip() if agent_head.returncode == 0 else None
        )
    report["completed"] = True
    write_json_atomic(report_path, report)
    return report


def require_oracle_preflight(run_dir: Path) -> dict[str, object]:
    path = run_dir / "preflight" / "oracle_preflight.json"
    if not path.is_file():
        raise ValueError("run oracle-preflight before planning agent runs")
    report = json.loads(path.read_text(encoding="utf-8"))
    tasks = report.get("tasks")
    frozen = load_audited_inputs(run_dir)
    expected = {task.task_id for task in frozen.tasks}
    passed = (
        {
            task_id
            for task_id, row in tasks.items()
            if isinstance(tasks, dict)
            and isinstance(row, dict)
            and row.get("passed") is True
        }
        if isinstance(tasks, dict)
        else set()
    )
    if (
        report.get("completed") is not True
        or report.get("benchflow_version") != EXPECTED_BENCHFLOW_VERSION
        or report.get("skillsbench_commit") != frozen.skillsbench_commit
        or report.get("sandbox_package_versions") != EXPECTED_SANDBOX_PACKAGES
        or not isinstance(report.get("tasks_root"), str)
        or not Path(str(report.get("tasks_root"))).is_dir()
        or passed != expected
    ):
        raise ValueError("oracle preflight is incomplete or uses a different runtime")
    return report

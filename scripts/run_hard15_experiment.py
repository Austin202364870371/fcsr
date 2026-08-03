"""Prepare and run the paired Flat/Hierarchy/Graph Hard-15 planning experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from agent.hard15_experiment import (
    compatible_completed,
    evaluate_attempts,
    experiment_fingerprint,
)
from agent.hard15_organizations import organize_task
from agent.hard15_pilot import Hard15Evaluation, Hard15Task, prepare_fixed_hard15
from agent.hard15_planning import PlanningAttempt, failed_attempt, plan_organized_task
from agent.llm import DeepSeekPlanningClient
from agent.task_catalog import load_pilot_catalog
from agent.task_packages import TaskSyncManifest, default_snapshot_downloader, sync_task_packages
from data_io import stream_jsonl, write_jsonl_atomic


PROJECT_ROOT = Path(__file__).resolve().parents[1]
METHODS = ("flat", "hierarchy", "graph")


def parse_args(argv=None):
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=Path, default=PROJECT_ROOT / "data/raw/evaluation_queries.jsonl.gz")
    parser.add_argument("--rankings", type=Path, default=PROJECT_ROOT / "reports/hard/fcsr/reranker_hard.jsonl")
    parser.add_argument("--skills", type=Path, default=PROJECT_ROOT / "data/raw/skills_hard.jsonl.gz")
    parser.add_argument("--catalog", type=Path, default=PROJECT_ROOT / "data/agent/hard15/task_catalog.json")
    parser.add_argument("--packages-dir", type=Path, default=PROJECT_ROOT / "data/agent/hard15/packages")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "reports/agent/hard15")
    parser.add_argument("--model", default=os.environ.get("DEEPSEEK_MODEL", "deepseek-chat"))
    parser.add_argument("--max-skills", type=int, default=8)
    parser.add_argument("--body-char-budget", type=int, default=12800)
    parser.add_argument("--max-groups", type=int, default=4)
    parser.add_argument("--sync", action="store_true", help="refresh public task context first")
    parser.add_argument("--dry-run", action="store_true", help="prepare all prompts without API calls")
    parser.add_argument("--skip-package-audit", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main(argv=None, *, client=None, downloader=default_snapshot_downloader) -> int:
    args = parse_args(argv)
    if args.max_skills < 1 or args.body_char_budget < 0 or args.max_groups < 1:
        raise ValueError("organization budgets are invalid")
    catalog = load_pilot_catalog(args.catalog)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.sync:
        manifest = sync_task_packages(catalog, args.packages_dir, downloader=downloader)
        _write_json(args.output_dir / "task_manifest.json", manifest.model_dump(mode="json"))
        if not manifest.ready:
            print("Task synchronization did not produce 15 planning-ready packages")
            return 2

    if not args.skip_package_audit:
        _validate_manifest(catalog, args.output_dir / "task_manifest.json")
    pilot = prepare_fixed_hard15(
        catalog,
        stream_jsonl(args.queries),
        stream_jsonl(args.rankings),
        stream_jsonl(args.skills),
        packages_root=None if args.skip_package_audit else args.packages_dir,
    )
    write_jsonl_atomic(
        args.output_dir / "tasks.jsonl",
        (task.model_dump(mode="json") for task in pilot.tasks),
    )
    write_jsonl_atomic(
        args.output_dir / "evaluation.jsonl",
        (item.model_dump(mode="json") for item in pilot.evaluations),
    )

    organized = {
        (task.task_id, method): organize_task(
            task,
            method=method,
            max_skills=args.max_skills,
            body_char_budget=args.body_char_budget,
            max_groups=args.max_groups,
        )
        for task in pilot.tasks
        for method in METHODS
    }
    for method in METHODS:
        write_jsonl_atomic(
            args.output_dir / "presentations" / f"{method}.jsonl",
            (
                organized[(task.task_id, method)].model_dump(mode="json")
                for task in pilot.tasks
            ),
        )
    preparation = {
        "status": "planning_prompts_ready",
        "result_type": "planning_only_not_task_success",
        "task_count": len(pilot.tasks),
        "method_count": len(METHODS),
        "planned_api_calls": len(pilot.tasks) * len(METHODS),
        "model": args.model,
        "max_skills": args.max_skills,
        "body_char_budget": args.body_char_budget,
        "max_groups": args.max_groups,
        "package_audit_skipped": args.skip_package_audit,
    }
    _write_json(args.output_dir / "preparation_summary.json", preparation)
    if args.dry_run:
        print(f"Prepared {len(pilot.tasks)} tasks and 45 organization prompts; no API calls made")
        return 0

    fingerprint = experiment_fingerprint(
        args.model,
        args.max_skills,
        args.body_char_budget,
        f"{catalog.github_commit}:{catalog.huggingface_revision}",
        max_groups=args.max_groups,
        input_digest=_pilot_digest(catalog, pilot),
    )
    existing_by_method = {
        method: _load_attempts(args.output_dir / "plans" / f"{method}.jsonl")
        for method in METHODS
    }
    completed_by_method = {
        method: compatible_completed(existing_by_method[method], fingerprint)
        for method in METHODS
    }
    if client is None:
        client = DeepSeekPlanningClient(
            api_key=os.environ.get("DEEPSEEK_API_KEY", ""),
            base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        )
    all_attempts: list[PlanningAttempt] = []
    for method in METHODS:
        path = args.output_dir / "plans" / f"{method}.jsonl"
        existing = existing_by_method[method]
        completed = completed_by_method[method]
        latest = {(item.task_id, item.method): item for item in existing}
        for task in pilot.tasks:
            key = (task.task_id, method)
            if key in completed:
                continue
            presentation = organized[key]
            try:
                attempt = plan_organized_task(
                    presentation,
                    client=client,
                    model=args.model,
                    fingerprint=fingerprint,
                )
            except Exception as exc:
                attempt = failed_attempt(
                    presentation,
                    model=args.model,
                    fingerprint=fingerprint,
                    error=exc,
                )
            latest[key] = attempt
            _write_method_checkpoint(path, method, pilot.tasks, latest)
            print(f"[{method}] {task.task_id}: {'valid' if attempt.valid else 'invalid'}")
        records = _ordered_method_records(method, pilot.tasks, latest)
        all_attempts.extend(records)
    summary = evaluate_attempts(all_attempts, pilot.evaluations)
    summary["fingerprint"] = fingerprint
    summary["configuration"] = preparation
    _write_json(args.output_dir / "summary.json", summary)
    valid = sum(item.valid for item in all_attempts)
    print(f"Completed {len(all_attempts)} paired planning attempts; valid={valid}")
    print(f"Summary: {args.output_dir / 'summary.json'}")
    return 0 if len(all_attempts) == 45 else 2


def _load_attempts(path: Path) -> list[PlanningAttempt]:
    if not path.is_file():
        return []
    return [PlanningAttempt.model_validate(row) for row in stream_jsonl(path)]


def _ordered_method_records(method, tasks, records):
    return [
        records[(task.task_id, method)]
        for task in tasks
        if (task.task_id, method) in records
    ]


def _write_method_checkpoint(path, method, tasks, records):
    ordered = _ordered_method_records(method, tasks, records)
    write_jsonl_atomic(path, (item.model_dump(mode="json") for item in ordered))


def _pilot_digest(catalog, pilot) -> str:
    payload = {
        "catalog": catalog.model_dump(mode="json"),
        "tasks": [
            {
                **task.model_dump(mode="json"),
                "stratum": task.stratum,
                "private_skill_structure": [
                    {"skill_id": skill.skill_id, "category_path": skill.category_path}
                    for skill in task.skills
                ],
            }
            for task in pilot.tasks
        ],
        "evaluations": [item.model_dump(mode="json") for item in pilot.evaluations],
        "prompt_schema": "hard15-json-plan-v2",
    }
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _validate_manifest(catalog, path: Path) -> None:
    if not path.is_file():
        raise ValueError("missing task_manifest.json; run once with --sync")
    manifest = TaskSyncManifest.model_validate_json(path.read_text(encoding="utf-8"))
    if manifest.repository != catalog.huggingface_repo:
        raise ValueError("task manifest repository differs from catalog")
    if manifest.revision != catalog.huggingface_revision:
        raise ValueError("task manifest revision differs from catalog")
    if manifest.catalog_commit != catalog.github_commit:
        raise ValueError("task manifest commit differs from catalog")
    if not manifest.ready:
        raise ValueError("task manifest is not planning-ready")
    task_ids = tuple(item.task_id for item in manifest.tasks)
    if task_ids != catalog.task_ids:
        raise ValueError("task manifest order differs from catalog")
    for manifest_item, catalog_item in zip(manifest.tasks, catalog.tasks):
        if manifest_item.source_path != catalog_item.source_path:
            raise ValueError(f"{catalog_item.task_id}: source path differs from catalog")
        if manifest_item.expected_stratum != catalog_item.stratum:
            raise ValueError(f"{catalog_item.task_id}: stratum differs from catalog")

def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(f"{path}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


if __name__ == "__main__":
    raise SystemExit(main())

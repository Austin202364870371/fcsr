"""Audit hard task packages and prepare an anonymized 15-task Agent pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agent.environment_audit import audit_task_environment
from agent.hard_pilot import InsufficientEligibleTasks, prepare_hard_pilot
from data_io import load_jsonl, stream_jsonl, write_jsonl_atomic


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--rankings", type=Path, required=True)
    parser.add_argument("--skills", type=Path, required=True)
    parser.add_argument("--task-environments", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    queries = load_jsonl(args.queries)
    audits = [
        audit_task_environment(record["query_id"], args.task_environments)
        for record in queries
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl_atomic(
        args.output_dir / "environment_audit.jsonl",
        (audit.model_dump(mode="json") for audit in audits),
    )
    eligible = {audit.task_id for audit in audits if audit.execution_ready}
    summary = {
        "total_tasks": len(audits),
        "execution_ready": len(eligible),
        "skipped": len(audits) - len(eligible),
        "seed": args.seed,
    }
    try:
        pilot = prepare_hard_pilot(
            queries,
            stream_jsonl(args.rankings),
            stream_jsonl(args.skills),
            eligible_task_ids=eligible,
            seed=args.seed,
        )
    except InsufficientEligibleTasks as exc:
        summary.update(status="insufficient_eligible_tasks", reason=str(exc))
        _write_summary(args.output_dir / "summary.json", summary)
        print(str(exc))
        return 2
    write_jsonl_atomic(
        args.output_dir / "tasks.jsonl",
        (task.model_dump(mode="json") for task in pilot.public_tasks),
    )
    write_jsonl_atomic(
        args.output_dir / "evaluation.jsonl",
        (record.model_dump(mode="json") for record in pilot.evaluations),
    )
    summary.update(status="ready", selected=len(pilot.public_tasks))
    _write_summary(args.output_dir / "summary.json", summary)
    print(f"Prepared {len(pilot.public_tasks)} hard pilot tasks")
    return 0


def _write_summary(path: Path, summary: dict) -> None:
    path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())

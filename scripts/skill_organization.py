"""Prepare and execute the Hard-15 Skill organization experiment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
DEFAULT_SMOKE_TASKS = ("jax-computing-basics", "citation-check")
if str(SRC_ROOT) in sys.path:
    sys.path.remove(str(SRC_ROOT))
sys.path.insert(0, str(SRC_ROOT))

from skill_organization.organizer import DeepSeekOrganizerClient
from skill_organization.runner import (
    RunSpec,
    build_run_matrix,
    execute_matrix,
    load_run_matrix,
    validate_matrix_protocol,
    write_run_matrix,
)
from skill_organization.results import collect_results, validate_smoke_gate
from skill_organization.workflow import (
    audit_run,
    organize_run,
    organizer_credentials,
    record_review,
    record_reviews,
    require_oracle_preflight,
    render_reviewed,
    run_oracle_preflight,
    validate_run,
)


def _path(value: str) -> Path:
    return Path(value).expanduser()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit = subparsers.add_parser("audit", help="freeze and fingerprint Hard-15 inputs")
    audit.add_argument("--output", type=_path, required=True)
    audit.add_argument(
        "--predictions",
        type=_path,
        default=PROJECT_ROOT
        / "reports/reranker/hard/fcsr-multiskill3x-rrf/predictions.json",
    )
    audit.add_argument(
        "--skills", type=_path, default=PROJECT_ROOT / "data/raw/skills_hard.jsonl.gz"
    )
    audit.add_argument(
        "--task-ids",
        type=_path,
        default=PROJECT_ROOT / "data/agent/hard15/task_ids.txt",
    )
    audit.add_argument(
        "--task-catalog",
        type=_path,
        default=PROJECT_ROOT / "data/agent/hard15/task_catalog.json",
    )
    audit.add_argument("--top-k", type=int, default=8)

    organize = subparsers.add_parser(
        "organize", help="generate task-blind hierarchy and graph"
    )
    organize.add_argument("--run-dir", type=_path, required=True)
    organize.add_argument("--api-key-env", default="LLM_API_KEY")
    organize.add_argument("--base-url-env", default="LLM_BASE_URL")
    organize.add_argument("--model-env", default="DEEPSEEK_ORGANIZER_MODEL")
    organize.add_argument("--default-model", default="deepseek-v4-flash")

    review = subparsers.add_parser("review", help="record task-blind human review")
    review.add_argument("--run-dir", type=_path, required=True)
    review_target = review.add_mutually_exclusive_group(required=True)
    review_target.add_argument("--task-key")
    review_target.add_argument("--all", dest="all_tasks", action="store_true")
    review.add_argument("--decision", choices=("approve", "reject"), required=True)
    review.add_argument("--reviewer", required=True)
    review.add_argument("--notes", default="")

    render = subparsers.add_parser("render", help="render all approved Skill packages")
    render.add_argument("--run-dir", type=_path, required=True)

    validate = subparsers.add_parser(
        "validate", help="validate packages and SkillsBench checkout"
    )
    validate.add_argument("--run-dir", type=_path, required=True)
    validate.add_argument("--tasks-root", type=_path, required=True)

    preflight = subparsers.add_parser(
        "oracle-preflight", help="verify all Hard-15 tasks and verifiers on Daytona"
    )
    preflight.add_argument("--run-dir", type=_path, required=True)
    preflight.add_argument("--tasks-root", type=_path, required=True)
    preflight.add_argument("--bench-bin", default="bench")

    plan = subparsers.add_parser(
        "plan-runs", help="write the rotated BenchFlow run matrix"
    )
    plan.add_argument("--run-dir", type=_path, required=True)
    plan.add_argument("--tasks-root", type=_path, required=True)
    plan.add_argument("--stage", choices=("smoke", "pilot"), required=True)
    plan.add_argument("--bench-bin", default="bench")

    run = subparsers.add_parser("run", help="execute a prepared run matrix")
    run.add_argument("--run-dir", type=_path, required=True)
    run.add_argument("--workers", type=int, default=1)
    run.add_argument("--stage", choices=("smoke", "pilot"), required=True)

    collect = subparsers.add_parser(
        "collect", help="collect BenchFlow verifier and efficiency results"
    )
    collect.add_argument("--run-dir", type=_path, required=True)
    collect.add_argument("--stage", choices=("smoke", "pilot"), required=True)
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def _read_task_ids_from_manifest(run_dir: Path) -> tuple[str, ...]:
    mapping = json.loads(
        (run_dir / "private" / "task_map.json").read_text(encoding="utf-8")
    )
    return tuple(mapping[key] for key in sorted(mapping))


def _read_task_strata(experiment_manifest: dict[str, object]) -> dict[str, str]:
    source_paths = experiment_manifest["source_paths"]
    catalog_path = Path(source_paths["task_catalog"])
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    return {
        str(task["task_id"]): str(task["stratum"])
        for task in catalog["tasks"]
        if isinstance(task, dict) and "task_id" in task and "stratum" in task
    }


def _validate_registered_matrix(
    *,
    run_dir: Path,
    specs: Sequence[RunSpec],
    stage: str,
    experiment_manifest: dict[str, object],
    oracle_report: dict[str, object],
) -> None:
    tasks_root = Path(str(oracle_report["tasks_root"]))
    validate_matrix_protocol(
        specs,
        stage,
        tasks_root=tasks_root,
        generated_root=run_dir / "generated",
        jobs_root=run_dir / "jobs" / stage,
    )
    input_sha256 = experiment_manifest["input_sha256"]
    expected_strata = _read_task_strata(experiment_manifest)
    for spec in specs:
        if spec.skillsbench_commit != experiment_manifest["skillsbench_commit"]:
            raise ValueError(f"SkillsBench commit changed in matrix row {spec.run_key}")
        if (
            spec.predictions_sha256 != input_sha256["predictions"]
            or spec.skills_sha256 != input_sha256["skills"]
            or spec.task_catalog_sha256 != input_sha256["task_catalog"]
        ):
            raise ValueError(f"frozen input hash changed in matrix row {spec.run_key}")
        if spec.stratum != expected_strata.get(spec.task_id):
            raise ValueError(f"task stratum changed in matrix row {spec.run_key}")
        if (
            spec.benchflow_version != oracle_report["benchflow_version"]
            or spec.benchflow_agents_commit
            != oracle_report.get("benchflow_agents_commit")
            or spec.sandbox_package_versions
            != oracle_report["sandbox_package_versions"]
        ):
            raise ValueError(f"preflight runtime changed in matrix row {spec.run_key}")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.command == "audit":
            result = audit_run(
                run_dir=args.output,
                predictions_path=args.predictions,
                skills_path=args.skills,
                task_ids_path=args.task_ids,
                task_catalog_path=args.task_catalog,
                top_k=args.top_k,
            )
        elif args.command == "organize":
            api_key, base_url, model = organizer_credentials(
                api_key_env=args.api_key_env,
                base_url_env=args.base_url_env,
                model_env=args.model_env,
                default_model=args.default_model,
            )
            result = organize_run(
                run_dir=args.run_dir,
                client=DeepSeekOrganizerClient(
                    api_key=api_key, base_url=base_url, model=model
                ),
                model=model,
                endpoint=base_url,
            )
        elif args.command == "review":
            if args.all_tasks:
                manifest = json.loads(
                    (args.run_dir / "experiment_manifest.json").read_text(
                        encoding="utf-8"
                    )
                )
                rows = record_reviews(
                    run_dir=args.run_dir,
                    task_keys=tuple(sorted(manifest["task_keys"])),
                    decision=args.decision,
                    reviewer=args.reviewer,
                    notes=args.notes,
                )
                result = {"reviewed": len(rows), "decision": args.decision}
            else:
                result = record_review(
                    run_dir=args.run_dir,
                    task_key=args.task_key,
                    decision=args.decision,
                    reviewer=args.reviewer,
                    notes=args.notes,
                )
        elif args.command == "render":
            result = render_reviewed(args.run_dir)
        elif args.command == "validate":
            result = validate_run(args.run_dir, args.tasks_root)
        elif args.command == "oracle-preflight":
            result = run_oracle_preflight(
                run_dir=args.run_dir,
                tasks_root=args.tasks_root,
                bench_bin=args.bench_bin,
            )
        elif args.command == "plan-runs":
            experiment_manifest = json.loads(
                (args.run_dir / "experiment_manifest.json").read_text(encoding="utf-8")
            )
            available = _read_task_ids_from_manifest(args.run_dir)
            if args.stage == "smoke":
                task_ids = DEFAULT_SMOKE_TASKS
            else:
                task_ids = available
            if args.stage == "smoke" and len(task_ids) != 2:
                raise ValueError("smoke stage requires exactly two task IDs")
            if args.stage == "pilot" and task_ids != available:
                raise ValueError(
                    "pilot stage must use the complete audited Hard-15 in fixed order"
                )
            oracle_report = require_oracle_preflight(args.run_dir)
            if (
                args.tasks_root.resolve()
                != Path(str(oracle_report["tasks_root"])).resolve()
            ):
                raise ValueError("plan-runs tasks root differs from oracle preflight")
            if args.stage == "pilot":
                smoke_specs = load_run_matrix(args.run_dir / "run_matrix_smoke.jsonl")
                _validate_registered_matrix(
                    run_dir=args.run_dir,
                    specs=smoke_specs,
                    stage="smoke",
                    experiment_manifest=experiment_manifest,
                    oracle_report=oracle_report,
                )
                validate_smoke_gate(args.run_dir / "results" / "smoke", smoke_specs)
            validate_run(args.run_dir, args.tasks_root)
            specs = build_run_matrix(
                task_ids=task_ids,
                tasks_root=args.tasks_root,
                generated_root=args.run_dir / "generated",
                jobs_root=args.run_dir / "jobs" / args.stage,
                repeats=1,
                bench_bin=args.bench_bin,
                stage=args.stage,
                skillsbench_commit=experiment_manifest["skillsbench_commit"],
                require_context_manifests=True,
                task_strata=_read_task_strata(experiment_manifest),
                input_sha256=experiment_manifest["input_sha256"],
                benchflow_version=str(oracle_report["benchflow_version"]),
                benchflow_agents_commit=(
                    str(oracle_report["benchflow_agents_commit"])
                    if oracle_report.get("benchflow_agents_commit")
                    else None
                ),
                sandbox_package_versions=oracle_report["sandbox_package_versions"],
            )
            _validate_registered_matrix(
                run_dir=args.run_dir,
                specs=specs,
                stage=args.stage,
                experiment_manifest=experiment_manifest,
                oracle_report=oracle_report,
            )
            path = args.run_dir / f"run_matrix_{args.stage}.jsonl"
            write_run_matrix(path, specs)
            result = {"runs": len(specs), "path": str(path)}
        elif args.command == "run":
            experiment_manifest = json.loads(
                (args.run_dir / "experiment_manifest.json").read_text(encoding="utf-8")
            )
            oracle_report = require_oracle_preflight(args.run_dir)
            specs = load_run_matrix(args.run_dir / f"run_matrix_{args.stage}.jsonl")
            if not specs or any(spec.stage != args.stage for spec in specs):
                raise ValueError("run matrix stage mismatch")
            _validate_registered_matrix(
                run_dir=args.run_dir,
                specs=specs,
                stage=args.stage,
                experiment_manifest=experiment_manifest,
                oracle_report=oracle_report,
            )
            if args.stage == "pilot":
                smoke_specs = load_run_matrix(args.run_dir / "run_matrix_smoke.jsonl")
                _validate_registered_matrix(
                    run_dir=args.run_dir,
                    specs=smoke_specs,
                    stage="smoke",
                    experiment_manifest=experiment_manifest,
                    oracle_report=oracle_report,
                )
                validate_smoke_gate(args.run_dir / "results" / "smoke", smoke_specs)
            validate_run(args.run_dir, Path(str(oracle_report["tasks_root"])))
            states = execute_matrix(
                specs,
                state_root=args.run_dir / "run_states" / args.stage,
                log_root=args.run_dir / "logs" / args.stage,
                workers=args.workers,
            )
            result = {
                "runs": len(states),
                "completed": sum(state["status"] == "completed" for state in states),
                "process_errors": sum(
                    state["status"] == "process_error" for state in states
                ),
            }
        elif args.command == "collect":
            experiment_manifest = json.loads(
                (args.run_dir / "experiment_manifest.json").read_text(encoding="utf-8")
            )
            oracle_report = require_oracle_preflight(args.run_dir)
            specs = load_run_matrix(args.run_dir / f"run_matrix_{args.stage}.jsonl")
            _validate_registered_matrix(
                run_dir=args.run_dir,
                specs=specs,
                stage=args.stage,
                experiment_manifest=experiment_manifest,
                oracle_report=oracle_report,
            )
            rows = collect_results(
                specs=specs,
                state_root=args.run_dir / "run_states" / args.stage,
                output_root=args.run_dir / "results" / args.stage,
            )
            result = {
                "runs": len(rows),
                "passed": sum(row.status == "passed" for row in rows),
                "failed": sum(row.status == "failed" for row in rows),
                "errors": sum(row.status not in {"passed", "failed"} for row in rows),
            }
        else:
            raise AssertionError(f"unhandled command: {args.command}")
    except (FileNotFoundError, KeyError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

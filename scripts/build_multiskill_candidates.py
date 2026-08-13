"""Build Contract-guided multi-Skill candidates without calling an LLM."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from multiskill_candidates import CandidateSettings, build_compositional_candidates
from data_io import stream_jsonl, write_jsonl_atomic
from preprocessing import collect_benchmark_skill_ids


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="build Contract-guided pair and triple candidates for multi-Skill tasks"
    )
    parser.add_argument(
        "--contracts",
        type=Path,
        default=Path("data/contracts_32k_prompt006/contracts.jsonl.gz"),
    )
    parser.add_argument(
        "--queries",
        type=Path,
        default=Path("data/synthetic/single_skill_v1/queries.jsonl.gz"),
    )
    parser.add_argument("--tasks", type=Path, default=Path("data/raw/evaluation_queries.jsonl.gz"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/synthetic/multiskill_v1/candidates.jsonl.gz"),
    )
    parser.add_argument(
        "--rejections",
        type=Path,
        default=Path("data/synthetic/multiskill_v1/candidate_rejections.jsonl.gz"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/synthetic/multiskill_v1/manifest.json"),
    )
    parser.add_argument("--max-pairs", type=int, default=7342)
    parser.add_argument("--max-triples", type=int, default=1000)
    parser.add_argument("--max-pairs-per-source", type=int, default=16)
    parser.add_argument("--max-artifact-frequency", type=int, default=5)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    for output in (args.output, args.rejections):
        if output.exists() and not args.overwrite:
            raise FileExistsError(f"output exists; pass --overwrite to replace it: {output}")

    result = build_compositional_candidates(
        stream_jsonl(args.contracts),
        stream_jsonl(args.queries),
        collect_benchmark_skill_ids(args.tasks),
        CandidateSettings(
            max_pairs=args.max_pairs,
            max_triples=args.max_triples,
            max_pairs_per_source=args.max_pairs_per_source,
            max_artifact_frequency=args.max_artifact_frequency,
        ),
    )
    candidate_count = write_jsonl_atomic(args.output, result.candidates)
    rejection_count = write_jsonl_atomic(args.rejections, result.rejections)
    summary = {
        "eligible_skills": len(result.eligible_skill_ids),
        "pairs": len(result.pairs),
        "triples": len(result.triples),
        "candidates": candidate_count,
        "rejections": rejection_count,
        "output": args.output.as_posix(),
        "rejections_output": args.rejections.as_posix(),
    }
    _update_manifest(args.manifest, summary, args)
    return summary


def _update_manifest(
    path: Path,
    summary: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    manifest = {}
    if path.exists():
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError(f"manifest must be a JSON object: {path}")
        manifest = value
    manifest["status"] = "candidates_generated"
    manifest["candidate_construction"] = {
        "contracts": args.contracts.as_posix(),
        "single_skill_queries": args.queries.as_posix(),
        "benchmark_tasks": args.tasks.as_posix(),
        "max_pairs": args.max_pairs,
        "max_triples": args.max_triples,
        "max_pairs_per_source": args.max_pairs_per_source,
        "max_artifact_frequency": args.max_artifact_frequency,
        **summary,
    }
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        artifacts = {}
        manifest["artifacts"] = artifacts
    artifacts["candidates"] = {"path": args.output.name, "records": summary["candidates"]}
    artifacts["candidate_rejections"] = {
        "path": args.rejections.name,
        "records": summary["rejections"],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    summary = run(build_parser().parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

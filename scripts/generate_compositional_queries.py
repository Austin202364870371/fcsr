"""Generate validated multi-Skill synthetic queries with a local Qwen model."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from compositional_generation import (
    CompletionClient,
    CompositionalGenerationConfig,
    TransformersJsonClient,
    generate_compositional_queries,
)
from data_io import stream_jsonl, write_jsonl_atomic


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="generate strictly validated multi-Skill queries with a local model"
    )
    parser.add_argument(
        "--candidates",
        type=Path,
        default=Path("data/synthetic/compositional_v1/candidates.jsonl.gz"),
    )
    parser.add_argument(
        "--contracts", type=Path, default=Path("data/contracts/contracts.jsonl.gz")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/synthetic/compositional_v1/compositional_queries.jsonl.gz"),
    )
    parser.add_argument(
        "--failures",
        type=Path,
        default=Path("data/synthetic/compositional_v1/failures.jsonl.gz"),
    )
    parser.add_argument(
        "--review-queue",
        type=Path,
        default=Path("data/synthetic/compositional_v1/review_queue.jsonl.gz"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/synthetic/compositional_v1/manifest.json"),
    )
    parser.add_argument("--model", default="models/Qwen3-8B")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--min-query-words", type=int, default=30)
    parser.add_argument("--max-query-words", type=int, default=260)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def run(args: argparse.Namespace, client: CompletionClient | None = None) -> dict[str, Any]:
    _validate_inputs(args)
    candidates = list(itertools.islice(stream_jsonl(args.candidates), args.limit))
    if args.dry_run:
        return {
            "dry_run": True,
            "candidates": len(candidates),
            "contracts": sum(1 for _ in stream_jsonl(args.contracts)),
            "model": args.model,
            "output": args.output.as_posix(),
        }
    if client is None and not Path(args.model).is_dir():
        raise FileNotFoundError(
            f"local model directory does not exist: {args.model}; download it before submitting GPU work"
        )
    generator = client or TransformersJsonClient(args.model, args.device)
    result = generate_compositional_queries(
        candidates,
        stream_jsonl(args.contracts),
        generator,
        CompositionalGenerationConfig(
            model=args.model,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
            max_attempts=args.max_attempts,
            min_query_words=args.min_query_words,
            max_query_words=args.max_query_words,
        ),
    )
    query_count = write_jsonl_atomic(args.output, result.queries)
    failure_count = write_jsonl_atomic(args.failures, result.failures)
    review_count = write_jsonl_atomic(args.review_queue, result.review_queue)
    summary = {
        "candidates": len(candidates),
        "queries": query_count,
        "failures": failure_count,
        "review_queue": review_count,
        "output": args.output.as_posix(),
        "failures_output": args.failures.as_posix(),
        "review_queue_output": args.review_queue.as_posix(),
    }
    _update_manifest(args.manifest, summary, args)
    return summary


def _validate_inputs(args: argparse.Namespace) -> None:
    for path in (args.candidates, args.contracts):
        if not path.is_file():
            raise FileNotFoundError(f"input file does not exist: {path}")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("limit must be positive")
    if not args.overwrite:
        existing = [
            path
            for path in (args.output, args.failures, args.review_queue)
            if path.exists()
        ]
        if existing:
            raise FileExistsError(
                "generation outputs exist; pass --overwrite to replace them: "
                + ", ".join(str(path) for path in existing)
            )


def _update_manifest(path: Path, summary: dict[str, Any], args: argparse.Namespace) -> None:
    manifest: dict[str, Any] = {}
    if path.exists():
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError(f"manifest must be a JSON object: {path}")
        manifest = value
    manifest["status"] = (
        "queries_generated" if summary["failures"] == 0 else "queries_generated_with_failures"
    )
    manifest["query_generation"] = {
        "provider": "local_transformers",
        "model": args.model,
        "device": args.device,
        "temperature": args.temperature,
        "max_new_tokens": args.max_new_tokens,
        "max_attempts": args.max_attempts,
        "min_query_words": args.min_query_words,
        "max_query_words": args.max_query_words,
        "limit": args.limit,
        "candidates": args.candidates.as_posix(),
        "contracts": args.contracts.as_posix(),
        **summary,
    }
    artifacts = manifest.setdefault("artifacts", {})
    if not isinstance(artifacts, dict):
        raise ValueError("manifest artifacts must be a JSON object")
    artifacts["queries"] = {"path": args.output.name, "records": summary["queries"]}
    artifacts["failures"] = {"path": args.failures.name, "records": summary["failures"]}
    artifacts["review_queue"] = {
        "path": args.review_queue.name,
        "records": summary["review_queue"],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    summary = run(build_parser().parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
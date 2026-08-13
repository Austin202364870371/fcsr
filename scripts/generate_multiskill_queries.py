"""Generate validated multi-Skill synthetic queries with DeepSeek."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Any, Callable

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from multiskill_generation import (
    CompletionClient,
    CompositionalGenerationConfig,
    CompositionalGenerationProgress,
    generate_compositional_queries,
)
from data_io import stream_jsonl, write_jsonl_atomic
from deepseek_client import DEFAULT_MODEL, DeepSeekJsonClient


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="generate strictly validated multi-Skill queries with DeepSeek"
    )
    parser.add_argument(
        "--candidates",
        type=Path,
        default=Path("data/synthetic/multiskill_v1/candidates.jsonl.gz"),
    )
    parser.add_argument(
        "--contracts",
        type=Path,
        default=Path("data/contracts_32k_prompt006/contracts.jsonl.gz"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/synthetic/multiskill_v1/queries.jsonl.gz"),
    )
    parser.add_argument(
        "--failures",
        type=Path,
        default=Path("data/synthetic/multiskill_v1/failures.jsonl.gz"),
    )
    parser.add_argument(
        "--review-queue",
        type=Path,
        default=Path("data/synthetic/multiskill_v1/review_queue.jsonl.gz"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/synthetic/multiskill_v1/manifest.json"),
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--min-query-words", type=int, default=30)
    parser.add_argument("--max-query-words", type=int, default=260)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    progress_group = parser.add_mutually_exclusive_group()
    progress_group.add_argument("--progress", dest="progress", action="store_true")
    progress_group.add_argument("--no-progress", dest="progress", action="store_false")
    parser.set_defaults(progress=None)
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
    load_dotenv(ROOT / ".env")
    generator = client or DeepSeekJsonClient(
        model=args.model,
        concurrency=args.concurrency,
        max_tokens=args.max_new_tokens,
        timeout=args.timeout,
    )
    progress_bar, progress_callback = _create_progress_callback(
        len(candidates), getattr(args, "progress", None)
    )
    try:
        result = generate_compositional_queries(
            candidates,
            stream_jsonl(args.contracts),
            generator,
            CompositionalGenerationConfig(
                model=args.model,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
                max_attempts=args.max_attempts,
                concurrency=args.concurrency,
                min_query_words=args.min_query_words,
                max_query_words=args.max_query_words,
            ),
            progress_callback=progress_callback,
        )
    finally:
        if progress_bar is not None:
            progress_bar.close()
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


def _create_progress_callback(
    total: int, requested: bool | None
) -> tuple[Any | None, Callable[[CompositionalGenerationProgress], None] | None]:
    if requested is False or (requested is None and not sys.stderr.isatty()):
        return None, None
    from tqdm import tqdm

    progress_bar = tqdm(total=total, desc="Generating queries", unit="candidate")

    def update(progress: CompositionalGenerationProgress) -> None:
        progress_bar.update(1)
        progress_bar.set_postfix(
            success=progress.queries,
            failures=progress.failures,
            review=progress.review_queue,
        )

    return progress_bar, update


def _validate_inputs(args: argparse.Namespace) -> None:
    for path in (args.candidates, args.contracts):
        if not path.is_file():
            raise FileNotFoundError(f"input file does not exist: {path}")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("limit must be positive")
    if args.concurrency <= 0:
        raise ValueError("concurrency must be positive")
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
        "provider": "deepseek",
        "model": args.model,
        "thinking": "disabled",
        "concurrency": args.concurrency,
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

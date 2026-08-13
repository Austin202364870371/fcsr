"""Local and server preprocessing entry point for FCSR."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from tqdm import tqdm
from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data_io import stream_jsonl, write_jsonl_atomic
from modeling import encode_texts, format_query, format_skill, load_embedding_model
from retrieval import (
    embedding_false_negative_filter,
    merge_negative_sources,
    semantic_topk,
)
from preprocessing import (
    LLMConfig,
    PipelineSummary,
    build_sampling_manifest,
    collect_benchmark_skill_ids,
    extract_contracts,
    generate_queries,
    mine_local_negatives,
    sha256_file,
    stratified_sample,
)
from deepseek_client import DEFAULT_MAX_TOKENS, DEFAULT_MODEL, DeepSeekJsonClient


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="FCSR preprocessing pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    sample = subparsers.add_parser("sample", help="create a deterministic skill sample")
    sample.add_argument("--skills", default="data/raw/skills_easy.jsonl.gz")
    sample.add_argument("--tasks", default="data/raw/evaluation_queries.jsonl.gz")
    sample.add_argument("--sample-size", type=int, default=32000)
    sample.add_argument("--seed", type=int, default=42)
    sample.add_argument("--output-dir", default="data/contracts_32k")
    sample.add_argument("--overwrite", action="store_true")

    contracts = subparsers.add_parser(
        "contracts", help="extract evidence-grounded contracts with DeepSeek"
    )
    _add_llm_arguments(contracts)
    contracts.add_argument("--sample", default="data/contracts_32k/sample_skills.jsonl.gz")
    contracts.add_argument("--output", default="data/contracts_32k_prompt007/contracts.jsonl.gz")
    contracts.add_argument("--failures", default="data/contracts_32k_prompt007/failures.jsonl.gz")
    contracts.add_argument("--no-progress", action="store_true")

    queries = subparsers.add_parser(
        "queries", help="generate contract-grounded queries with DeepSeek"
    )
    _add_llm_arguments(queries)
    queries.add_argument("--sample", default="data/contracts_32k/sample_skills.jsonl.gz")
    queries.add_argument("--contracts", default="data/contracts_32k_prompt007/contracts.jsonl.gz")
    queries.add_argument("--output", default="data/synthetic/single_skill_v1/queries.jsonl.gz")
    queries.add_argument(
        "--failures",
        default="data/synthetic/single_skill_v1/query_failures.jsonl.gz",
    )
    queries.add_argument("--no-progress", action="store_true")

    local = subparsers.add_parser(
        "local-negatives", help="mine BM25, same-category, and random negatives"
    )
    local.add_argument("--queries", default="data/synthetic/single_skill_v1/queries.jsonl.gz")
    local.add_argument("--skills", default="data/raw/skills_easy.jsonl.gz")
    local.add_argument("--output", default="data/synthetic/single_skill_v1/local_negatives.jsonl.gz")
    local.add_argument("--seed", type=int, default=42)
    local.add_argument("--overlap-threshold", type=float, default=0.85)
    local.add_argument("--overwrite", action="store_true")
    local.add_argument("--no-progress", action="store_true")

    semantic = subparsers.add_parser(
        "semantic-negatives", help="mine semantic negatives with a Qwen encoder"
    )
    semantic.add_argument("--local", default="data/synthetic/single_skill_v1/local_negatives.jsonl.gz")
    semantic.add_argument("--skills", default="data/raw/skills_easy.jsonl.gz")
    semantic.add_argument("--output", default="data/synthetic/single_skill_v1/train_biencoder.jsonl.gz")
    semantic.add_argument(
        "--review", default="data/processed/semantic_negative_review.jsonl.gz"
    )
    semantic.add_argument("--model", default="Qwen/Qwen3-Embedding-0.6B")
    semantic.add_argument("--top-k", type=int, default=50)
    semantic.add_argument("--threshold", type=float, default=0.95)
    semantic.add_argument("--batch-size", type=int, default=8)
    semantic.add_argument("--query-max-length", type=int, default=512)
    semantic.add_argument("--skill-max-length", type=int, default=2048)
    semantic.add_argument("--device", default="cuda")
    semantic.add_argument("--overwrite", action="store_true")
    return parser


def _add_llm_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--backoff", type=float, default=2.0)
    parser.add_argument("--limit", type=int)


def run_sample(args: argparse.Namespace) -> dict:
    output_dir = Path(args.output_dir)
    sample_path = output_dir / "sample_skills.jsonl.gz"
    manifest_path = output_dir / "manifest.json"
    if not args.overwrite and (sample_path.exists() or manifest_path.exists()):
        raise FileExistsError("sample outputs exist; pass --overwrite to replace them")

    excluded_ids = collect_benchmark_skill_ids(args.tasks)
    result = stratified_sample(
        stream_jsonl(args.skills),
        excluded_ids,
        args.sample_size,
        args.seed,
    )
    write_jsonl_atomic(sample_path, result.records)
    manifest = build_sampling_manifest(
        result,
        args.sample_size,
        args.seed,
        sha256_file(args.skills),
        excluded_ids,
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {
        "sample_path": str(sample_path),
        "manifest_path": str(manifest_path),
        "selected": len(result.records),
        "excluded": result.excluded_count,
        "duplicates": result.duplicate_count,
    }


def build_llm_client(args: argparse.Namespace) -> DeepSeekJsonClient:
    load_dotenv(ROOT / ".env")
    return DeepSeekJsonClient(
        model=args.model,
        concurrency=args.concurrency,
        max_tokens=args.max_new_tokens,
        timeout=args.timeout,
    )


def _llm_config(args: argparse.Namespace) -> LLMConfig:
    return LLMConfig(
        model=args.model,
        provider="deepseek",
        temperature=args.temperature,
        max_attempts=args.max_attempts,
        backoff_seconds=args.backoff,
        batch_size=args.concurrency,
        limit=args.limit,
    )


def run_contracts(args: argparse.Namespace) -> dict:
    config = _llm_config(args)
    total = sum(1 for _ in stream_jsonl(args.sample))
    if config.limit is not None:
        total = min(total, config.limit)

    previous = 0
    with tqdm(
        total=total,
        desc="Contracts",
        unit="skill",
        dynamic_ncols=True,
        disable=args.no_progress,
    ) as progress_bar:

        def update_progress(current: PipelineSummary) -> None:
            nonlocal previous
            processed = current.succeeded + current.skipped + current.failed
            progress_bar.update(processed - previous)
            progress_bar.set_postfix(
                ok=current.succeeded,
                skip=current.skipped,
                fail=current.failed,
                refresh=True,
            )
            previous = processed

        summary = extract_contracts(
            args.sample,
            args.output,
            args.failures,
            build_llm_client(args),
            config,
            progress=update_progress,
        )
    return asdict(summary)


def run_queries(args: argparse.Namespace) -> dict:
    config = _llm_config(args)
    total = sum(1 for _ in stream_jsonl(args.sample))
    if config.limit is not None:
        total = min(total, config.limit)

    previous = 0
    with tqdm(
        total=total,
        desc="Queries",
        unit="skill",
        dynamic_ncols=True,
        disable=args.no_progress,
    ) as progress_bar:

        def update_progress(current: PipelineSummary) -> None:
            nonlocal previous
            processed = current.succeeded + current.skipped + current.failed
            progress_bar.update(processed - previous)
            progress_bar.set_postfix(
                ok=current.succeeded,
                skip=current.skipped,
                fail=current.failed,
                refresh=True,
            )
            previous = processed

        summary = generate_queries(
            args.sample,
            args.contracts,
            args.output,
            args.failures,
            build_llm_client(args),
            config,
            progress=update_progress,
        )
    return asdict(summary)


def run_local_negatives(args: argparse.Namespace) -> dict:
    output = Path(args.output)
    if output.exists() and not args.overwrite:
        raise FileExistsError("output exists; pass --overwrite to replace it")
    total = sum(1 for _ in stream_jsonl(args.queries))
    stage_labels = {
        "loading_skills": "Local negatives: loading skills",
        "building_bm25": "Local negatives: building BM25",
        "mining_queries": "Local negatives: mining",
    }
    with tqdm(
        total=total,
        desc=stage_labels["loading_skills"],
        unit="query",
        dynamic_ncols=True,
        disable=args.no_progress,
    ) as progress_bar:

        def update_stage(current: str) -> None:
            if current == "mining_queries":
                progress_bar.reset(total=total)
            progress_bar.set_description(stage_labels[current], refresh=True)

        def update_progress(processed: int) -> None:
            progress_bar.update(processed - progress_bar.n)

        count = write_jsonl_atomic(
            output,
            mine_local_negatives(
                stream_jsonl(args.queries),
                stream_jsonl(args.skills),
                seed=args.seed,
                overlap_threshold=args.overlap_threshold,
                stage=update_stage,
                progress=update_progress,
            ),
        )
    return {"output": str(output), "records": count}


def run_semantic_negatives(args: argparse.Namespace) -> dict:
    output = Path(args.output)
    review = Path(args.review)
    if not args.overwrite and (output.exists() or review.exists()):
        raise FileExistsError("semantic outputs exist; pass --overwrite to replace them")

    skills = list(stream_jsonl(args.skills))
    local_records = list(stream_jsonl(args.local))
    skill_ids = [skill["skill_id"] for skill in skills]
    index_by_id = {skill_id: index for index, skill_id in enumerate(skill_ids)}
    model, tokenizer = load_embedding_model(args.model, args.device)
    skill_embeddings = encode_texts(
        model,
        tokenizer,
        [format_skill(skill) for skill in skills],
        args.skill_max_length,
        args.batch_size,
        args.device,
    )
    query_embeddings = encode_texts(
        model,
        tokenizer,
        [format_query(str(record.get("query", ""))) for record in local_records],
        args.query_max_length,
        args.batch_size,
        args.device,
    )
    top_indices, top_scores = semantic_topk(
        query_embeddings,
        skill_embeddings,
        args.top_k,
        device=args.device,
    )

    final_records = []
    review_records = []
    for row, local_record in enumerate(local_records):
        positive_id = local_record["positive_skill_id"]
        positive_index = index_by_id.get(positive_id)
        if positive_index is None:
            raise ValueError(f"positive skill missing from pool: {positive_id!r}")
        semantic = [
            {"skill_id": skill_ids[index], "score": float(score)}
            for index, score in zip(top_indices[row], top_scores[row])
            if skill_ids[index] != positive_id
        ]
        merged = merge_negative_sources(local_record, semantic)
        candidate_embeddings = {
            item["skill_id"]: skill_embeddings[index_by_id[item["skill_id"]]]
            for item in merged["negative_candidates"]
        }
        filtered = embedding_false_negative_filter(
            skill_embeddings[positive_index],
            merged["negative_candidates"],
            candidate_embeddings,
            args.threshold,
        )
        merged["negative_candidates"] = filtered.kept
        merged["filtered"] = list(merged.get("filtered", [])) + filtered.removed
        final_records.append(merged)
        review_records.extend(
            {
                "query_id": merged.get("query_id"),
                "positive_skill_id": positive_id,
                **item,
            }
            for item in filtered.removed
        )

    output_count = write_jsonl_atomic(output, final_records)
    review_count = write_jsonl_atomic(review, review_records)
    return {
        "output": str(output),
        "records": output_count,
        "review": str(review),
        "review_records": review_count,
    }


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "sample":
        summary = run_sample(args)
    elif args.command == "contracts":
        summary = run_contracts(args)
    elif args.command == "queries":
        summary = run_queries(args)
    elif args.command == "local-negatives":
        summary = run_local_negatives(args)
    elif args.command == "semantic-negatives":
        summary = run_semantic_negatives(args)
    else:
        parser.error(f"unsupported command: {args.command}")
        return
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

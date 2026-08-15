"""Export retrieval/reranker predictions and score them with the SR protocol."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data_io import load_jsonl, stream_jsonl, write_jsonl_atomic
from evaluation import evaluate_predictions
from modeling import (
    encode_texts,
    format_query,
    format_rerank_prompt,
    format_skill,
    get_reranker_template_tokens,
    load_embedding_model,
    tokenize_reranker_text,
)
from retrieval import BM25Index, reciprocal_rank_fusion, semantic_topk


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="FCSR inference and SR evaluation")
    commands = parser.add_subparsers(dest="command", required=True)

    retrieve = commands.add_parser("retrieve", help="export bi-encoder Top-K")
    retrieve.add_argument("--queries", required=True)
    retrieve.add_argument("--skills", required=True)
    retrieve.add_argument("--model", required=True)
    retrieve.add_argument("--output-predictions", required=True)
    retrieve.add_argument("--output-records", required=True)
    retrieve.add_argument("--top-k", type=int, default=50)
    retrieve.add_argument("--batch-size", type=int, default=8)
    retrieve.add_argument("--query-max-length", type=int, default=512)
    retrieve.add_argument("--skill-max-length", type=int, default=4096)
    retrieve.add_argument("--device", default="cuda")

    bm25 = commands.add_parser("bm25", help="export flat BM25 Top-K")
    bm25.add_argument("--queries", required=True)
    bm25.add_argument("--skills", required=True)
    bm25.add_argument("--output-predictions", required=True)
    bm25.add_argument("--output-records", required=True)
    bm25.add_argument("--top-k", type=int, default=50)

    hybrid = commands.add_parser("hybrid", help="export BM25+dense RRF Top-K")
    hybrid.add_argument("--queries", required=True)
    hybrid.add_argument("--skills", required=True)
    hybrid.add_argument("--model", required=True)
    hybrid.add_argument("--output-predictions", required=True)
    hybrid.add_argument("--output-records", required=True)
    hybrid.add_argument("--top-k", type=int, default=50)
    hybrid.add_argument("--fusion-depth", type=int, default=100)
    hybrid.add_argument("--rrf-k", type=int, default=60)
    hybrid.add_argument("--batch-size", type=int, default=8)
    hybrid.add_argument("--query-max-length", type=int, default=512)
    hybrid.add_argument("--skill-max-length", type=int, default=2048)
    hybrid.add_argument("--device", default="cuda")
    rerank = commands.add_parser("rerank", help="rerank exported candidates")
    rerank.add_argument("--retrieval-records", required=True)
    rerank.add_argument("--skills", required=True)
    rerank.add_argument("--model", required=True)
    rerank.add_argument("--output-predictions", required=True)
    rerank.add_argument("--output-records", required=True)
    rerank.add_argument("--top-k", type=int, default=20)
    rerank.add_argument("--batch-size", type=int, default=4)
    rerank.add_argument("--max-length", type=int, default=4096)
    rerank.add_argument("--device", default="cuda")

    score = commands.add_parser("score", help="score a public-format prediction map")
    score.add_argument("--tasks", required=True)
    score.add_argument("--skills", required=True)
    score.add_argument("--predictions", required=True)
    score.add_argument("--stage", choices=("retrieval", "reranker"), required=True)
    score.add_argument("--output-dir", default="reports")
    return parser


def format_retrieval_skill(skill: dict[str, Any]) -> str:
    """Keep retrieval skill text identical to bi-encoder training text."""
    return format_skill(skill)

def run_retrieve(args: argparse.Namespace) -> dict[str, Any]:
    queries = load_jsonl(args.queries)
    skills = list(stream_jsonl(args.skills))
    model, tokenizer = load_embedding_model(args.model, args.device)
    with tqdm(
        total=len(skills),
        desc="Retrieve: encoding skills",
        unit="skill",
        dynamic_ncols=True,
    ) as progress:
        skill_embeddings = encode_texts(
            model,
            tokenizer,
            [format_retrieval_skill(skill) for skill in skills],
            args.skill_max_length,
            args.batch_size,
            args.device,
            progress=progress.update,
        )
    with tqdm(
        total=len(queries),
        desc="Retrieve: encoding queries",
        unit="query",
        dynamic_ncols=True,
    ) as progress:
        query_embeddings = encode_texts(
            model,
            tokenizer,
            [format_query(_query_text(query)) for query in queries],
            args.query_max_length,
            args.batch_size,
            args.device,
            progress=progress.update,
        )
    with tqdm(
        total=len(queries),
        desc="Retrieve: scoring queries",
        unit="query",
        dynamic_ncols=True,
    ) as progress:
        indices, scores = semantic_topk(
            query_embeddings,
            skill_embeddings,
            args.top_k,
            device=args.device,
            progress=progress.update,
        )
    records = []
    predictions = {}
    for row, query in enumerate(queries):
        query_id = _query_id(query)
        candidates = [
            {
                "skill_id": skills[index]["skill_id"],
                "score": float(score),
                "rank": rank,
            }
            for rank, (index, score) in enumerate(
                zip(indices[row], scores[row]),
                start=1,
            )
        ]
        predictions[query_id] = [item["skill_id"] for item in candidates]
        records.append(
            {
                "query_id": query_id,
                "query": _query_text(query),
                "positive_skill_id": query.get("positive_skill_id"),
                "positive_skill_ids": sorted(_positive_ids(query)),
                "retrieved_candidates": candidates,
            }
        )
    _write_json(args.output_predictions, predictions)
    write_jsonl_atomic(args.output_records, records)
    return {
        "queries": len(queries),
        "pool_size": len(skills),
        "top_k": min(args.top_k, len(skills)),
        "predictions": args.output_predictions,
        "records": args.output_records,
    }


def run_bm25(args: argparse.Namespace) -> dict[str, Any]:
    queries = load_jsonl(args.queries)
    skills = list(stream_jsonl(args.skills))
    with tqdm(
        total=len(skills),
        desc="BM25: indexing skills",
        unit="skill",
        dynamic_ncols=True,
    ) as progress:
        index = BM25Index(
            [format_retrieval_skill(skill) for skill in skills],
            progress=progress.update,
        )
    candidate_lists = []
    with tqdm(
        total=len(queries),
        desc="BM25: scoring queries",
        unit="query",
        dynamic_ncols=True,
    ) as progress:
        for query in queries:
            scores, indices = index.topk(_query_text(query), args.top_k)
            candidate_lists.append(
                [
                    {
                        "skill_id": skills[index]["skill_id"],
                        "score": float(scores[index]),
                        "rank": rank,
                    }
                    for rank, index in enumerate(indices, start=1)
                ]
            )
            progress.update(1)
    return _write_retrieval_outputs(args, queries, skills, candidate_lists)


def run_hybrid(args: argparse.Namespace) -> dict[str, Any]:
    if args.fusion_depth < args.top_k:
        raise ValueError("fusion_depth must be greater than or equal to top_k")
    queries = load_jsonl(args.queries)
    skills = list(stream_jsonl(args.skills))
    model, tokenizer = load_embedding_model(args.model, args.device)
    with tqdm(
        total=len(skills),
        desc="Hybrid: encoding skills",
        unit="skill",
        dynamic_ncols=True,
    ) as progress:
        skill_embeddings = encode_texts(
            model,
            tokenizer,
            [format_retrieval_skill(skill) for skill in skills],
            args.skill_max_length,
            args.batch_size,
            args.device,
            progress=progress.update,
        )
    with tqdm(
        total=len(queries),
        desc="Hybrid: encoding queries",
        unit="query",
        dynamic_ncols=True,
    ) as progress:
        query_embeddings = encode_texts(
            model,
            tokenizer,
            [format_query(_query_text(query)) for query in queries],
            args.query_max_length,
            args.batch_size,
            args.device,
            progress=progress.update,
        )
    with tqdm(
        total=len(skills),
        desc="Hybrid: indexing BM25",
        unit="skill",
        dynamic_ncols=True,
    ) as progress:
        index = BM25Index(
            [format_retrieval_skill(skill) for skill in skills],
            progress=progress.update,
        )
    depth = min(args.fusion_depth, len(skills))
    with tqdm(
        total=len(queries),
        desc="Hybrid: dense scoring queries",
        unit="query",
        dynamic_ncols=True,
    ) as progress:
        dense_indices, _ = semantic_topk(
            query_embeddings,
            skill_embeddings,
            depth,
            device=args.device,
            progress=progress.update,
        )
    candidate_lists = []
    with tqdm(
        total=len(queries),
        desc="Hybrid: fusing rankings",
        unit="query",
        dynamic_ncols=True,
    ) as progress:
        for row, query in enumerate(queries):
            _, lexical_indices = index.topk(_query_text(query), depth)
            dense_ranking = [skills[index]["skill_id"] for index in dense_indices[row]]
            lexical_ranking = [skills[index]["skill_id"] for index in lexical_indices]
            fused = reciprocal_rank_fusion(
                [lexical_ranking, dense_ranking],
                top_k=args.top_k,
                rrf_k=args.rrf_k,
            )
            candidate_lists.append(
                [
                    {
                        "skill_id": item["skill_id"],
                        "score": float(item["rrf_score"]),
                        "rrf_score": float(item["rrf_score"]),
                        "rank": rank,
                    }
                    for rank, item in enumerate(fused, start=1)
                ]
            )
            progress.update(1)
    return _write_retrieval_outputs(args, queries, skills, candidate_lists)


def _write_retrieval_outputs(
    args: argparse.Namespace,
    queries: list[dict[str, Any]],
    skills: list[dict[str, Any]],
    candidate_lists: list[list[dict[str, Any]]],
) -> dict[str, Any]:
    if len(queries) != len(candidate_lists):
        raise ValueError("candidate_lists must contain one list per query")
    records = []
    predictions = {}
    for query, candidates in zip(queries, candidate_lists):
        query_id = _query_id(query)
        predictions[query_id] = [item["skill_id"] for item in candidates]
        records.append(
            {
                "query_id": query_id,
                "query": _query_text(query),
                "positive_skill_id": query.get("positive_skill_id"),
                "positive_skill_ids": sorted(_positive_ids(query)),
                "retrieved_candidates": candidates,
            }
        )
    _write_json(args.output_predictions, predictions)
    write_jsonl_atomic(args.output_records, records)
    return {
        "queries": len(queries),
        "pool_size": len(skills),
        "top_k": min(args.top_k, len(skills)),
        "predictions": args.output_predictions,
        "records": args.output_records,
    }

def run_rerank(args: argparse.Namespace) -> dict[str, Any]:
    records = load_jsonl(args.retrieval_records)
    skills = {
        skill["skill_id"]: skill
        for skill in stream_jsonl(args.skills)
        if isinstance(skill.get("skill_id"), str)
    }
    model, tokenizer, yes_id, no_id = _load_reranker(args.model, args.device)
    reranked_records = []
    predictions = {}
    with create_rerank_progress(len(records)) as progress:
        for record in records:
            candidates = record.get("retrieved_candidates", [])[: args.top_k]
            prompts = [
                format_rerank_prompt(_query_text(record), skills[item["skill_id"]])
                for item in candidates
            ]
            rerank_scores = _score_prompts(
                model,
                tokenizer,
                prompts,
                yes_id,
                no_id,
                args.max_length,
                args.batch_size,
                args.device,
            )
            reranked = sorted(
                (
                    {**candidate, "reranker_score": float(score)}
                    for candidate, score in zip(candidates, rerank_scores)
                ),
                key=lambda item: (-item["reranker_score"], item["skill_id"]),
            )
            for rank, candidate in enumerate(reranked, start=1):
                candidate["reranker_rank"] = rank
            query_id = _query_id(record)
            predictions[query_id] = [item["skill_id"] for item in reranked]
            reranked_records.append({**record, "reranked_candidates": reranked})
            progress.update(1)
    _write_json(args.output_predictions, predictions)
    write_jsonl_atomic(args.output_records, reranked_records)
    return {
        "queries": len(records),
        "rerank_top_k": args.top_k,
        "predictions": args.output_predictions,
        "records": args.output_records,
    }


def create_rerank_progress(
    total_queries: int,
    progress_factory: Any | None = None,
) -> Any:
    if progress_factory is None:
        progress_factory = tqdm
    return progress_factory(
        total=total_queries,
        desc="Rerank: scoring queries",
        unit="query",
        dynamic_ncols=True,
    )

def run_score(args: argparse.Namespace) -> dict[str, Any]:
    tasks = load_jsonl(args.tasks)
    pool_ids = {skill["skill_id"] for skill in stream_jsonl(args.skills)}
    predictions = json.loads(Path(args.predictions).read_text(encoding="utf-8"))
    if not isinstance(predictions, dict):
        raise ValueError("predictions must be a JSON object keyed by task id")
    result = evaluate_predictions(tasks, predictions, pool_ids)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "stage": args.stage,
        "benchmark": "hard",
        "metrics": result.summary,
        "skipped_generic_only": result.skipped_generic_only,
        "skipped_missing_prediction": result.skipped_missing_prediction,
        "skipped_no_gt_in_pool": result.skipped_no_gt_in_pool,
    }
    _write_json(output_dir / "summary.json", summary)
    write_jsonl_atomic(output_dir / "details.jsonl", result.details)
    return summary


def _load_reranker(name_or_path: str, device: str) -> tuple[Any, Any, int, int]:
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("install requirements.txt for reranking") from exc
    model_path = str(name_or_path)
    adapter = Path(model_path) / "adapter_config.json"
    base_path = model_path
    peft_model = None
    if adapter.exists():
        from peft import PeftConfig, PeftModel

        base_path = PeftConfig.from_pretrained(model_path).base_model_name_or_path
        peft_model = PeftModel
    tokenizer_path = (
        model_path if (Path(model_path) / "tokenizer_config.json").exists() else base_path
    )
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        padding_side="left",
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_path,
        torch_dtype="auto",
        trust_remote_code=True,
    )
    if peft_model is not None:
        model = peft_model.from_pretrained(model, model_path)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    model.to(device)
    model.eval()
    return (
        model,
        tokenizer,
        tokenizer.convert_tokens_to_ids("yes"),
        tokenizer.convert_tokens_to_ids("no"),
    )


def _score_prompts(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    yes_id: int,
    no_id: int,
    max_length: int,
    batch_size: int,
    device: str,
) -> list[float]:
    import torch

    prefix_tokens, suffix_tokens = get_reranker_template_tokens(tokenizer)
    tokenized = [
        tokenize_reranker_text(
            prompt,
            tokenizer,
            prefix_tokens,
            suffix_tokens,
            max_length,
        )
        for prompt in prompts
    ]
    scores = []
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    with torch.no_grad():
        for start in range(0, len(tokenized), batch_size):
            batch_ids = tokenized[start : start + batch_size]
            batch_max = max(len(ids) for ids in batch_ids)
            padded = [[pad_id] * (batch_max - len(ids)) + ids for ids in batch_ids]
            masks = [[0] * (batch_max - len(ids)) + [1] * len(ids) for ids in batch_ids]
            input_ids = torch.tensor(padded, dtype=torch.long, device=device)
            attention_mask = torch.tensor(masks, dtype=torch.long, device=device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[:, -1, :]
            batch_scores = logits[:, yes_id] - logits[:, no_id]
            scores.extend(batch_scores.float().cpu().tolist())
    return scores


def _query_id(record: dict[str, Any]) -> str:
    value = record.get("task_id") or record.get("query_id")
    if not isinstance(value, str) or not value:
        raise ValueError("query record lacks task_id/query_id")
    return value


def _query_text(record: dict[str, Any]) -> str:
    value = record.get("query") or record.get("task")
    if not isinstance(value, str):
        raise ValueError("query record lacks query text")
    return value


def _positive_ids(record: dict[str, Any]) -> set[str]:
    result = set()
    for field in (
        "positive_skill_ids",
        "core_gt_ids",
        "core_gold_skill_ids",
        "gt_skill_ids",
        "gold_skill_ids",
    ):
        values = record.get(field)
        if isinstance(values, list):
            result.update(value for value in values if isinstance(value, str))
    value = record.get("positive_skill_id")
    if isinstance(value, str):
        result.add(value)
    return result


def _write_json(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "retrieve":
        result = run_retrieve(args)
    elif args.command == "bm25":
        result = run_bm25(args)
    elif args.command == "hybrid":
        result = run_hybrid(args)
    elif args.command == "rerank":
        result = run_rerank(args)
    else:
        result = run_score(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

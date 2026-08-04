"""Build deterministic RQ1 mixed single- and multi-Skill training data."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path
from typing import Any

from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data_io import stream_jsonl, write_jsonl_atomic
from modeling import encode_texts, format_query, format_skill, load_embedding_model
from retrieval import embedding_false_negative_filter, semantic_topk
from rq1_training_data import build_mixed_training_records


DEFAULT_OUTPUT_DIR = Path("data/training/rq1-mixed-3x")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="build RQ1 single- plus multi-Skill training data with safe negatives"
    )
    parser.add_argument(
        "--single-biencoder",
        type=Path,
        default=Path("data/synthetic/single_v1/train_biencoder.jsonl.gz"),
    )
    parser.add_argument(
        "--single-reranker",
        type=Path,
        default=Path("data/synthetic/single_v1/train_reranker.jsonl.gz"),
    )
    parser.add_argument(
        "--compositional",
        type=Path,
        default=Path("data/synthetic/compositional_v1/compositional_queries.jsonl.gz"),
    )
    parser.add_argument(
        "--skills",
        type=Path,
        default=Path("data/raw/skills_easy.jsonl.gz"),
    )
    parser.add_argument(
        "--negative-model",
        default="models/Qwen3-Embedding-0.6B",
        help="local base embedding model used only to mine semantic negatives",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--multiplier", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--semantic-top-k", type=int, default=64)
    parser.add_argument("--semantic-fn-threshold", type=float, default=0.95)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--query-max-length", type=int, default=512)
    parser.add_argument("--skill-max-length", type=int, default=2048)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    _validate_inputs(args)
    outputs = _output_paths(args.output_dir)
    _validate_outputs(outputs, args.overwrite)

    single_biencoder = list(stream_jsonl(args.single_biencoder))
    single_reranker = list(stream_jsonl(args.single_reranker))
    compositional = list(stream_jsonl(args.compositional))
    skills = list(stream_jsonl(args.skills))
    _validate_unique_query_ids(compositional)
    semantic_candidates = mine_semantic_candidates(
        compositional,
        skills,
        model_path=args.negative_model,
        semantic_top_k=args.semantic_top_k,
        semantic_fn_threshold=args.semantic_fn_threshold,
        batch_size=args.batch_size,
        query_max_length=args.query_max_length,
        skill_max_length=args.skill_max_length,
        device=args.device,
        show_progress=not args.no_progress,
    )
    with _progress(
        len(compositional), "Building multi-Skill records", "query", args.no_progress
    ) as progress:
        result = build_mixed_training_records(
            single_biencoder,
            single_reranker,
            compositional,
            skills,
            semantic_candidates,
            multiplier=args.multiplier,
            seed=args.seed,
            progress=progress.update,
        )
    biencoder_count = write_jsonl_atomic(outputs["biencoder"], result.biencoder_records)
    reranker_count = write_jsonl_atomic(outputs["reranker"], result.reranker_groups)
    counts = {
        "single_biencoder": len(single_biencoder),
        "single_reranker": len(single_reranker),
        "compositional_queries": result.compositional_query_count,
        "compositional_biencoder_examples": result.compositional_biencoder_examples,
        "compositional_reranker_groups": result.compositional_reranker_groups,
        "biencoder_records": biencoder_count,
        "reranker_groups": reranker_count,
    }
    manifest = build_manifest(
        single_biencoder_path=args.single_biencoder,
        single_reranker_path=args.single_reranker,
        compositional_path=args.compositional,
        skills_path=args.skills,
        negative_model=args.negative_model,
        semantic_top_k=args.semantic_top_k,
        semantic_fn_threshold=args.semantic_fn_threshold,
        multiplier=args.multiplier,
        seed=args.seed,
        counts=counts,
    )
    outputs["manifest"].write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        **counts,
        "biencoder_output": outputs["biencoder"].as_posix(),
        "reranker_output": outputs["reranker"].as_posix(),
        "manifest_output": outputs["manifest"].as_posix(),
    }


def mine_semantic_candidates(
    compositional: list[dict[str, Any]],
    skills: list[dict[str, Any]],
    *,
    model_path: str,
    semantic_top_k: int,
    semantic_fn_threshold: float,
    batch_size: int,
    query_max_length: int,
    skill_max_length: int,
    device: str,
    show_progress: bool,
) -> dict[str, list[dict[str, Any]]]:
    if semantic_top_k <= 0:
        raise ValueError("semantic_top_k must be positive")
    if not -1 <= semantic_fn_threshold <= 1:
        raise ValueError("semantic_fn_threshold must be between -1 and 1")
    if not compositional:
        return {}
    if not Path(model_path).is_dir():
        raise FileNotFoundError(f"local negative model directory does not exist: {model_path}")
    skill_ids = [record.get("skill_id") for record in skills]
    index_by_id = {skill_id: index for index, skill_id in enumerate(skill_ids)}
    if not all(isinstance(skill_id, str) and skill_id for skill_id in skill_ids):
        raise ValueError("skills must contain non-empty skill_id values")
    model, tokenizer = load_embedding_model(model_path, device=device)
    try:
        with _progress(len(skills), "Encoding Skills for negatives", "skill", not show_progress) as progress:
            skill_embeddings = encode_texts(
                model,
                tokenizer,
                [format_skill(record) for record in skills],
                max_length=skill_max_length,
                batch_size=batch_size,
                device=device,
                progress=progress.update,
            )
        with _progress(len(compositional), "Encoding multi-Skill queries", "query", not show_progress) as progress:
            query_embeddings = encode_texts(
                model,
                tokenizer,
                [format_query(str(record.get("query", ""))) for record in compositional],
                max_length=query_max_length,
                batch_size=batch_size,
                device=device,
                progress=progress.update,
            )
        with _progress(len(compositional), "Retrieving semantic negatives", "query", not show_progress) as progress:
            indices, scores = semantic_topk(
                query_embeddings,
                skill_embeddings,
                k=semantic_top_k,
                device=device,
                progress=progress.update,
            )
    finally:
        del model, tokenizer
        gc.collect()
        _empty_cuda_cache()

    candidates: dict[str, list[dict[str, Any]]] = {}
    for record, row_indices, row_scores in zip(compositional, indices, scores):
        query_id = record["query_id"]
        raw_candidates = [
            {"skill_id": skill_ids[index], "score": float(score)}
            for index, score in zip(row_indices.tolist(), row_scores.tolist())
        ]
        positive_ids = record.get("positive_skill_ids")
        if not isinstance(positive_ids, list):
            raise ValueError("compositional query must contain positive_skill_ids")
        candidates[query_id] = filter_semantic_false_negatives(
            raw_candidates,
            positive_ids,
            skill_embeddings,
            index_by_id,
            threshold=semantic_fn_threshold,
        )
    return candidates


def filter_semantic_false_negatives(
    candidates: list[dict[str, Any]],
    positive_skill_ids: list[str],
    skill_embeddings: Any,
    index_by_id: dict[str, int],
    *,
    threshold: float,
) -> list[dict[str, Any]]:
    """Drop candidates highly similar to any true Skill in a composition."""
    if not -1 <= threshold <= 1:
        raise ValueError("threshold must be between -1 and 1")
    kept = list(candidates)
    candidate_embeddings = {
        candidate["skill_id"]: skill_embeddings[index_by_id[candidate["skill_id"]]]
        for candidate in candidates
        if candidate.get("skill_id") in index_by_id
    }
    for positive_id in positive_skill_ids:
        if not isinstance(positive_id, str) or positive_id not in index_by_id:
            raise ValueError(f"positive skill not found for semantic filtering: {positive_id!r}")
        filtered = embedding_false_negative_filter(
            skill_embeddings[index_by_id[positive_id]],
            kept,
            candidate_embeddings,
            threshold,
        )
        kept = filtered.kept
    return kept

def build_manifest(
    *,
    single_biencoder_path: Path,
    single_reranker_path: Path,
    compositional_path: Path,
    skills_path: Path,
    negative_model: str,
    semantic_top_k: int,
    multiplier: int,
    seed: int,
    counts: dict[str, int],
    semantic_fn_threshold: float = 0.95,
) -> dict[str, Any]:
    return {
        "schema_version": "rq1_mixed_training_v1",
        "task": "rq1_multi_skill_retrieval",
        "storage_format": "jsonl.gz",
        "inputs": {
            "single_biencoder": single_biencoder_path.as_posix(),
            "single_reranker": single_reranker_path.as_posix(),
            "compositional_queries": compositional_path.as_posix(),
            "skills": skills_path.as_posix(),
        },
        "negative_mining": {
            "semantic_model": negative_model,
            "semantic_top_k": semantic_top_k,
            "semantic_fn_threshold": semantic_fn_threshold,
            "sources": {"semantic": 4, "bm25": 3, "same_category": 2, "random": 1},
            "multi_positive_policy": "exclude_every_positive_skill_id_from_negatives",
        },
        "sampling": {
            "compositional_multiplier": multiplier,
            "seed": seed,
            "unit": "original_compositional_query_group",
        },
        "counts": counts,
        "outputs": {
            "biencoder": {"path": "biencoder.jsonl.gz", "records": counts["biencoder_records"]},
            "reranker": {"path": "reranker.jsonl.gz", "records": counts["reranker_groups"]},
        },
    }


def _output_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "biencoder": output_dir / "biencoder.jsonl.gz",
        "reranker": output_dir / "reranker.jsonl.gz",
        "manifest": output_dir / "manifest.json",
    }


def _validate_inputs(args: argparse.Namespace) -> None:
    for path in (
        args.single_biencoder,
        args.single_reranker,
        args.compositional,
        args.skills,
    ):
        if not path.is_file():
            raise FileNotFoundError(f"input file does not exist: {path}")
    if args.multiplier <= 0:
        raise ValueError("multiplier must be positive")
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive")


def _validate_outputs(outputs: dict[str, Path], overwrite: bool) -> None:
    if overwrite:
        return
    existing = [path for path in outputs.values() if path.exists()]
    if existing:
        raise FileExistsError(
            "training data outputs already exist; pass --overwrite to replace them: "
            + ", ".join(str(path) for path in existing)
        )


def _validate_unique_query_ids(records: list[dict[str, Any]]) -> None:
    query_ids = [record.get("query_id") for record in records]
    if not all(isinstance(query_id, str) and query_id for query_id in query_ids):
        raise ValueError("compositional queries must contain non-empty query_id values")
    if len(set(query_ids)) != len(query_ids):
        raise ValueError("compositional queries contain duplicate query_id values")


def _progress(total: int, description: str, unit: str, disabled: bool):
    return tqdm(total=total, desc=description, unit=unit, dynamic_ncols=True, disable=disabled)


def _empty_cuda_cache() -> None:
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    summary = run(build_parser().parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
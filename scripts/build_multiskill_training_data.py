"""Build deterministic single- and multi-Skill training data."""

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
from multiskill_training_data import build_mixed_training_records


DEFAULT_OUTPUT_DIR = Path("data/training")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="build mixed single- and multi-Skill training data with safe negatives"
    )
    parser.add_argument(
        "--single-biencoder",
        type=Path,
        default=Path("data/synthetic/single_skill/train_biencoder.jsonl.gz"),
    )
    parser.add_argument(
        "--single-reranker",
        type=Path,
        default=Path("data/synthetic/single_skill/train_reranker.jsonl.gz"),
    )
    parser.add_argument(
        "--multiskill-queries",
        dest="multiskill_queries",
        type=Path,
        default=Path("data/synthetic/multi_skill/queries.jsonl.gz"),
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
    parser.add_argument("--biencoder-multi-loss-weight", type=float, default=1.5)
    parser.add_argument("--reranker-multi-loss-weight", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--semantic-top-k", type=int, default=64)
    parser.add_argument("--semantic-fn-threshold", type=float, default=0.95)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--query-max-length", type=int, default=512)
    parser.add_argument("--skill-max-length", type=int, default=2048)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--semantic-review",
        type=Path,
        help=(
            "optional path for semantic candidates removed as likely false negatives; "
            "defaults to <output-dir>/semantic_fn_review.jsonl.gz"
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    _validate_inputs(args)
    outputs = _output_paths(args.output_dir)
    if args.semantic_review is not None:
        outputs["semantic_fn_review"] = args.semantic_review
    _validate_outputs(outputs, args.overwrite)

    single_biencoder = list(stream_jsonl(args.single_biencoder))
    single_reranker = list(stream_jsonl(args.single_reranker))
    multiskill_queries = list(stream_jsonl(args.multiskill_queries))
    skills = list(stream_jsonl(args.skills))
    _validate_unique_query_ids(multiskill_queries)
    semantic_review_records: list[dict[str, Any]] = []
    semantic_candidates = mine_semantic_candidates(
        multiskill_queries,
        skills,
        model_path=args.negative_model,
        semantic_top_k=args.semantic_top_k,
        semantic_fn_threshold=args.semantic_fn_threshold,
        batch_size=args.batch_size,
        query_max_length=args.query_max_length,
        skill_max_length=args.skill_max_length,
        device=args.device,
        show_progress=not args.no_progress,
        semantic_review_records=semantic_review_records,
    )
    with _progress(
        len(multiskill_queries), "Building multi-Skill records", "query", args.no_progress
    ) as progress:
        result = build_mixed_training_records(
            single_biencoder,
            single_reranker,
            multiskill_queries,
            skills,
            semantic_candidates,
            biencoder_multi_loss_weight=args.biencoder_multi_loss_weight,
            reranker_multi_loss_weight=args.reranker_multi_loss_weight,
            seed=args.seed,
            progress=progress.update,
        )
    biencoder_count = write_jsonl_atomic(outputs["biencoder"], result.biencoder_records)
    reranker_count = write_jsonl_atomic(outputs["reranker"], result.reranker_groups)
    semantic_review_count = write_jsonl_atomic(
        outputs["semantic_fn_review"], semantic_review_records
    )
    counts = {
        "single_biencoder": len(single_biencoder),
        "single_reranker": len(single_reranker),
        "multiskill_queries": result.compositional_query_count,
        "multiskill_biencoder_examples": result.compositional_biencoder_examples,
        "multiskill_reranker_groups": result.compositional_reranker_groups,
        "biencoder_records": biencoder_count,
        "reranker_groups": reranker_count,
        "multiskill_semantic_fn_removed": semantic_review_count,
    }
    manifest = build_manifest(
        single_biencoder_path=args.single_biencoder,
        single_reranker_path=args.single_reranker,
        multiskill_queries_path=args.multiskill_queries,
        skills_path=args.skills,
        negative_model=args.negative_model,
        semantic_top_k=args.semantic_top_k,
        semantic_fn_threshold=args.semantic_fn_threshold,
        biencoder_multi_loss_weight=args.biencoder_multi_loss_weight,
        reranker_multi_loss_weight=args.reranker_multi_loss_weight,
        seed=args.seed,
        counts=counts,
        semantic_review_path=outputs["semantic_fn_review"],
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
        "semantic_review_output": outputs["semantic_fn_review"].as_posix(),
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
    semantic_review_records: list[dict[str, Any]] | None = None,
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
            query_id=query_id,
            review_records=semantic_review_records,
        )
    return candidates


def filter_semantic_false_negatives(
    candidates: list[dict[str, Any]],
    positive_skill_ids: list[str],
    skill_embeddings: Any,
    index_by_id: dict[str, int],
    *,
    threshold: float,
    query_id: str | None = None,
    review_records: list[dict[str, Any]] | None = None,
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
        if review_records is not None:
            review_records.extend(
                {
                    "query_id": query_id,
                    "positive_skill_id": positive_id,
                    **item,
                }
                for item in filtered.removed
            )
        kept = filtered.kept
    return kept

def build_manifest(
    *,
    single_biencoder_path: Path,
    single_reranker_path: Path,
    multiskill_queries_path: Path,
    skills_path: Path,
    negative_model: str,
    semantic_top_k: int,
    biencoder_multi_loss_weight: float,
    reranker_multi_loss_weight: float,
    seed: int,
    counts: dict[str, int],
    semantic_fn_threshold: float = 0.95,
    semantic_review_path: Path = Path(
        "data/training/semantic_fn_review.jsonl.gz"
    ),
) -> dict[str, Any]:
    return {
        "schema_version": "multiskill_training_v2",
        "task": "multi_skill_retrieval",
        "storage_format": "jsonl.gz",
        "inputs": {
            "single_biencoder": single_biencoder_path.as_posix(),
            "single_reranker": single_reranker_path.as_posix(),
            "multiskill_queries": multiskill_queries_path.as_posix(),
            "skills": skills_path.as_posix(),
        },
        "negative_mining": {
            "semantic_model": negative_model,
            "semantic_top_k": semantic_top_k,
            "semantic_fn_threshold": semantic_fn_threshold,
            "sources": {"semantic": 4, "bm25": 3, "same_category": 2, "random": 1},
            "multi_positive_policy": "exclude_every_positive_skill_id_from_negatives",
        },
        "mixture": {
            "strategy": "single_pass_type_weighted_loss",
            "biencoder_multi_loss_weight": biencoder_multi_loss_weight,
            "reranker_multi_loss_weight": reranker_multi_loss_weight,
            "seed": seed,
            "replicated_multiskill_queries": False,
            "interleave_policy": "shuffle_all_examples_each_epoch",
        },
        "counts": counts,
        "outputs": {
            "biencoder": {"path": "biencoder.jsonl.gz", "records": counts["biencoder_records"]},
            "reranker": {"path": "reranker.jsonl.gz", "records": counts["reranker_groups"]},
            "semantic_fn_review": {
                "path": semantic_review_path.name,
                "records": counts.get("multiskill_semantic_fn_removed", 0),
            },
        },
    }


def _output_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "biencoder": output_dir / "biencoder.jsonl.gz",
        "reranker": output_dir / "reranker.jsonl.gz",
        "manifest": output_dir / "manifest.json",
        "semantic_fn_review": output_dir / "semantic_fn_review.jsonl.gz",
    }


def _validate_inputs(args: argparse.Namespace) -> None:
    for path in (
        args.single_biencoder,
        args.single_reranker,
        args.multiskill_queries,
        args.skills,
    ):
        if not path.is_file():
            raise FileNotFoundError(f"input file does not exist: {path}")
    if args.biencoder_multi_loss_weight <= 0:
        raise ValueError("biencoder multi-Skill loss weight must be positive")
    if args.reranker_multi_loss_weight <= 0:
        raise ValueError("reranker multi-Skill loss weight must be positive")
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
        raise ValueError("multi-Skill queries must contain non-empty query_id values")
    if len(set(query_ids)) != len(query_ids):
        raise ValueError("multi-Skill queries contain duplicate query_id values")


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

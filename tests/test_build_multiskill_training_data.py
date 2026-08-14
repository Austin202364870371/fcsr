import unittest

import numpy as np
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from data_io import stream_jsonl, write_jsonl_atomic
from scripts.build_multiskill_training_data import (
    build_manifest,
    build_parser,
    filter_semantic_false_negatives,
    run,
)


class BuildMultiskillTrainingDataTests(unittest.TestCase):
    def test_parser_defaults_to_the_weighted_single_pass_layout(self) -> None:
        args = build_parser().parse_args([])

        self.assertEqual(args.biencoder_multi_loss_weight, 1.5)
        self.assertEqual(args.reranker_multi_loss_weight, 3.0)
        self.assertEqual(args.output_dir, Path("data/training"))
        self.assertEqual(args.semantic_top_k, 64)
        self.assertIsNone(args.semantic_review)

    def test_semantic_filter_rejects_candidates_close_to_any_positive(self) -> None:
        candidates = [
            {"skill_id": "dev/near-a", "source": "bm25", "score": 0.9},
            {"skill_id": "dev/near-b", "source": "same_category", "score": 0.8},
            {"skill_id": "dev/far", "source": "random", "score": 0.7},
        ]
        skill_embeddings = np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [0.99, 0.01],
                [0.01, 0.99],
                [0.7, 0.7],
            ],
            dtype=np.float32,
        )
        index_by_id = {
            "dev/a": 0,
            "dev/b": 1,
            "dev/near-a": 2,
            "dev/near-b": 3,
            "dev/far": 4,
        }

        review_records = []
        kept = filter_semantic_false_negatives(
            candidates,
            ["dev/a", "dev/b"],
            skill_embeddings,
            index_by_id,
            threshold=0.95,
            query_id="compq::one",
            review_records=review_records,
            filter_scope="selected_all_sources",
        )

        self.assertEqual([item["skill_id"] for item in kept], ["dev/far"])
        self.assertEqual(
            [
                (
                    item["query_id"],
                    item["positive_skill_id"],
                    item["skill_id"],
                    item["candidate_source"],
                    item["filter_scope"],
                )
                for item in review_records
            ],
            [
                (
                    "compq::one",
                    "dev/a",
                    "dev/near-a",
                    "bm25",
                    "selected_all_sources",
                ),
                (
                    "compq::one",
                    "dev/b",
                    "dev/near-b",
                    "same_category",
                    "selected_all_sources",
                ),
            ],
        )

    def test_run_writes_directly_trainable_outputs(self) -> None:
        def make_skill(skill_id: str) -> dict:
            return {
                "skill_id": skill_id,
                "name": skill_id,
                "description": f"Description {skill_id}",
                "body": f"Distinct detailed workflow for {skill_id}.",
                "category": "development",
            }

        with TemporaryDirectory(dir=Path(__file__).parent) as directory:
            root = Path(directory)
            single_biencoder = root / "single-bi.jsonl.gz"
            single_reranker = root / "single-rank.jsonl.gz"
            compositional = root / "comp.jsonl.gz"
            skills = root / "skills.jsonl.gz"
            output_dir = root / "output"
            write_jsonl_atomic(single_biencoder, [{
                "query_id": "syn::one", "query": "one", "positive_skill_id": "dev/a",
                "negative_candidates": [{"skill_id": "dev/c", "source": "semantic"}],
            }])
            write_jsonl_atomic(single_reranker, [{
                "query_id": "syn::one", "query": "one",
                "candidates": [{"skill_id": "dev/a", "rank": 1, "retrieval_score": 1.0, "label": 1}],
                "positive_mask": [True],
            }])
            write_jsonl_atomic(compositional, [{
                "query_id": "compq::one", "query": "complete a then b",
                "positive_skill_ids": ["dev/a", "dev/b"],
            }])
            write_jsonl_atomic(skills, [make_skill(f"dev/{letter}") for letter in "abcdefgh"])
            args = build_parser().parse_args([
                "--single-biencoder", str(single_biencoder),
                "--single-reranker", str(single_reranker),
                "--multiskill-queries", str(compositional),
                "--skills", str(skills),
                "--output-dir", str(output_dir),
                "--no-progress",
            ])
            semantic = {"compq::one": [
                {"skill_id": f"dev/{letter}", "score": 1.0 - index / 10}
                for index, letter in enumerate("bcdefgh")
            ]}
            with patch(
                "scripts.build_multiskill_training_data.mine_semantic_candidates",
                return_value=semantic,
            ):
                summary = run(args)

            self.assertEqual(summary["biencoder_records"], 3)
            self.assertEqual(summary["reranker_groups"], 2)
            self.assertEqual(summary["multiskill_embedding_fn_removed"], 0)
            self.assertTrue((output_dir / "semantic_fn_review.jsonl.gz").is_file())
            groups = list(stream_jsonl(output_dir / "reranker.jsonl.gz"))
            self.assertEqual(sum(groups[1]["positive_mask"]), 2)
            self.assertEqual(groups[1]["loss_weight"], 3.0)
    def test_manifest_records_sources_and_output_counts(self) -> None:
        manifest = build_manifest(
            single_biencoder_path=Path("single-bi.jsonl.gz"),
            single_reranker_path=Path("single-rank.jsonl.gz"),
            multiskill_queries_path=Path("comp.jsonl.gz"),
            skills_path=Path("skills.jsonl.gz"),
            negative_model="models/Qwen3-Embedding-0.6B",
            semantic_top_k=96,
            biencoder_multi_loss_weight=1.5,
            reranker_multi_loss_weight=3.0,
            seed=42,
            counts={
                "single_biencoder": 7342,
                "single_reranker": 7342,
                "multiskill_queries": 541,
                "biencoder_records": 10741,
                "reranker_groups": 8965,
            },
        )

        self.assertEqual(manifest["schema_version"], "multiskill_training_v2")
        self.assertEqual(manifest["mixture"]["biencoder_multi_loss_weight"], 1.5)
        self.assertEqual(manifest["mixture"]["reranker_multi_loss_weight"], 3.0)
        self.assertFalse(manifest["mixture"]["replicated_multiskill_queries"])
        self.assertEqual(manifest["negative_mining"]["semantic_top_k"], 96)
        self.assertEqual(
            manifest["negative_mining"]["embedding_fn_scope"],
            "all_sources_against_every_positive_before_selection",
        )
        self.assertTrue(
            manifest["negative_mining"]["refill_after_embedding_fn_filter"]
        )
        self.assertEqual(manifest["outputs"]["biencoder"]["records"], 10741)
        self.assertEqual(
            manifest["outputs"]["semantic_fn_review"],
            {"path": "semantic_fn_review.jsonl.gz", "records": 0},
        )


if __name__ == "__main__":
    unittest.main()

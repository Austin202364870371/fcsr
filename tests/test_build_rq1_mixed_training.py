import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from data_io import stream_jsonl, write_jsonl_atomic
from scripts.build_rq1_mixed_training import build_manifest, build_parser, run


class BuildRq1MixedTrainingScriptTests(unittest.TestCase):
    def test_parser_defaults_to_the_threefold_rq1_layout(self) -> None:
        args = build_parser().parse_args([])

        self.assertEqual(args.multiplier, 3)
        self.assertEqual(args.output_dir, Path("data/training/rq1-mixed-3x"))
        self.assertEqual(args.semantic_top_k, 64)

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
                "--compositional", str(compositional),
                "--skills", str(skills),
                "--output-dir", str(output_dir),
                "--no-progress",
            ])
            semantic = {"compq::one": [
                {"skill_id": f"dev/{letter}", "score": 1.0 - index / 10}
                for index, letter in enumerate("bcdefgh")
            ]}
            with patch(
                "scripts.build_rq1_mixed_training.mine_semantic_candidates",
                return_value=semantic,
            ):
                summary = run(args)

            self.assertEqual(summary["biencoder_records"], 7)
            self.assertEqual(summary["reranker_groups"], 4)
            groups = list(stream_jsonl(output_dir / "reranker.jsonl.gz"))
            self.assertEqual(sum(groups[1]["positive_mask"]), 2)
    def test_manifest_records_sources_and_output_counts(self) -> None:
        manifest = build_manifest(
            single_biencoder_path=Path("single-bi.jsonl.gz"),
            single_reranker_path=Path("single-rank.jsonl.gz"),
            compositional_path=Path("comp.jsonl.gz"),
            skills_path=Path("skills.jsonl.gz"),
            negative_model="models/Qwen3-Embedding-0.6B",
            semantic_top_k=96,
            multiplier=3,
            seed=42,
            counts={
                "single_biencoder": 7342,
                "single_reranker": 7342,
                "compositional_queries": 541,
                "biencoder_records": 10741,
                "reranker_groups": 8965,
            },
        )

        self.assertEqual(manifest["schema_version"], "rq1_mixed_training_v1")
        self.assertEqual(manifest["sampling"]["compositional_multiplier"], 3)
        self.assertEqual(manifest["negative_mining"]["semantic_top_k"], 96)
        self.assertEqual(manifest["outputs"]["biencoder"]["records"], 10741)


if __name__ == "__main__":
    unittest.main()
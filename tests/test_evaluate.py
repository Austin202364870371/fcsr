import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from evaluation import evaluate_predictions
from scripts.evaluate import build_parser, run_bm25


class EvaluationTests(unittest.TestCase):
    def test_matches_core_protocol_and_aggregates_single_multi(self) -> None:
        tasks = [
            {
                "query_id": "generic",
                "task_type": "generic_only",
                "core_gold_skill_ids": ["g"],
                "relevance": {"g": 3},
            },
            {
                "query_id": "single",
                "core_gold_skill_ids": ["p"],
                "relevance": {"p": 3, "d": 1},
            },
            {
                "query_id": "multi",
                "core_gold_skill_ids": ["p1", "p2"],
                "relevance": {"p1": 3, "p2": 3},
            },
        ]
        predictions = {
            "generic": ["g"],
            "single": ["outside", "d", "p"],
            "multi": ["p1", "p2"],
        }
        pool_ids = {"g", "p", "d", "p1", "p2"}

        result = evaluate_predictions(tasks, predictions, pool_ids)

        self.assertEqual(result.summary["all"]["count"], 2)
        self.assertEqual(result.summary["single"]["count"], 1)
        self.assertEqual(result.summary["multi"]["count"], 1)
        self.assertLess(result.summary["single"]["nDCG@3"], 1.0)
        self.assertEqual(result.summary["multi"]["FullCoverage@3"], 1.0)
        self.assertEqual(result.skipped_generic_only, 1)

    def test_intersects_ground_truth_and_relevance_with_tier_pool(self) -> None:
        tasks = [
            {
                "task_id": "multi",
                "core_gt_ids": ["p1", "p2"],
                "relevance": {"p1": 3, "p2": 3, "degraded": 1},
            }
        ]

        result = evaluate_predictions(
            tasks,
            {"multi": ["p2", "degraded", "p1"]},
            {"p1", "degraded"},
        )

        self.assertEqual(result.summary["multi"]["count"], 1)
        self.assertEqual(result.details[0]["gt_skill_ids"], ["p1"])
        self.assertEqual(result.details[0]["ranked_skill_ids"], ["degraded", "p1"])



    def test_bm25_exports_standard_retrieval_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            queries = root / "queries.jsonl"
            skills = root / "skills.jsonl"
            predictions = root / "predictions.json"
            records = root / "records.jsonl"
            queries.write_text(
                json.dumps({"query_id": "q1", "query": "alpha"}) + "\n",
                encoding="utf-8",
            )
            skills.write_text(
                "\n".join(
                    json.dumps(item)
                    for item in (
                        {"skill_id": "alpha", "name": "alpha", "description": "", "body": ""},
                        {"skill_id": "beta", "name": "beta", "description": "", "body": ""},
                    )
                ) + "\n",
                encoding="utf-8",
            )

            result = run_bm25(
                Namespace(
                    queries=str(queries),
                    skills=str(skills),
                    output_predictions=str(predictions),
                    output_records=str(records),
                    top_k=2,
                )
            )

            self.assertEqual(result["top_k"], 2)
            self.assertEqual(json.loads(predictions.read_text(encoding="utf-8"))["q1"][0], "alpha")
            record = json.loads(records.read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(record["retrieved_candidates"][0]["skill_id"], "alpha")
    def test_parser_accepts_bm25_and_hybrid_retrieval_baselines(self) -> None:
        parser = build_parser()
        bm25 = parser.parse_args(
            [
                "bm25", "--queries", "tasks.jsonl", "--skills", "skills.jsonl",
                "--output-predictions", "bm25.json", "--output-records", "bm25.jsonl",
            ]
        )
        hybrid = parser.parse_args(
            [
                "hybrid", "--queries", "tasks.jsonl", "--skills", "skills.jsonl",
                "--model", "Qwen/Qwen3-Embedding-0.6B",
                "--output-predictions", "hybrid.json", "--output-records", "hybrid.jsonl",
            ]
        )

        self.assertEqual(bm25.command, "bm25")
        self.assertEqual(bm25.top_k, 50)
        self.assertEqual(hybrid.command, "hybrid")
        self.assertEqual(hybrid.fusion_depth, 100)
if __name__ == "__main__":
    unittest.main()

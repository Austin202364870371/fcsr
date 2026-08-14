import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from data_io import stream_jsonl, write_jsonl_atomic
from generate_multiskill_queries import (
    _select_pilot_candidates,
    build_parser,
    run,
)


class FakeClient:
    def complete(self, messages, *, temperature: float, max_new_tokens: int) -> str:
        return """{
          "query": "Prepare a complete API contract for an implementation team, then validate the endpoint definitions, request and response schemas, authentication behavior, error handling, versioning notes, and acceptance criteria before publishing a final specification and validation report for release approval.",
          "positive_skill_ids": ["build-api", "validate-api"],
          "subtasks": [
            {"step_id": "s1", "skill_id": "build-api", "instruction": "Draft the API specification."},
            {"step_id": "s2", "skill_id": "validate-api", "instruction": "Validate the completed specification."}
          ],
          "dependencies": [{"from_step_id": "s1", "to_step_id": "s2"}]
        }"""


class GenerateCompositionalQueriesScriptTests(unittest.TestCase):
    def test_parser_supports_explicit_progress_controls(self) -> None:
        parser = build_parser()

        self.assertIsNone(parser.parse_args([]).progress)
        self.assertTrue(parser.parse_args(["--progress"]).progress)
        self.assertFalse(parser.parse_args(["--no-progress"]).progress)

    def test_limit_uses_deterministic_type_stratification(self) -> None:
        candidates = [
            {
                "candidate_id": f"pair-{index}",
                "candidate_type": "pair",
            }
            for index in range(8)
        ] + [
            {
                "candidate_id": f"triple-{index}",
                "candidate_type": "triple",
            }
            for index in range(2)
        ]

        selected = _select_pilot_candidates(candidates, 5)

        self.assertEqual(
            [item["candidate_id"] for item in selected],
            ["pair-0", "pair-2", "pair-4", "pair-7", "triple-0"],
        )
        self.assertIs(_select_pilot_candidates(candidates, None), candidates)

    def test_run_writes_gzip_outputs_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            root = Path(directory)
            candidates = root / "candidates.jsonl.gz"
            contracts = root / "contracts.jsonl.gz"
            output = root / "queries.jsonl.gz"
            failures = root / "failures.jsonl.gz"
            review = root / "review.jsonl.gz"
            manifest = root / "manifest.json"
            write_jsonl_atomic(
                candidates,
                [
                    {
                        "candidate_id": "comp::pair::01",
                        "candidate_type": "pair",
                        "skill_ids": ["build-api", "validate-api"],
                        "edges": [
                            {
                                "from_skill_id": "build-api",
                                "to_skill_id": "validate-api",
                                "operation_relation": "producer_to_consumer",
                                "matched_artifact_tokens": ["openapi"],
                            }
                        ],
                    }
                ],
            )
            write_jsonl_atomic(contracts, [contract("build-api"), contract("validate-api")])
            manifest.write_text("{}", encoding="utf-8")

            summary = run(
                argparse.Namespace(
                    candidates=candidates,
                    contracts=contracts,
                    output=output,
                    failures=failures,
                    review_queue=review,
                    manifest=manifest,
                    model="deepseek-v4-flash",
                    concurrency=16,
                    timeout=180.0,
                    temperature=0.2,
                    max_new_tokens=1024,
                    max_attempts=1,
                    min_query_words=30,
                    max_query_words=260,
                    limit=None,
                    overwrite=False,
                    dry_run=False,
                ),
                client=FakeClient(),
            )

            self.assertEqual(summary["queries"], 1)
            self.assertEqual(len(list(stream_jsonl(output))), 1)
            self.assertEqual(list(stream_jsonl(failures)), [])
            saved_manifest = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertEqual(saved_manifest["status"], "queries_generated")
            self.assertEqual(
                saved_manifest["query_generation"]["candidate_types"],
                {"pair": 1},
            )
            self.assertEqual(saved_manifest["artifacts"]["queries"]["records"], 1)


def contract(skill_id: str) -> dict:
    return {
        "skill_id": skill_id,
        "source_hash": f"hash-{skill_id}",
        "extraction": {"status": "validated"},
        "capability": {"summary": f"Capability for {skill_id}."},
        "operations": [],
        "inputs": [],
        "outputs": [],
        "constraints": [],
        "quality_criteria": [],
    }


if __name__ == "__main__":
    unittest.main()

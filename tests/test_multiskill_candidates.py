import argparse
import json
import tempfile
import unittest
from pathlib import Path

from multiskill_candidates import CandidateSettings, build_compositional_candidates
from data_io import load_jsonl, write_jsonl_atomic
from scripts.build_multiskill_candidates import run


def contract(skill_id, source_hash, outputs, inputs, actions):
    return {
        "skill_id": skill_id,
        "source_hash": source_hash,
        "outputs": [{"artifact": value, "format": None} for value in outputs],
        "inputs": [
            {"artifact": value, "format": None, "required": True}
            for value in inputs
        ],
        "operations": [{"action": value} for value in actions],
        "dependencies": [],
        "extraction": {"status": "validated"},
    }


class CompositionalCandidateTests(unittest.TestCase):
    def test_builds_ordered_pairs_and_triples_from_contract_handoffs(self):
        contracts = [
            contract("a", "hash-a", ["OpenAPI specification"], [], ["generate"]),
            contract(
                "b",
                "hash-b",
                ["validation report"],
                ["OpenAPI specification"],
                ["validate"],
            ),
            contract(
                "c",
                "hash-c",
                [],
                ["validation report"],
                ["publish"],
            ),
        ]
        queries = [
            {"positive_skill_id": "a", "source_hash": "hash-a"},
            {"positive_skill_id": "b", "source_hash": "hash-b"},
            {"positive_skill_id": "c", "source_hash": "hash-c"},
        ]

        result = build_compositional_candidates(
            contracts,
            queries,
            set(),
            CandidateSettings(max_pairs=10, max_triples=10, max_pairs_per_source=10),
        )

        self.assertEqual([item["skill_ids"] for item in result.pairs], [["a", "b"], ["b", "c"]])
        self.assertEqual([item["skill_ids"] for item in result.triples], [["a", "b", "c"]])
        self.assertEqual(result.pairs[0]["edges"][0]["matched_artifact_tokens"], ["openapi", "specification"])

    def test_rejects_stale_benchmark_and_non_complementary_candidates(self):
        contracts = [
            contract("producer", "fresh", ["release manifest"], [], ["generate"]),
            contract("same-phase", "hash-s", [], ["release manifest"], ["generate"]),
            contract("stale", "current", ["release manifest"], [], ["generate"]),
            contract("consumer", "hash-c", [], ["release manifest"], ["validate"]),
        ]
        queries = [
            {"positive_skill_id": "producer", "source_hash": "fresh"},
            {"positive_skill_id": "same-phase", "source_hash": "hash-s"},
            {"positive_skill_id": "stale", "source_hash": "old"},
            {"positive_skill_id": "consumer", "source_hash": "hash-c"},
        ]

        result = build_compositional_candidates(
            contracts,
            queries,
            {"consumer"},
            CandidateSettings(max_pairs=10, max_triples=10, max_pairs_per_source=10),
        )

        self.assertEqual(result.pairs, [])
        self.assertEqual(result.eligible_skill_ids, ["producer", "same-phase"])
        self.assertIn(
            ["producer", "same-phase"],
            [item["skill_ids"] for item in result.rejections],
        )
        rejection = next(
            item for item in result.rejections if item["skill_ids"] == ["producer", "same-phase"]
        )
        self.assertEqual(rejection["reasons"], ["missing_complementary_operation"])


    def test_rejects_a_single_loose_artifact_token_overlap(self):
        contracts = [
            contract("a", "hash-a", ["requirements memo"], [], ["generate"]),
            contract("b", "hash-b", [], ["backup requirements"], ["validate"]),
        ]
        queries = [
            {"positive_skill_id": "a", "source_hash": "hash-a"},
            {"positive_skill_id": "b", "source_hash": "hash-b"},
        ]

        result = build_compositional_candidates(
            contracts,
            queries,
            set(),
            CandidateSettings(max_pairs=10, max_triples=10, max_pairs_per_source=10),
        )

        self.assertEqual(result.pairs, [])
        rejection = next(
            item for item in result.rejections if item["skill_ids"] == ["a", "b"]
        )
        self.assertEqual(rejection["reasons"], ["weak_artifact_handoff"])

    def test_accepts_phrase_containment_and_explicit_identifiers(self):
        contracts = [
            contract(
                "phrase-producer",
                "hash-phrase-producer",
                ["OpenAPI specification"],
                [],
                ["generate"],
            ),
            contract(
                "phrase-consumer",
                "hash-phrase-consumer",
                [],
                ["versioned signed OpenAPI specification deployment bundle"],
                ["validate"],
            ),
            contract(
                "identifier-producer",
                "hash-identifier-producer",
                ["Dockerfile"],
                [],
                ["write"],
            ),
            contract(
                "identifier-consumer",
                "hash-identifier-consumer",
                [],
                ["production Dockerfile"],
                ["audit"],
            ),
        ]
        contracts[2]["outputs"][0]["format"] = "dockerfile"
        contracts[3]["inputs"][0]["format"] = "dockerfile"
        queries = [
            {"positive_skill_id": item["skill_id"], "source_hash": item["source_hash"]}
            for item in contracts
        ]

        result = build_compositional_candidates(
            contracts,
            queries,
            set(),
            CandidateSettings(max_pairs=10, max_triples=10, max_pairs_per_source=10),
        )

        self.assertIn(
            ["phrase-producer", "phrase-consumer"],
            [item["skill_ids"] for item in result.pairs],
        )
        self.assertIn(
            ["identifier-producer", "identifier-consumer"],
            [item["skill_ids"] for item in result.pairs],
        )

    def test_rejects_uncorroborated_single_identifier(self):
        contracts = [
            contract("a", "hash-a", ["Dockerfile"], [], ["write"]),
            contract("b", "hash-b", [], ["production Dockerfile"], ["audit"]),
        ]
        queries = [
            {"positive_skill_id": item["skill_id"], "source_hash": item["source_hash"]}
            for item in contracts
        ]

        result = build_compositional_candidates(
            contracts,
            queries,
            set(),
            CandidateSettings(max_pairs=10, max_triples=10, max_pairs_per_source=10),
        )

        self.assertEqual(result.pairs, [])
        rejection = next(
            item for item in result.rejections if item["skill_ids"] == ["a", "b"]
        )
        self.assertEqual(rejection["reasons"], ["weak_artifact_handoff"])

    def test_run_writes_gzip_artifacts_and_updates_manifest(self):
        contracts = [
            contract("a", "hash-a", ["OpenAPI specification"], [], ["generate"]),
            contract(
                "b",
                "hash-b",
                ["validation report"],
                ["OpenAPI specification"],
                ["validate"],
            ),
            contract("c", "hash-c", [], ["validation report"], ["publish"]),
        ]
        queries = [
            {"positive_skill_id": "a", "source_hash": "hash-a"},
            {"positive_skill_id": "b", "source_hash": "hash-b"},
            {"positive_skill_id": "c", "source_hash": "hash-c"},
        ]
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            root = Path(directory)
            contracts_path = root / "contracts.jsonl.gz"
            queries_path = root / "queries.jsonl.gz"
            tasks_path = root / "tasks.jsonl.gz"
            candidates_path = root / "candidates.jsonl.gz"
            rejections_path = root / "candidate_rejections.jsonl.gz"
            manifest_path = root / "manifest.json"
            write_jsonl_atomic(contracts_path, contracts)
            write_jsonl_atomic(queries_path, queries)
            write_jsonl_atomic(tasks_path, [])
            manifest_path.write_text("{}", encoding="utf-8")

            summary = run(
                argparse.Namespace(
                    contracts=contracts_path,
                    queries=queries_path,
                    tasks=tasks_path,
                    output=candidates_path,
                    rejections=rejections_path,
                    manifest=manifest_path,
                    max_pairs=10,
                    max_triples=10,
                    max_pairs_per_source=10,
                    max_artifact_frequency=5,
                    overwrite=False,
                )
            )

            self.assertEqual(summary["pairs"], 2)
            self.assertEqual(summary["triples"], 1)
            self.assertEqual(len(load_jsonl(candidates_path)), 3)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "candidates_generated")
            self.assertEqual(
                manifest["candidate_construction"]["contracts"],
                contracts_path.as_posix(),
            )
if __name__ == "__main__":
    unittest.main()

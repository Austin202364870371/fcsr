import copy
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from contract_schema import compute_source_hash
from data_io import load_jsonl, write_jsonl_atomic
from preprocessing import (
    LLMConfig,
    build_contract_messages,
    build_query_messages,
    extract_contracts,
    generate_queries,
)


SKILL = {
    "skill_id": "design/affordances",
    "name": "affordances",
    "description": "Las affordances sugieren cómo puede ser usado un objeto.",
    "body": "Los botones con sombra parecen presionables.",
    "category": "design",
}


def semantic_contract(quote: str | None = None) -> dict:
    return {
        "source_languages": ["es"],
        "canonical_language": "en",
        "capability": {
            "summary": "Design interfaces whose controls communicate their use.",
            "evidence_quotes": [
                {
                    "source_field": "description",
                    "quote": quote or SKILL["description"],
                }
            ],
        },
        "operations": [
            {
                "action": "design",
                "target": "interface controls",
                "outcome": "make intended interactions discoverable",
                "qualifiers": ["use visible interaction cues"],
                "evidence_quotes": [
                    {"source_field": "body", "quote": SKILL["body"]}
                ],
            }
        ],
        "inputs": [],
        "outputs": [],
        "preconditions": [],
        "constraints": [],
        "dependencies": [],
        "exclusions": [],
        "quality_criteria": [],
    }


def valid_generated_query() -> str:
    return (
        "Redesign a checkout form so shoppers can identify which controls are "
        "clickable and understand the result of each interaction without written "
        "instructions. Review the visual treatment of buttons, links, text fields, "
        "checkboxes, and sliders. Make each control communicate its intended use "
        "through visible cues and clear interaction states while preserving the "
        "existing checkout flow. Provide revised interface specifications for the "
        "design team and explain how the proposed controls improve discoverability "
        "for first-time shoppers across desktop, mobile, and modern touchscreen devices."
    )


class FakeClient:
    def __init__(self, responses: list[object]) -> None:
        self.responses = list(responses)
        self.calls = 0
        self.requests: list[dict[str, object]] = []

    def complete(self, **kwargs: object) -> str:
        self.calls += 1
        self.requests.append(kwargs)
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        if isinstance(response, str):
            return response
        return json.dumps(response, ensure_ascii=False)


class ConcurrentFakeClient:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.active = 0
        self.max_active = 0
        self.calls: dict[str, int] = {}

    def complete(self, **kwargs: object) -> str:
        content = kwargs["messages"][-1]["content"]
        skill_id = (
            "design/affordances-mobile"
            if "design/affordances-mobile" in content
            else SKILL["skill_id"]
        )
        with self.lock:
            attempt = self.calls.get(skill_id, 0) + 1
            self.calls[skill_id] = attempt
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        time.sleep(0.02)
        with self.lock:
            self.active -= 1
        response = semantic_contract(
            "not present in the source"
            if skill_id == SKILL["skill_id"] and attempt == 1
            else None
        )
        return json.dumps(response, ensure_ascii=False)


class LLMPreprocessingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(dir=Path(__file__).parent)
        self.root = Path(self.temp.name)
        self.sample = self.root / "sample.jsonl"
        self.contracts = self.root / "contracts.jsonl"
        self.failures = self.root / "failures.jsonl"
        self.queries = self.root / "queries.jsonl"
        write_jsonl_atomic(self.sample, [SKILL])
        self.config = LLMConfig(
            model="deepseek-v4-flash",
            max_attempts=2,
            backoff_seconds=0,
        )

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_accepts_multilingual_evidence_backed_contract(self) -> None:
        client = FakeClient([semantic_contract()])

        summary = extract_contracts(
            self.sample,
            self.contracts,
            self.failures,
            client,
            self.config,
        )

        contract = load_jsonl(self.contracts)[0]
        self.assertEqual(summary.succeeded, 1)
        self.assertEqual(
            set(client.requests[0]),
            {"messages", "temperature"},
        )
        self.assertEqual(contract["source_languages"], ["es"])
        self.assertEqual(contract["canonical_language"], "en")
        self.assertEqual(
            contract["evidence"][0]["quote"],
            SKILL["description"],
        )

    def test_success_prunes_resolved_failure_but_keeps_unrelated_failure(self) -> None:
        write_jsonl_atomic(
            self.failures,
            [
                {
                    "stage": "contract",
                    "skill_id": SKILL["skill_id"],
                    "source_hash": compute_source_hash(SKILL),
                    "attempts": 2,
                    "error_type": "ValueError",
                    "error": "old failure",
                },
                {
                    "stage": "contract",
                    "skill_id": "other/skill",
                    "source_hash": "f" * 64,
                    "attempts": 3,
                    "error_type": "ValueError",
                    "error": "unresolved",
                },
            ],
        )

        extract_contracts(
            self.sample,
            self.contracts,
            self.failures,
            FakeClient([semantic_contract()]),
            self.config,
        )

        failures = load_jsonl(self.failures)
        self.assertEqual(len(failures), 1)
        self.assertEqual(failures[0]["skill_id"], "other/skill")
    def test_invalid_evidence_retries_with_exponential_backoff(self) -> None:
        client = FakeClient(
            [
                semantic_contract("not an exact source quote"),
                semantic_contract(),
            ]
        )
        config = LLMConfig(
            model="deepseek-v4-flash",
            max_attempts=2,
            backoff_seconds=2,
        )

        with patch("preprocessing.time.sleep") as sleep:
            summary = extract_contracts(
                self.sample,
                self.contracts,
                self.failures,
                client,
                config,
            )

        self.assertEqual(summary.succeeded, 1)
        self.assertEqual(client.calls, 2)
        sleep.assert_called_once_with(2)

        retry_message = client.requests[1]["messages"][-1]["content"]
        self.assertIn("failed deterministic validation", retry_message)
        self.assertIn("not an exact source quote", retry_message)

    def test_aligns_markdown_only_evidence_difference_to_exact_source(self) -> None:
        markdown_line = (
            "- **Fetches OpenAPI 3.1.0 specification** "
            "from MikoPBX (~9MB, 259 endpoints)"
        )
        skill = {**SKILL, "body": f"{SKILL['body']}\n{markdown_line}"}
        write_jsonl_atomic(self.sample, [skill])
        semantic = semantic_contract()
        semantic["capability"]["evidence_quotes"] = [
            {
                "source_field": "body",
                "quote": (
                    "Fetches OpenAPI 3.1.0 specification "
                    "from MikoPBX (~9MB, 259 endpoints)"
                ),
            }
        ]

        summary = extract_contracts(
            self.sample,
            self.contracts,
            self.failures,
            FakeClient([semantic]),
            self.config,
        )

        self.assertEqual(summary.succeeded, 1)
        contract = load_jsonl(self.contracts)[0]
        capability_evidence = next(
            item
            for item in contract["evidence"]
            if item["evidence_id"] in contract["capability"]["evidence_ids"]
        )
        source_slice = skill["body"][
            capability_evidence["start_char"] : capability_evidence["end_char"]
        ]
        self.assertEqual(capability_evidence["quote"], source_slice)
        self.assertIn("**", capability_evidence["quote"])
        self.assertIn(
            "evidence_quote_markdown_aligned:capability:body",
            contract["extraction"]["warnings"],
        )

    def test_drops_optional_item_when_all_its_evidence_is_unsupported(self) -> None:
        semantic = semantic_contract()
        semantic["operations"].append(
            {
                "action": "send message",
                "target": "message bus",
                "outcome": "deliver an agent message",
                "qualifiers": [],
                "evidence_quotes": [
                    {
                        "source_field": "body",
                        "quote": "await self.send_message(message)",
                    }
                ],
            }
        )

        summary = extract_contracts(
            self.sample,
            self.contracts,
            self.failures,
            FakeClient([semantic]),
            self.config,
        )

        self.assertEqual(summary.succeeded, 1)
        contract = load_jsonl(self.contracts)[0]
        self.assertEqual(len(contract["operations"]), 1)
        self.assertIn(
            "dropped_unsupported_item:operations[1]",
            contract["extraction"]["warnings"],
        )

    def test_contract_prompt_caps_body_and_query_prompt_does_not_mutate_contract(self) -> None:
        long_skill = {**SKILL, "body": "x" * 13000}
        messages = build_contract_messages(long_skill)
        self.assertNotIn("x" * 12001, messages[1]["content"])

        client = FakeClient([semantic_contract()])
        extract_contracts(
            self.sample,
            self.contracts,
            self.failures,
            client,
            self.config,
        )
        contract = load_jsonl(self.contracts)[0]
        before = copy.deepcopy(contract)
        build_query_messages(SKILL, contract)
        self.assertEqual(contract, before)

    def test_contract_prompt_defines_every_collection_item_shape(self) -> None:
        messages = build_contract_messages(SKILL)
        user_prompt = messages[1]["content"]
        schema_text = user_prompt.split(
            "Extract the contract using exactly this response shape:\n", 1
        )[1].split("\n\nSource skill:\n", 1)[0]
        schema = json.loads(schema_text)

        for field in ("inputs", "outputs"):
            self.assertEqual(
                set(schema[field][0]),
                {"artifact", "format", "required", "constraints", "evidence_quotes"},
            )
        for field in (
            "preconditions",
            "constraints",
            "exclusions",
            "quality_criteria",
        ):
            self.assertEqual(set(schema[field][0]), {"statement", "evidence_quotes"})
        self.assertEqual(
            set(schema["dependencies"][0]),
            {"name", "type", "required", "evidence_quotes"},
        )
        self.assertIn("Accuracy is more important than field coverage", messages[0]["content"])
        self.assertIn(
            "never infer an exclusion from a positive capability",
            messages[0]["content"],
        )
        self.assertIn("Quality criteria require an explicit check", messages[0]["content"])
        self.assertEqual(self.config.contract_prompt_version, "contract_v2_prompt_004")

    def test_query_prompt_requires_single_skill_grounding(self) -> None:
        client = FakeClient([semantic_contract()])
        extract_contracts(
            self.sample,
            self.contracts,
            self.failures,
            client,
            self.config,
        )
        contract = load_jsonl(self.contracts)[0]

        prompt = build_query_messages(SKILL, contract)[1]["content"]

        self.assertIn("one supplied Skill Contract", prompt)
        self.assertIn("Do not invent URLs", prompt)
        self.assertIn("other specialized capabilities", prompt)
        self.assertIn("orchestration", prompt)
        self.assertIn("business workflow as already existing", prompt)
        self.assertIn("must not become requested deliverables", prompt)
        self.assertIn("strict allowlist of requested work", prompt)
        self.assertIn("Every imperative verb", prompt)
        self.assertIn("Bad: Build a checkout flow", prompt)
        self.assertIn("MUST contain 80-180 English words", prompt)
        self.assertEqual(self.config.query_prompt_version, "contract_query_prompt_005")

    def test_invalid_json_is_retried(self) -> None:
        client = FakeClient(["not json", semantic_contract()])

        summary = extract_contracts(
            self.sample,
            self.contracts,
            self.failures,
            client,
            self.config,
        )

        self.assertEqual(summary.succeeded, 1)
        self.assertEqual(client.calls, 2)
        self.assertEqual(load_jsonl(self.contracts)[0]["extraction"]["attempts"], 2)

    def test_contract_workers_process_one_skill_per_request_and_retry_only_rejected(self) -> None:
        second_skill = {**SKILL, "skill_id": "design/affordances-mobile"}
        write_jsonl_atomic(self.sample, [SKILL, second_skill])
        client = ConcurrentFakeClient()
        config = LLMConfig(
            model="deepseek-v4-flash",
            max_attempts=2,
            backoff_seconds=0,
            batch_size=2,
        )

        summary = extract_contracts(
            self.sample,
            self.contracts,
            self.failures,
            client,
            config,
        )

        self.assertEqual(summary.succeeded, 2)
        self.assertEqual(client.max_active, 2)
        self.assertEqual(client.calls[SKILL["skill_id"]], 2)
        self.assertEqual(client.calls[second_skill["skill_id"]], 1)
        contracts = {item["skill_id"]: item for item in load_jsonl(self.contracts)}
        self.assertEqual(contracts[SKILL["skill_id"]]["extraction"]["attempts"], 2)
        self.assertEqual(contracts[second_skill["skill_id"]]["extraction"]["attempts"], 1)

    def test_evidence_mismatch_is_written_to_failures(self) -> None:
        client = FakeClient(
            [
                semantic_contract("not present in the source"),
                semantic_contract("still not present"),
            ]
        )

        summary = extract_contracts(
            self.sample,
            self.contracts,
            self.failures,
            client,
            self.config,
        )

        self.assertEqual(summary.failed, 1)
        failure = load_jsonl(self.failures)[0]
        self.assertEqual(failure["skill_id"], SKILL["skill_id"])
        self.assertIn("quote", failure["error"])

    def test_matching_skill_and_source_hash_are_skipped_on_resume(self) -> None:
        first_client = FakeClient([semantic_contract()])
        extract_contracts(
            self.sample,
            self.contracts,
            self.failures,
            first_client,
            self.config,
        )
        second_client = FakeClient([])

        summary = extract_contracts(
            self.sample,
            self.contracts,
            self.failures,
            second_client,
            self.config,
        )

        self.assertEqual(summary.skipped, 1)
        self.assertEqual(second_client.calls, 0)
        self.assertEqual(len(load_jsonl(self.contracts)), 1)

    def test_gzip_contract_output_supports_incremental_resume(self) -> None:
        contracts = self.root / "contracts.jsonl.gz"
        failures = self.root / "failures.jsonl.gz"
        extract_contracts(
            self.sample,
            contracts,
            failures,
            FakeClient([semantic_contract()]),
            self.config,
        )

        summary = extract_contracts(
            self.sample,
            contracts,
            failures,
            FakeClient([]),
            self.config,
        )

        self.assertEqual(summary.skipped, 1)
        self.assertEqual(len(load_jsonl(contracts)), 1)

    def test_contract_progress_reports_each_succeeded_or_skipped_skill(self) -> None:
        write_jsonl_atomic(self.sample, [SKILL, SKILL])
        updates = []

        summary = extract_contracts(
            self.sample,
            self.contracts,
            self.failures,
            FakeClient([semantic_contract()]),
            self.config,
            progress=updates.append,
        )

        self.assertEqual(summary.succeeded, 1)
        self.assertEqual(summary.skipped, 1)
        self.assertEqual(len(updates), 2)
        self.assertEqual(updates[1], summary)

    def test_query_name_leak_is_retried(self) -> None:
        contract_client = FakeClient([semantic_contract()])
        extract_contracts(
            self.sample,
            self.contracts,
            self.failures,
            contract_client,
            self.config,
        )
        query_client = FakeClient(
            [
                {"query": "Please apply affordances to this interface."},
                {"query": valid_generated_query()},
            ]
        )

        summary = generate_queries(
            self.sample,
            self.contracts,
            self.queries,
            self.failures,
            query_client,
            self.config,
        )

        query = load_jsonl(self.queries)[0]
        self.assertEqual(summary.succeeded, 1)
        self.assertEqual(query_client.calls, 2)
        self.assertNotIn(SKILL["name"], query["query"].lower())
        self.assertEqual(query["positive_skill_id"], SKILL["skill_id"])
        self.assertEqual(
            query["generator"]["prompt_version"],
            "contract_query_prompt_005",
        )

    def test_query_progress_reports_each_succeeded_or_skipped_skill(self) -> None:
        extract_contracts(
            self.sample,
            self.contracts,
            self.failures,
            FakeClient([semantic_contract()]),
            self.config,
        )
        write_jsonl_atomic(self.sample, [SKILL, SKILL])
        updates = []

        summary = generate_queries(
            self.sample,
            self.contracts,
            self.queries,
            self.failures,
            FakeClient([{"query": valid_generated_query()}]),
            self.config,
            progress=updates.append,
        )

        self.assertEqual(summary.succeeded, 1)
        self.assertEqual(summary.skipped, 1)
        self.assertEqual(len(updates), 2)
        self.assertEqual(updates[0].succeeded, 1)
        self.assertEqual(updates[1], summary)

    def test_query_outside_word_limit_is_retried(self) -> None:
        extract_contracts(
            self.sample,
            self.contracts,
            self.failures,
            FakeClient([semantic_contract()]),
            self.config,
        )
        query_client = FakeClient(
            [
                {"query": "Make the controls clearer."},
                {"query": valid_generated_query()},
            ]
        )

        summary = generate_queries(
            self.sample,
            self.contracts,
            self.queries,
            self.failures,
            query_client,
            self.config,
        )

        self.assertEqual(summary.succeeded, 1)
        self.assertEqual(query_client.calls, 2)
        self.assertEqual(
            load_jsonl(self.queries)[0]["query"],
            valid_generated_query(),
        )

    def test_missing_contract_prevents_query_generation(self) -> None:
        client = FakeClient([])
        updates = []

        summary = generate_queries(
            self.sample,
            self.contracts,
            self.queries,
            self.failures,
            client,
            self.config,
            progress=updates.append,
        )

        self.assertEqual(summary.failed, 1)
        self.assertEqual(client.calls, 0)
        self.assertIn("missing contract", load_jsonl(self.failures)[0]["error"])
        self.assertEqual(updates, [summary])


if __name__ == "__main__":
    unittest.main()

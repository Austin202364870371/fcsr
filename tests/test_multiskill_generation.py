import threading
import time
import unittest

from multiskill_generation import (
    CompositionalGenerationConfig,
    generate_compositional_queries,
)


class FakeClient:
    def __init__(self, response: str) -> None:
        self.response = response
        self.calls = []

    def complete(self, messages, *, temperature: float, max_new_tokens: int) -> str:
        self.calls.append(
            {
                "messages": messages,
                "temperature": temperature,
                "max_new_tokens": max_new_tokens,
            }
        )
        return self.response


class SequencedClient:
    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.calls = []

    def complete(self, messages, *, temperature: float, max_new_tokens: int) -> str:
        self.calls.append(messages)
        return self.responses.pop(0)


class ConcurrentClient(FakeClient):
    def __init__(self, response: str) -> None:
        super().__init__(response)
        self.lock = threading.Lock()
        self.active = 0
        self.max_active = 0

    def complete(self, messages, *, temperature: float, max_new_tokens: int) -> str:
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        time.sleep(0.02)
        with self.lock:
            self.active -= 1
        return super().complete(
            messages, temperature=temperature, max_new_tokens=max_new_tokens
        )


class CompositionalGenerationTests(unittest.TestCase):
    def test_generates_valid_pair_with_ordered_skills_and_dependency(self) -> None:
        candidate = {
            "candidate_id": "comp::pair::01",
            "candidate_type": "pair",
            "skill_ids": ["build-api", "validate-api"],
            "edges": [
                {
                    "from_skill_id": "build-api",
                    "to_skill_id": "validate-api",
                    "matched_artifact_tokens": ["openapi", "specification"],
                    "operation_relation": "producer_to_consumer",
                }
            ],
        }
        contracts = [
            contract("build-api", "hash-build", "Generate an OpenAPI specification."),
            contract("validate-api", "hash-validate", "Validate an OpenAPI specification."),
        ]
        client = FakeClient(
            """{
              \"query\": \"Create a deployable API contract for a new service and verify that every endpoint, schema, error response, and authentication requirement is internally consistent before it is handed to the implementation team. Provide the validated specification and a concise validation report.\",
              \"positive_skill_ids\": [\"build-api\", \"validate-api\"],
              \"subtasks\": [
                {\"step_id\": \"s1\", \"skill_id\": \"build-api\", \"instruction\": \"Draft the API specification.\"},
                {\"step_id\": \"s2\", \"skill_id\": \"validate-api\", \"instruction\": \"Validate the completed specification.\"}
              ],
              \"dependencies\": [{\"from_step_id\": \"s1\", \"to_step_id\": \"s2\"}]
            }"""
        )

        result = generate_compositional_queries(
            [candidate],
            contracts,
            client,
            CompositionalGenerationConfig(model="deepseek-v4-flash", max_attempts=1),
        )

        self.assertEqual(len(result.queries), 1)
        record = result.queries[0]
        self.assertEqual(record["positive_skill_ids"], ["build-api", "validate-api"])
        self.assertEqual([step["skill_id"] for step in record["subtasks"]], ["build-api", "validate-api"])
        self.assertEqual(record["dependencies"], [{"from_step_id": "s1", "to_step_id": "s2"}])
        self.assertEqual(record["source_hashes"], ["hash-build", "hash-validate"])
        self.assertEqual(record["generator"]["provider"], "deepseek")
        self.assertEqual(len(result.failures), 0)
        self.assertEqual(len(client.calls), 1)
        self.assertIn("producer_to_consumer", client.calls[0]["messages"][1]["content"])


    def test_rejects_payload_with_dependency_not_supported_by_candidate(self) -> None:
        candidate = {
            "candidate_id": "comp::pair::02",
            "candidate_type": "pair",
            "skill_ids": ["build-api", "validate-api"],
            "edges": [
                {
                    "from_skill_id": "build-api",
                    "to_skill_id": "validate-api",
                    "matched_artifact_tokens": ["openapi"],
                    "operation_relation": "producer_to_consumer",
                }
            ],
        }
        client = FakeClient(
            """{
              "query": "Prepare an API contract that a delivery team can implement, review, and hand off with a written report that documents the validation work completed before release approval. Include endpoint definitions, request and response schemas, authentication behavior, error handling, versioning notes, and an acceptance checklist for the implementation team.",
              "positive_skill_ids": ["build-api", "validate-api"],
              "subtasks": [
                {"step_id": "s1", "skill_id": "build-api", "instruction": "Draft the specification."},
                {"step_id": "s2", "skill_id": "validate-api", "instruction": "Validate the specification."}
              ],
              "dependencies": [{"from_step_id": "s2", "to_step_id": "s1"}]
            }"""
        )

        result = generate_compositional_queries(
            [candidate],
            [
                contract("build-api", "hash-build", "Generate an OpenAPI specification."),
                contract("validate-api", "hash-validate", "Validate an OpenAPI specification."),
            ],
            client,
            CompositionalGenerationConfig(model="deepseek-v4-flash", max_attempts=1),
        )

        self.assertEqual(result.queries, [])
        self.assertEqual(len(result.failures), 1)
        self.assertIn("handoff edges", result.failures[0]["error"])

    def test_reports_terminal_progress_for_success_and_failure(self) -> None:
        valid_candidate = {
            "candidate_id": "comp::pair::progress",
            "candidate_type": "pair",
            "skill_ids": ["build-api", "validate-api"],
            "edges": [{"from_skill_id": "build-api", "to_skill_id": "validate-api"}],
        }
        progress = []

        result = generate_compositional_queries(
            [valid_candidate, {"candidate_type": "pair"}],
            [
                contract("build-api", "hash-build", "Generate an OpenAPI specification."),
                contract("validate-api", "hash-validate", "Validate an OpenAPI specification."),
            ],
            FakeClient(
                """{
                  "query": "Prepare a complete API contract for an implementation team, then validate endpoint definitions, request and response schemas, authentication behavior, error handling, versioning notes, and acceptance criteria before publishing a final specification and validation report for release approval.",
                  "positive_skill_ids": ["build-api", "validate-api"],
                  "subtasks": [
                    {"step_id": "s1", "skill_id": "build-api", "instruction": "Draft the specification."},
                    {"step_id": "s2", "skill_id": "validate-api", "instruction": "Validate the specification."}
                  ],
                  "dependencies": [{"from_step_id": "s1", "to_step_id": "s2"}]
                }"""
            ),
            CompositionalGenerationConfig(model="deepseek-v4-flash"),
            progress_callback=progress.append,
        )

        self.assertEqual(len(result.queries), 1)
        self.assertEqual(len(result.failures), 1)
        self.assertEqual(
            [(event.completed, event.queries, event.failures, event.review_queue) for event in progress],
            [(1, 1, 0, 0), (2, 1, 1, 0)],
        )

    def test_generates_candidates_with_bounded_parallel_requests(self) -> None:
        candidates = [
            {
                "candidate_id": f"comp::pair::parallel-{index}",
                "candidate_type": "pair",
                "skill_ids": ["build-api", "validate-api"],
                "edges": [{"from_skill_id": "build-api", "to_skill_id": "validate-api"}],
            }
            for index in range(2)
        ]
        client = ConcurrentClient(
            """{
              "query": "Prepare a complete API contract for an implementation team, then validate endpoint definitions, request and response schemas, authentication behavior, error handling, versioning notes, and acceptance criteria before publishing a final specification and validation report for release approval.",
              "positive_skill_ids": ["build-api", "validate-api"],
              "subtasks": [
                {"step_id": "s1", "skill_id": "build-api", "instruction": "Draft the specification."},
                {"step_id": "s2", "skill_id": "validate-api", "instruction": "Validate the specification."}
              ],
              "dependencies": [{"from_step_id": "s1", "to_step_id": "s2"}]
            }"""
        )

        result = generate_compositional_queries(
            candidates,
            [
                contract("build-api", "hash-build", "Generate an OpenAPI specification."),
                contract("validate-api", "hash-validate", "Validate an OpenAPI specification."),
            ],
            client,
            CompositionalGenerationConfig(
                model="deepseek-v4-flash", max_attempts=1, concurrency=2
            ),
        )

        self.assertEqual(len(result.queries), 2)
        self.assertEqual(client.max_active, 2)

    def test_network_error_is_recorded_per_candidate(self) -> None:
        class FailingClient:
            def complete(self, messages, *, temperature: float, max_new_tokens: int) -> str:
                raise RuntimeError("temporary network failure")

        candidate = {
            "candidate_id": "comp::pair::network",
            "candidate_type": "pair",
            "skill_ids": ["build-api", "validate-api"],
            "edges": [{"from_skill_id": "build-api", "to_skill_id": "validate-api"}],
        }
        result = generate_compositional_queries(
            [candidate],
            [
                contract("build-api", "hash-build", "Generate an OpenAPI specification."),
                contract("validate-api", "hash-validate", "Validate an OpenAPI specification."),
            ],
            FailingClient(),
            CompositionalGenerationConfig(model="deepseek-v4-flash", max_attempts=1),
        )

        self.assertEqual(result.queries, [])
        self.assertIn("temporary network failure", result.failures[0]["error"])

    def test_retry_prompt_includes_word_limit_and_validation_error(self) -> None:
        candidate = {
            "candidate_id": "comp::pair::03",
            "candidate_type": "pair",
            "skill_ids": ["build-api", "validate-api"],
            "edges": [{"from_skill_id": "build-api", "to_skill_id": "validate-api"}],
        }
        client = SequencedClient(
            [
                """{
                  "query": "Draft and validate a deployable API specification for release approval.",
                  "positive_skill_ids": ["build-api", "validate-api"],
                  "subtasks": [
                    {"step_id": "s1", "skill_id": "build-api", "instruction": "Draft the specification."},
                    {"step_id": "s2", "skill_id": "validate-api", "instruction": "Validate the specification."}
                  ],
                  "dependencies": [{"from_step_id": "s1", "to_step_id": "s2"}]
                }""",
                """{
                  "query": "Prepare a complete API contract for an implementation team, then validate endpoint definitions, request and response schemas, authentication behavior, error handling, versioning notes, and acceptance criteria before publishing a final specification and validation report for release approval.",
                  "positive_skill_ids": ["build-api", "validate-api"],
                  "subtasks": [
                    {"step_id": "s1", "skill_id": "build-api", "instruction": "Draft the specification."},
                    {"step_id": "s2", "skill_id": "validate-api", "instruction": "Validate the specification."}
                  ],
                  "dependencies": [{"from_step_id": "s1", "to_step_id": "s2"}]
                }""",
            ]
        )

        result = generate_compositional_queries(
            [candidate],
            [
                contract("build-api", "hash-build", "Generate an OpenAPI specification."),
                contract("validate-api", "hash-validate", "Validate an OpenAPI specification."),
            ],
            client,
            CompositionalGenerationConfig(model="deepseek-v4-flash", max_attempts=2),
        )

        self.assertEqual(len(result.queries), 1)
        self.assertEqual(result.queries[0]["generator"]["attempts"], 2)
        self.assertIn("30 to 260 words", client.calls[0][1]["content"])
        self.assertIn("query word count", client.calls[1][-1]["content"])

def contract(skill_id: str, source_hash: str, summary: str) -> dict:
    return {
        "skill_id": skill_id,
        "source_hash": source_hash,
        "extraction": {"status": "validated"},
        "capability": {"summary": summary},
        "operations": [],
        "inputs": [],
        "outputs": [],
        "constraints": [],
        "quality_criteria": [],
    }


if __name__ == "__main__":
    unittest.main()

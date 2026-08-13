"""Validated local generation of multi-Skill synthetic queries."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Protocol


PROMPT_VERSION = "compositional_query_prompt_002"
SCHEMA_VERSION = "compositional_query_v1"


class CompletionClient(Protocol):
    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float,
        max_new_tokens: int,
    ) -> str: ...


@dataclass(frozen=True)
class CompositionalGenerationConfig:
    model: str
    temperature: float = 0.2
    max_new_tokens: int = 1024
    max_attempts: int = 2
    min_query_words: int = 30
    max_query_words: int = 260


@dataclass(frozen=True)
class CompositionalGenerationResult:
    queries: list[dict[str, Any]]
    failures: list[dict[str, Any]]
    review_queue: list[dict[str, Any]]


@dataclass(frozen=True)
class CompositionalGenerationProgress:
    candidate_id: str | None
    completed: int
    queries: int
    failures: int
    review_queue: int


class TransformersJsonClient:
    """Batched Qwen client for a single Slurm-allocated local GPU."""

    def __init__(self, model_name_or_path: str, device: str = "cuda") -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("local generation requires requirements.txt") from exc
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested but torch.cuda.is_available() is false")
        dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
        self._torch = torch
        self._device = device
        self._tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, local_files_only=True)
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._tokenizer.padding_side = "left"
        self._model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=dtype,
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
            local_files_only=True,
        ).to(device)
        self._model.eval()

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float,
        max_new_tokens: int,
    ) -> str:
        return self.complete_many(
            [messages],
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )[0]

    def complete_many(
        self,
        messages_batch: list[list[dict[str, str]]],
        *,
        temperature: float,
        max_new_tokens: int,
    ) -> list[str]:
        """Generate a response for every message list in one padded GPU batch."""
        if not messages_batch:
            return []
        prompts = [self._format_prompt(messages) for messages in messages_batch]
        return self._generate_prompts(
            prompts,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )

    def _format_prompt(self, messages: list[dict[str, str]]) -> str:
        try:
            return self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            return self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

    def _generate_prompts(
        self,
        prompts: list[str],
        *,
        temperature: float,
        max_new_tokens: int,
    ) -> list[str]:
        try:
            return self._generate_prompts_once(
                prompts,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
            )
        except self._torch.cuda.OutOfMemoryError:
            if len(prompts) == 1 or not self._device.startswith("cuda"):
                raise
            self._torch.cuda.empty_cache()
            midpoint = len(prompts) // 2
            return self._generate_prompts(
                prompts[:midpoint],
                temperature=temperature,
                max_new_tokens=max_new_tokens,
            ) + self._generate_prompts(
                prompts[midpoint:],
                temperature=temperature,
                max_new_tokens=max_new_tokens,
            )

    def _generate_prompts_once(
        self,
        prompts: list[str],
        *,
        temperature: float,
        max_new_tokens: int,
    ) -> list[str]:
        inputs = self._tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
        ).to(self._device)
        generation = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
            "pad_token_id": self._tokenizer.eos_token_id,
            "use_cache": True,
        }
        if temperature > 0:
            generation.update({"temperature": temperature, "top_p": 0.95})
        with self._torch.inference_mode():
            output = self._model.generate(**inputs, **generation)
        prompt_width = inputs["input_ids"].shape[1]
        return self._tokenizer.batch_decode(
            output[:, prompt_width:],
            skip_special_tokens=True,
        )


class VllmJsonClient:
    """Offline vLLM client with continuous batching on one allocated GPU."""

    def __init__(
        self,
        model_name_or_path: str,
        *,
        max_model_len: int = 16384,
        max_num_seqs: int = 32,
        gpu_memory_utilization: float = 0.9,
    ) -> None:
        try:
            from transformers import AutoTokenizer
            from vllm import LLM, SamplingParams
        except ImportError as exc:
            raise RuntimeError(
                "vLLM generation requires requirements-vllm.txt on the Linux server"
            ) from exc
        if max_model_len <= 0 or max_num_seqs <= 0:
            raise ValueError("vLLM length and concurrency limits must be positive")
        if not 0 < gpu_memory_utilization < 1:
            raise ValueError("gpu_memory_utilization must be between 0 and 1")
        self._sampling_params_class = SamplingParams
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            local_files_only=True,
        )
        self._engine = LLM(
            model=model_name_or_path,
            tokenizer=model_name_or_path,
            dtype="bfloat16",
            max_model_len=max_model_len,
            max_num_seqs=max_num_seqs,
            gpu_memory_utilization=gpu_memory_utilization,
            enable_prefix_caching=True,
            trust_remote_code=False,
        )

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float,
        max_new_tokens: int,
    ) -> str:
        return self.complete_many(
            [messages],
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )[0]

    def complete_many(
        self,
        messages_batch: list[list[dict[str, str]]],
        *,
        temperature: float,
        max_new_tokens: int,
    ) -> list[str]:
        if not messages_batch:
            return []
        prompts = [self._format_prompt(messages) for messages in messages_batch]
        sampling_params = self._sampling_params_class(
            temperature=temperature,
            top_p=0.95 if temperature > 0 else 1.0,
            max_tokens=max_new_tokens,
        )
        outputs = self._engine.generate(
            prompts,
            sampling_params=sampling_params,
            use_tqdm=True,
        )
        if len(outputs) != len(prompts):
            raise RuntimeError("vLLM returned the wrong number of responses")
        return [output.outputs[0].text for output in outputs]

    def _format_prompt(self, messages: list[dict[str, str]]) -> str:
        try:
            return self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            return self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )


def generate_compositional_queries(
    candidates: Iterable[dict[str, Any]],
    contracts: Iterable[dict[str, Any]],
    client: CompletionClient,
    config: CompositionalGenerationConfig,
    progress_callback: Callable[[CompositionalGenerationProgress], None] | None = None,
) -> CompositionalGenerationResult:
    """Generate only queries that preserve a verified candidate's composition."""
    _validate_config(config)
    contract_by_id = _validated_contracts(contracts)
    queries: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    review_queue: list[dict[str, Any]] = []

    def report_progress(candidate: dict[str, Any]) -> None:
        if progress_callback is None:
            return
        candidate_id = candidate.get("candidate_id")
        progress = CompositionalGenerationProgress(
            candidate_id=candidate_id if isinstance(candidate_id, str) else None,
            completed=len(queries) + len(failures),
            queries=len(queries),
            failures=len(failures),
            review_queue=len(review_queue),
        )
        try:
            progress_callback(progress)
        except Exception:
            pass

    for candidate in candidates:
        candidate_id = candidate.get("candidate_id")
        skill_ids = candidate.get("skill_ids")
        if not isinstance(candidate_id, str) or not candidate_id:
            failures.append(_failure(candidate, 0, "candidate_id is required"))
            report_progress(candidate)
            continue
        if not isinstance(skill_ids, list) or len(skill_ids) not in (2, 3) or not all(
            isinstance(skill_id, str) and skill_id for skill_id in skill_ids
        ):
            failures.append(_failure(candidate, 0, "candidate must contain two or three skill_ids"))
            report_progress(candidate)
            continue
        missing = [skill_id for skill_id in skill_ids if skill_id not in contract_by_id]
        if missing:
            failures.append(_failure(candidate, 0, f"missing validated contracts: {missing}"))
            report_progress(candidate)
            continue

        messages = build_compositional_messages(candidate, contract_by_id, config)
        last_error = "generation did not return a valid payload"
        for attempt in range(1, config.max_attempts + 1):
            attempt_messages = (
                messages if attempt == 1 else [*messages, _retry_message(last_error)]
            )
            try:
                payload = _parse_json_object(
                    client.complete(
                        attempt_messages,
                        temperature=config.temperature,
                        max_new_tokens=config.max_new_tokens,
                    )
                )
                record = _validate_payload(candidate, contract_by_id, payload, config, attempt)
            except (TypeError, ValueError) as exc:
                last_error = str(exc)
                continue
            queries.append(record)
            if attempt > 1 or candidate.get("candidate_type") == "triple":
                review_queue.append(
                    {
                        "query_id": record["query_id"],
                        "candidate_id": candidate_id,
                        "reason": "retried_generation" if attempt > 1 else "triple_candidate",
                    }
                )
            report_progress(candidate)
            break
        else:
            failures.append(_failure(candidate, config.max_attempts, last_error))
            report_progress(candidate)

    return CompositionalGenerationResult(queries, failures, review_queue)


def build_compositional_messages(
    candidate: dict[str, Any],
    contract_by_id: dict[str, dict[str, Any]],
    config: CompositionalGenerationConfig,
) -> list[dict[str, str]]:
    skill_ids = candidate["skill_ids"]
    edge_lines = []
    for edge in candidate.get("edges", []):
        if not isinstance(edge, dict):
            continue
        edge_lines.append(
            "{source} -> {target}; relation={relation}; artifacts={artifacts}".format(
                source=edge.get("from_skill_id", ""),
                target=edge.get("to_skill_id", ""),
                relation=edge.get("operation_relation", ""),
                artifacts=", ".join(edge.get("matched_artifact_tokens", [])),
            )
        )
    contract_blocks = []
    for position, skill_id in enumerate(skill_ids, start=1):
        contract = contract_by_id[skill_id]
        contract_blocks.append(
            "Skill {position} ID: {skill_id}\n{contract}".format(
                position=position,
                skill_id=skill_id,
                contract=json.dumps(_prompt_contract(contract), ensure_ascii=False),
            )
        )
    return [
        {
            "role": "system",
            "content": (
                "You write realistic user requests requiring every supplied Skill in order. "
                "Return one JSON object only, without markdown or commentary."
            ),
        },
        {
            "role": "user",
            "content": "\n\n".join(
                [
                    "Write one natural English task for the validated multi-Skill candidate below.",
                    "The task must require all Skills, preserve the stated handoffs, and never name a Skill ID.",
                    f"The query must contain {config.min_query_words} to {config.max_query_words} words.",
                    "Use this exact JSON schema:",
                    json.dumps(
                        {
                            "query": "string",
                            "positive_skill_ids": skill_ids,
                            "subtasks": [
                                {
                                    "step_id": "s1",
                                    "skill_id": skill_ids[0],
                                    "instruction": "string",
                                }
                            ],
                            "dependencies": [{"from_step_id": "s1", "to_step_id": "s2"}],
                        },
                        ensure_ascii=False,
                    ),
                    "Skill order: " + " -> ".join(skill_ids),
                    "Validated handoffs:\n" + "\n".join(edge_lines),
                    "Contracts:\n" + "\n\n".join(contract_blocks),
                ]
            ),
        },
    ]


def _retry_message(error: str) -> dict[str, str]:
    return {
        "role": "user",
        "content": (
            "The previous JSON was rejected by the deterministic validator: "
            f"{error}. Regenerate the entire JSON object and correct that error."
        ),
    }
def _validated_contracts(contracts: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result = {}
    for contract in contracts:
        skill_id = contract.get("skill_id")
        if (
            isinstance(skill_id, str)
            and skill_id
            and contract.get("extraction", {}).get("status") == "validated"
            and skill_id not in result
        ):
            result[skill_id] = contract
    return result


def _prompt_contract(contract: dict[str, Any]) -> dict[str, Any]:
    return {
        "capability": contract.get("capability", {}),
        "operations": contract.get("operations", []),
        "inputs": contract.get("inputs", []),
        "outputs": contract.get("outputs", []),
        "constraints": contract.get("constraints", []),
        "quality_criteria": contract.get("quality_criteria", []),
    }


def _parse_json_object(response: str) -> dict[str, Any]:
    if not isinstance(response, str) or not response.strip():
        raise ValueError("model returned an empty response")
    text = response.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else ""
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
    payload = json.loads(text.strip())
    if not isinstance(payload, dict):
        raise ValueError("model response must be a JSON object")
    return payload


def _validate_payload(
    candidate: dict[str, Any],
    contract_by_id: dict[str, dict[str, Any]],
    payload: dict[str, Any],
    config: CompositionalGenerationConfig,
    attempts: int,
) -> dict[str, Any]:
    query = payload.get("query")
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    query = query.strip()
    word_count = len(query.split())
    if not config.min_query_words <= word_count <= config.max_query_words:
        raise ValueError(
            f"query word count must be {config.min_query_words}..{config.max_query_words}, got {word_count}"
        )
    skill_ids = candidate["skill_ids"]
    if payload.get("positive_skill_ids") != skill_ids:
        raise ValueError("positive_skill_ids must exactly equal the candidate skill_ids")
    if any(skill_id.casefold() in query.casefold() for skill_id in skill_ids):
        raise ValueError("query must not reveal a Skill ID")

    subtasks = payload.get("subtasks")
    if not isinstance(subtasks, list) or len(subtasks) != len(skill_ids):
        raise ValueError("subtasks must contain exactly one item for each Skill")
    step_ids = []
    actual_skill_ids = []
    normalized_subtasks = []
    for subtask in subtasks:
        if not isinstance(subtask, dict):
            raise ValueError("each subtask must be a JSON object")
        step_id = subtask.get("step_id")
        skill_id = subtask.get("skill_id")
        instruction = subtask.get("instruction")
        if not isinstance(step_id, str) or not step_id.strip():
            raise ValueError("each subtask needs a step_id")
        if not isinstance(skill_id, str) or not skill_id:
            raise ValueError("each subtask needs a skill_id")
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("each subtask needs an instruction")
        step_ids.append(step_id)
        actual_skill_ids.append(skill_id)
        normalized_subtasks.append(
            {"step_id": step_id, "skill_id": skill_id, "instruction": instruction.strip()}
        )
    if len(set(step_ids)) != len(step_ids):
        raise ValueError("subtask step_ids must be unique")
    if actual_skill_ids != skill_ids:
        raise ValueError("subtask skill_ids must exactly follow the candidate order")

    dependencies = _validate_dependencies(candidate, payload.get("dependencies"), step_ids, skill_ids)
    source_hashes = [str(contract_by_id[skill_id].get("source_hash", "")) for skill_id in skill_ids]
    if not all(source_hashes):
        raise ValueError("each selected contract must contain a source_hash")
    return {
        "schema_version": SCHEMA_VERSION,
        "query_id": "compq::" + candidate["candidate_id"],
        "candidate_id": candidate["candidate_id"],
        "candidate_type": candidate.get("candidate_type"),
        "query": query,
        "positive_skill_ids": skill_ids,
        "source_hashes": source_hashes,
        "subtasks": normalized_subtasks,
        "dependencies": dependencies,
        "generator": {
            "provider": "local_transformers",
            "model": config.model,
            "prompt_version": PROMPT_VERSION,
            "temperature": config.temperature,
            "attempts": attempts,
        },
    }


def _validate_dependencies(
    candidate: dict[str, Any],
    dependencies: Any,
    step_ids: list[str],
    skill_ids: list[str],
) -> list[dict[str, str]]:
    if not isinstance(dependencies, list):
        raise ValueError("dependencies must be a list")
    step_by_skill = dict(zip(skill_ids, step_ids))
    expected = {
        (step_by_skill[edge["from_skill_id"]], step_by_skill[edge["to_skill_id"]])
        for edge in candidate.get("edges", [])
        if isinstance(edge, dict)
        and edge.get("from_skill_id") in step_by_skill
        and edge.get("to_skill_id") in step_by_skill
    }
    actual = set()
    normalized = []
    for dependency in dependencies:
        if not isinstance(dependency, dict):
            raise ValueError("each dependency must be a JSON object")
        source = dependency.get("from_step_id")
        target = dependency.get("to_step_id")
        if source not in step_ids or target not in step_ids or source == target:
            raise ValueError("dependencies must reference distinct declared subtask steps")
        actual.add((source, target))
        normalized.append({"from_step_id": source, "to_step_id": target})
    if actual != expected:
        raise ValueError("dependencies must exactly represent the candidate handoff edges")
    if len(actual) != len(normalized) or _has_cycle(actual, step_ids):
        raise ValueError("dependencies must be unique and acyclic")
    return normalized


def _has_cycle(edges: set[tuple[str, str]], nodes: list[str]) -> bool:
    children = {node: set() for node in nodes}
    indegree = {node: 0 for node in nodes}
    for source, target in edges:
        children[source].add(target)
        indegree[target] += 1
    ready = sorted(node for node, degree in indegree.items() if degree == 0)
    visited = 0
    while ready:
        node = ready.pop()
        visited += 1
        for target in sorted(children[node]):
            indegree[target] -= 1
            if indegree[target] == 0:
                ready.append(target)
    return visited != len(nodes)


def _failure(candidate: dict[str, Any], attempts: int, error: str) -> dict[str, Any]:
    return {
        "candidate_id": candidate.get("candidate_id"),
        "skill_ids": candidate.get("skill_ids", []),
        "stage": "compositional_query_generation",
        "attempts": attempts,
        "error": error,
    }


def _validate_config(config: CompositionalGenerationConfig) -> None:
    if config.max_attempts <= 0:
        raise ValueError("max_attempts must be positive")
    if config.max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    if not 0 <= config.temperature <= 2:
        raise ValueError("temperature must be between 0 and 2")
    if config.min_query_words <= 0 or config.max_query_words < config.min_query_words:
        raise ValueError("query word bounds are invalid")

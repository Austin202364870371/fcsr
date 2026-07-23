"""SkillRouter-compatible text formatting and optional model helpers."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Callable


QUERY_INSTRUCTION = (
    "Instruct: Given a coding task description, retrieve the most relevant "
    "skill document that would help an agent complete the task\nQuery:"
)
RERANK_INSTRUCTION = (
    "Given a coding task description, judge whether the skill document "
    "is relevant and useful for completing the task"
)


def format_query(raw_query: str, max_len: int = 1500) -> str:
    if not isinstance(raw_query, str):
        raise TypeError("raw_query must be a string")
    return f"{QUERY_INSTRUCTION}{raw_query[:max_len]}"


def format_skill(
    skill: Mapping[str, Any],
    desc_max: int = 300,
    body_max: int = 2500,
) -> str:
    name = _text(skill.get("name"))
    description = _text(skill.get("description"))[:desc_max]
    body = _text(skill.get("body"))[:body_max]
    return f"{name} | {description} | {body}"


def format_rerank_prompt(
    query_text: str,
    skill: Mapping[str, Any],
    prompt_format: str = "flat-full",
    desc_max: int = 500,
    body_max: int = 2000,
) -> str:
    name = _text(skill.get("name"))
    description = _text(skill.get("description"))[:desc_max]
    body = _text(skill.get("body"))[:body_max]

    if prompt_format == "flat-full":
        document = f"{name} | {description} | {body}"
    elif prompt_format == "flat-nd":
        document = f"{name} | {description}"
    elif prompt_format == "struct":
        return (
            f"<Instruct>: {RERANK_INSTRUCTION}\n\n"
            f"<Query>: {query_text}\n\n"
            f"<Skill>:\n<Name>: {name}\n<Description>: {description}\n<Body>: {body}"
        )
    else:
        raise ValueError(f"unknown prompt_format: {prompt_format}")

    return (
        f"<Instruct>: {RERANK_INSTRUCTION}\n\n"
        f"<Query>: {query_text}\n\n"
        f"<Document>: {document}"
    )


def _text(value: Any) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise TypeError("skill text fields must be strings")
    return value


def last_token_pool(last_hidden_states: Any, attention_mask: Any) -> Any:
    """Pool the last non-padding token exactly as SkillRouter does."""
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if bool(left_padding):
        return last_hidden_states[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]
    return last_hidden_states[range(batch_size), sequence_lengths]


def load_embedding_model(
    name_or_path: str,
    device: str = "cuda",
) -> tuple[Any, Any]:
    """Load a base Qwen encoder or an FCSR LoRA checkpoint."""
    try:
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "transformers is required for model loading; install requirements-train.txt"
        ) from exc
    model_path = str(name_or_path)
    adapter_config = Path(model_path) / "adapter_config.json"
    base_path = model_path
    peft_model = None
    if adapter_config.exists():
        try:
            from peft import PeftConfig, PeftModel
        except ImportError as exc:
            raise RuntimeError("peft is required to load a LoRA checkpoint") from exc
        peft_config = PeftConfig.from_pretrained(model_path)
        base_path = peft_config.base_model_name_or_path
        peft_model = PeftModel
    tokenizer_path = (
        model_path
        if (Path(model_path) / "tokenizer_config.json").exists()
        else base_path
    )
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        padding_side="left",
        trust_remote_code=True,
    )
    model = AutoModel.from_pretrained(
        base_path,
        torch_dtype="auto",
        trust_remote_code=True,
    )
    if peft_model is not None:
        model = peft_model.from_pretrained(model, model_path)
    model.to(device)
    model.eval()
    return model, tokenizer

def encode_texts(
    model: Any,
    tokenizer: Any,
    texts: list[str],
    max_length: int,
    batch_size: int,
    device: str,
    progress: Callable[[int], None] | None = None,
) -> Any:
    """Encode text batches into normalized NumPy vectors."""
    try:
        import torch
        import torch.nn.functional as functional
    except ImportError as exc:
        raise RuntimeError(
            "torch is required for encoding; install requirements-train.txt"
        ) from exc
    if max_length <= 0 or batch_size <= 0:
        raise ValueError("max_length and batch_size must be positive")
    batches = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start : start + batch_size]
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            outputs = model(**encoded)
            hidden_states = getattr(outputs, "last_hidden_state", outputs[0])
            pooled = last_token_pool(hidden_states, encoded["attention_mask"])
            batches.append(functional.normalize(pooled, p=2, dim=1).float().cpu())
            if progress is not None:
                progress(len(batch_texts))
    if not batches:
        return torch.empty((0, 0), dtype=torch.float32).numpy()
    return torch.cat(batches, dim=0).numpy()

def build_biencoder_examples(
    records: Any,
    skills: Any,
    max_negatives: int = 10,
) -> list[dict[str, Any]]:
    if max_negatives < 0:
        raise ValueError("max_negatives must not be negative")
    skill_lookup = {
        skill["skill_id"]: skill
        for skill in skills
        if isinstance(skill.get("skill_id"), str) and skill["skill_id"]
    }
    examples = []
    for record in records:
        positive_id = record.get("positive_skill_id")
        positive = skill_lookup.get(positive_id)
        if positive is None:
            raise ValueError(f"positive skill not found: {positive_id!r}")
        negative_ids = []
        negative_sources = []
        seen = {positive_id}
        for candidate in record.get("negative_candidates", []):
            skill_id = candidate.get("skill_id")
            if skill_id in seen:
                continue
            if skill_id not in skill_lookup:
                raise ValueError(f"negative skill not found: {skill_id!r}")
            seen.add(skill_id)
            negative_ids.append(skill_id)
            negative_sources.append(candidate.get("source", "unknown"))
            if len(negative_ids) >= max_negatives:
                break
        examples.append(
            {
                "query_id": record.get("query_id"),
                "query_text": format_query(str(record.get("query", ""))),
                "positive_skill_id": positive_id,
                "positive_text": format_skill(positive),
                "negative_skill_ids": negative_ids,
                "negative_texts": [
                    format_skill(skill_lookup[skill_id]) for skill_id in negative_ids
                ],
                "negative_sources": negative_sources,
            }
        )
    return examples


def info_nce_loss(
    query_embeddings: Any,
    document_embeddings: Any,
    positive_indices: Any,
    temperature: float,
) -> Any:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    try:
        import torch.nn.functional as functional
    except ImportError as exc:
        raise RuntimeError(
            "torch is required for InfoNCE; install requirements-train.txt"
        ) from exc
    if query_embeddings.ndim != 2 or document_embeddings.ndim != 2:
        raise ValueError("embeddings must be 2D tensors")
    if query_embeddings.shape[1] != document_embeddings.shape[1]:
        raise ValueError("query and document dimensions must match")
    if positive_indices.ndim != 1 or positive_indices.shape[0] != query_embeddings.shape[0]:
        raise ValueError("positive_indices must contain one index per query")
    if positive_indices.numel() and (
        positive_indices.min().item() < 0
        or positive_indices.max().item() >= document_embeddings.shape[0]
    ):
        raise ValueError("positive_indices contains an out-of-range index")
    queries = functional.normalize(query_embeddings, p=2, dim=1)
    documents = functional.normalize(document_embeddings, p=2, dim=1)
    logits = queries @ documents.transpose(0, 1) / temperature
    return functional.cross_entropy(logits, positive_indices.long())

from dataclasses import dataclass


@dataclass(frozen=True)
class RerankerGroupResult:
    groups: list[dict[str, Any]]
    total_records: int
    dropped_no_positive: int


def build_reranker_groups(
    records: Any,
    skills: Any,
    top_k: int = 20,
    progress: Callable[[int], None] | None = None,
) -> RerankerGroupResult:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    lookup = {
        skill["skill_id"]: skill
        for skill in skills
        if isinstance(skill.get("skill_id"), str) and skill["skill_id"]
    }
    groups = []
    total = 0
    dropped = 0
    for record in records:
        total += 1
        positive_ids = set(record.get("positive_skill_ids", []))
        positive_id = record.get("positive_skill_id")
        if isinstance(positive_id, str) and positive_id:
            positive_ids.add(positive_id)
        candidates = record.get("retrieved_candidates")
        if not isinstance(candidates, list):
            negatives = record.get("negative_candidates", [])
            candidates = [{"skill_id": positive_id, "score": 1.0}, *negatives]

        group_candidates = []
        positive_mask = []
        seen = set()
        for candidate in candidates:
            skill_id = candidate.get("skill_id")
            if skill_id in seen or skill_id not in lookup:
                continue
            seen.add(skill_id)
            rank = len(group_candidates) + 1
            is_positive = skill_id in positive_ids
            group_candidates.append(
                {
                    "skill_id": skill_id,
                    "rank": rank,
                    "retrieval_score": float(candidate.get("score", 0.0)),
                    "label": 1 if is_positive else 0,
                    "prompt": format_rerank_prompt(
                        str(record.get("query", "")),
                        lookup[skill_id],
                    ),
                }
            )
            positive_mask.append(is_positive)
            if len(group_candidates) >= top_k:
                break
        if not any(positive_mask):
            dropped += 1
            if progress is not None:
                progress(1)
            continue
        groups.append(
            {
                "query_id": record.get("query_id"),
                "query": record.get("query"),
                "candidates": group_candidates,
                "positive_mask": positive_mask,
            }
        )
        if progress is not None:
            progress(1)
    return RerankerGroupResult(
        groups=groups,
        total_records=total,
        dropped_no_positive=dropped,
    )


def listwise_cross_entropy(scores: Any, positive_mask: Any) -> Any:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "torch is required for listwise loss; install requirements-train.txt"
        ) from exc
    if scores.shape != positive_mask.shape:
        raise ValueError("scores and positive_mask must have the same shape")
    if scores.ndim not in (1, 2):
        raise ValueError("scores must be a 1D group or a 2D batch of groups")
    mask = positive_mask.bool()
    if scores.ndim == 1:
        if not bool(mask.any()):
            raise ValueError("each listwise group must contain a positive")
        return torch.logsumexp(scores, dim=0) - torch.logsumexp(scores[mask], dim=0)
    if not bool(mask.any(dim=1).all()):
        raise ValueError("each listwise group must contain a positive")
    positive_scores = scores.masked_fill(~mask, float("-inf"))
    return (
        torch.logsumexp(scores, dim=1)
        - torch.logsumexp(positive_scores, dim=1)
    ).mean()

def get_reranker_template_tokens(tokenizer: Any) -> tuple[list[int], list[int]]:
    prefix = (
        '<|im_start|>system\nJudge whether the Document meets the requirements '
        'based on the Query and the Instruct provided. Note that the answer can '
        'only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
    )
    suffix = '<|im_end|>\n<|im_start|>assistant\n\n\n\n\n'
    return (
        tokenizer.encode(prefix, add_special_tokens=False),
        tokenizer.encode(suffix, add_special_tokens=False),
    )


def tokenize_reranker_text(
    text: str,
    tokenizer: Any,
    prefix_tokens: list[int],
    suffix_tokens: list[int],
    max_length: int,
) -> list[int]:
    content_length = max_length - len(prefix_tokens) - len(suffix_tokens)
    if content_length <= 0:
        raise ValueError("max_length is too small for the reranker template")
    inputs = tokenizer(
        text,
        padding=False,
        truncation=True,
        max_length=content_length,
        return_attention_mask=False,
    )
    return prefix_tokens + inputs["input_ids"] + suffix_tokens
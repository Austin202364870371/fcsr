"""Train the FCSR listwise reranker."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable

from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data_io import load_jsonl, stream_jsonl
from modeling import (
    get_reranker_template_tokens,
    listwise_cross_entropy,
    materialize_reranker_groups,
    tokenize_reranker_text,
)


def create_training_progress(
    epoch: int,
    total_groups: int,
    progress_factory: Any | None = None,
) -> Any:
    if progress_factory is None:
        progress_factory = tqdm
    return progress_factory(
        total=total_groups,
        desc=f"Reranker epoch {epoch}",
        unit="group",
        dynamic_ncols=True,
    )


def enable_checkpoint_input_gradients(model: Any) -> None:
    enable = getattr(model, "enable_input_require_grads", None)
    if not callable(enable):
        raise RuntimeError(
            "the base model does not support input gradients required by "
            "LoRA with gradient checkpointing"
        )
    enable()


def longest_group_index(
    groups: list[dict[str, Any]],
    prompt_length: Callable[[str], int],
    progress: Callable[[int], None] | None = None,
) -> tuple[int, int]:
    """Find the group with the largest within-group padded sequence length."""
    if not groups:
        raise ValueError("reranker training data contains no groups")
    best_index = -1
    best_length = -1
    for index, group in enumerate(groups):
        candidates = group.get("candidates", [])
        if not candidates:
            raise ValueError(f"group {index} contains no candidates")
        group_length = max(prompt_length(str(item.get("prompt", ""))) for item in candidates)
        if group_length > best_length:
            best_index = index
            best_length = group_length
        if progress is not None:
            progress(1)
    return best_index, best_length


def _group_loss(
    model: Any, group: dict[str, Any], tokenizer: Any, prefix_tokens: list[int],
    suffix_tokens: list[int], settings: dict[str, Any], yes_id: int, no_id: int,
    device: str, use_bf16: bool, torch: Any,
) -> Any:
    tokenized = [
        tokenize_reranker_text(str(candidate.get("prompt", "")), tokenizer, prefix_tokens,
                                suffix_tokens, settings["max_length"])
        for candidate in group["candidates"]
    ]
    group_max_length = max(len(ids) for ids in tokenized)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    padded = [[pad_id] * (group_max_length - len(ids)) + ids for ids in tokenized]
    masks = [[0] * (group_max_length - len(ids)) + [1] * len(ids) for ids in tokenized]
    input_ids = torch.tensor(padded, dtype=torch.long, device=device)
    attention_mask = torch.tensor(masks, dtype=torch.long, device=device)
    with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=use_bf16):
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[:, -1, :]
        scores = logits[:, yes_id] - logits[:, no_id]
        return listwise_cross_entropy(scores, torch.tensor(group["positive_mask"], device=device))


def _run_memory_preflight(
    model: Any, groups: list[dict[str, Any]], tokenizer: Any, prefix_tokens: list[int],
    suffix_tokens: list[int], settings: dict[str, Any], yes_id: int, no_id: int,
    device: str, use_bf16: bool, optimizer: Any, torch: Any,
) -> None:
    def prompt_length(prompt: str) -> int:
        return len(tokenize_reranker_text(prompt, tokenizer, prefix_tokens, suffix_tokens,
                                           settings["max_length"]))

    with tqdm(total=len(groups), desc="Reranker preflight: scanning groups", unit="group",
              dynamic_ncols=True) as progress:
        group_index, padded_tokens = longest_group_index(groups, prompt_length, progress.update)
    python_rng_state = random.getstate()
    torch_rng_state = torch.get_rng_state()
    cuda_rng_state_all = torch.cuda.get_rng_state_all() if device == "cuda" else None
    try:
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        loss = _group_loss(model, groups[group_index], tokenizer, prefix_tokens, suffix_tokens,
                           settings, yes_id, no_id, device, use_bf16, torch)
        loss.backward()
        if device == "cuda":
            torch.cuda.synchronize()
            peak_memory_mib = round(torch.cuda.max_memory_allocated() / (1024 * 1024))
        else:
            peak_memory_mib = 0
    except torch.OutOfMemoryError as exc:
        raise RuntimeError(
            "memory preflight failed on the longest reranker group "
            f"(index={group_index}, padded_tokens={padded_tokens}). "
            "Reduce --max-length before starting the full epoch."
        ) from exc
    finally:
        optimizer.zero_grad(set_to_none=True)
        random.setstate(python_rng_state)
        torch.set_rng_state(torch_rng_state)
        if device == "cuda" and cuda_rng_state_all is not None:
            torch.cuda.set_rng_state_all(cuda_rng_state_all)
            torch.cuda.empty_cache()
    print("Reranker memory preflight:", json.dumps({
        "group_index": group_index,
        "padded_tokens": padded_tokens,
        "peak_memory_mib": peak_memory_mib,
    }))

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the FCSR Qwen reranker")
    parser.add_argument("--config", default="configs/fcsr.yaml")
    parser.add_argument("--groups", default="data/training/reranker.jsonl.gz")
    parser.add_argument("--skills", default="data/raw/skills_easy.jsonl.gz")
    parser.add_argument("--output-dir")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def load_config(path: str | Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("pyyaml is required to read the training config") from exc
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError("training config must be a YAML object")
    return config


def resolve_settings(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(args.config)
    model = config.get("model", {})
    training = config.get("training", {})
    reranker = training.get("reranker", {})
    lora = training.get("lora", {})
    paths = config.get("paths", {})
    return {
        "model": model.get("reranker", "models/Qwen3-Reranker-0.6B"),
        "output_dir": args.output_dir
        or paths.get("reranker_checkpoint")
        or "checkpoints/fcsr/reranker",
        "epochs": reranker.get("epochs", 1),
        "gradient_accumulation_steps": reranker.get(
            "gradient_accumulation_steps", 16
        ),
        "learning_rate": reranker.get("learning_rate", 1e-5),
        "max_length": reranker.get("max_length", 4096),
        "precision": training.get("precision", "bf16"),
        "gradient_checkpointing": training.get("gradient_checkpointing", True),
        "lora_r": lora.get("r", 8),
        "lora_alpha": lora.get("alpha", 16),
        "lora_dropout": lora.get("dropout", 0.05),
    }


def summarize_groups(
    groups: list[dict[str, Any]],
    settings: dict[str, Any],
) -> dict[str, Any]:
    candidate_counts = [len(group.get("candidates", [])) for group in groups]
    positive_counts = [sum(group.get("positive_mask", [])) for group in groups]
    invalid = sum(
        not candidates or positives <= 0
        for candidates, positives in zip(candidate_counts, positive_counts)
    )
    if invalid:
        raise ValueError(f"{invalid} groups are empty or contain no positive")
    type_counts = Counter(group.get("training_type", "single_skill") for group in groups)
    weighted_type_mass: Counter[str] = Counter()
    for group in groups:
        weight = float(group.get("loss_weight", 1.0))
        if weight <= 0:
            raise ValueError("reranker loss weights must be positive")
        weighted_type_mass[group.get("training_type", "single_skill")] += weight
    return {
        "groups": len(groups),
        "training_types": dict(sorted(type_counts.items())),
        "weighted_type_mass": dict(sorted(weighted_type_mass.items())),
        "mean_candidates": (
            sum(candidate_counts) / len(candidate_counts) if candidate_counts else 0.0
        ),
        "mean_positives": (
            sum(positive_counts) / len(positive_counts) if positive_counts else 0.0
        ),
        "settings": settings,
    }


def train(
    groups: list[dict[str, Any]],
    settings: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    try:
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            get_linear_schedule_with_warmup,
        )
    except ImportError as exc:
        raise RuntimeError(
            "training dependencies are missing; install requirements.txt"
        ) from exc
    if not groups:
        raise ValueError("reranker training data contains no groups")

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(
        settings["model"],
        padding_side="left",
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        settings["model"],
        torch_dtype="auto",
        trust_remote_code=True,
    )
    if settings["gradient_checkpointing"]:
        enable_checkpoint_input_gradients(model)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    yes_id = tokenizer.convert_tokens_to_ids("yes")
    no_id = tokenizer.convert_tokens_to_ids("no")
    prefix_tokens, suffix_tokens = get_reranker_template_tokens(tokenizer)

    model = get_peft_model(
        model,
        LoraConfig(
            task_type="CAUSAL_LM",
            r=settings["lora_r"],
            lora_alpha=settings["lora_alpha"],
            lora_dropout=settings["lora_dropout"],
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        ),
    )
    if settings["gradient_checkpointing"]:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=settings["learning_rate"],
    )
    accumulation = settings["gradient_accumulation_steps"]
    optimizer_steps = math.ceil(len(groups) / accumulation) * settings["epochs"]
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(optimizer_steps * 0.05)),
        num_training_steps=optimizer_steps,
    )
    use_bf16 = (
        settings["precision"] == "bf16"
        and device == "cuda"
        and torch.cuda.is_bf16_supported()
    )
    rng = random.Random(seed)
    history = []
    optimizer.zero_grad(set_to_none=True)
    _run_memory_preflight(
        model, groups, tokenizer, prefix_tokens, suffix_tokens, settings, yes_id, no_id,
        device, use_bf16, optimizer, torch,
    )

    for epoch in range(settings["epochs"]):
        order = list(range(len(groups)))
        rng.shuffle(order)
        accumulation_count = 0
        progress = create_training_progress(
            epoch=epoch + 1,
            total_groups=len(order),
        )
        for position, index in enumerate(order):
            group = groups[index]
            tokenized = [
                tokenize_reranker_text(
                    candidate["prompt"],
                    tokenizer,
                    prefix_tokens,
                    suffix_tokens,
                    settings["max_length"],
                )
                for candidate in group["candidates"]
            ]
            max_length = max(len(ids) for ids in tokenized)
            pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
            padded = [[pad_id] * (max_length - len(ids)) + ids for ids in tokenized]
            masks = [[0] * (max_length - len(ids)) + [1] * len(ids) for ids in tokenized]
            input_ids = torch.tensor(padded, dtype=torch.long, device=device)
            attention_mask = torch.tensor(masks, dtype=torch.long, device=device)
            with torch.autocast(
                device_type=device,
                dtype=torch.bfloat16,
                enabled=use_bf16,
            ):
                logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[:, -1, :]
                scores = logits[:, yes_id] - logits[:, no_id]
                loss = listwise_cross_entropy(
                    scores,
                    torch.tensor(group["positive_mask"], device=device),
                )
                loss_weight = float(group.get("loss_weight", 1.0))
                if loss_weight <= 0:
                    raise ValueError("reranker loss weights must be positive")
                loss = loss * loss_weight
            (loss / accumulation).backward()
            loss_value = float(loss.detach().cpu())
            accumulation_count += 1
            history.append(
                {
                    "epoch": epoch + 1,
                    "group": position + 1,
                    "loss": loss_value,
                }
            )
            progress.update(1)
            progress.set_postfix(loss=f"{loss_value:.4f}")
            is_last = position + 1 == len(order)
            if accumulation_count >= accumulation or is_last:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                accumulation_count = 0
        progress.close()

    output_dir = Path(settings["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    summary = {
        **summarize_groups(groups, settings),
        "optimizer_steps": optimizer_steps,
        "loss_history": history,
    }
    (output_dir / "training_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def run_train(args: argparse.Namespace) -> dict[str, Any]:
    settings = resolve_settings(args)
    groups = load_jsonl(args.groups)
    groups = materialize_reranker_groups(groups, stream_jsonl(args.skills))
    if args.dry_run:
        return summarize_groups(groups, settings)
    return train(groups, settings, args.seed)


def main() -> None:
    args = build_parser().parse_args()
    result = run_train(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

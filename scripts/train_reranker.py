"""Prepare Top-20 groups and train the FCSR listwise reranker."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data_io import load_jsonl, stream_jsonl, write_jsonl_atomic
from modeling import (
    build_reranker_groups,
    get_reranker_template_tokens,
    listwise_cross_entropy,
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="FCSR listwise reranker pipeline")
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare", help="build ordered Top-20 groups")
    prepare.add_argument("--retrieval", default="data/synthetic/train_biencoder.jsonl")
    prepare.add_argument("--skills", default="data/raw/skills_easy.jsonl")
    prepare.add_argument("--output", default="data/synthetic/train_reranker.jsonl")
    prepare.add_argument("--top-k", type=int, default=20)
    prepare.add_argument("--overwrite", action="store_true")

    train = commands.add_parser("train", help="train Qwen3-Reranker listwise")
    train.add_argument("--config", default="configs/model_qwen3_0_6b.yaml")
    train.add_argument("--groups", default="data/synthetic/train_reranker.jsonl")
    train.add_argument("--model")
    train.add_argument("--output-dir")
    train.add_argument("--method", choices=("lora", "full"))
    train.add_argument("--epochs", type=int)
    train.add_argument("--gradient-accumulation-steps", type=int)
    train.add_argument("--learning-rate", type=float)
    train.add_argument("--max-length", type=int)
    train.add_argument("--precision", choices=("bf16", "fp32"))
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--dry-run", action="store_true")
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


def run_prepare(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output)
    if output.exists() and not args.overwrite:
        raise FileExistsError("output exists; pass --overwrite to replace it")
    records = list(stream_jsonl(args.retrieval))
    with tqdm(
        total=len(records),
        desc="Reranker: building groups",
        unit="query",
        dynamic_ncols=True,
    ) as progress:
        result = build_reranker_groups(
            records,
            stream_jsonl(args.skills),
            top_k=args.top_k,
            progress=progress.update,
        )
    write_jsonl_atomic(output, result.groups)
    counts = [len(group["candidates"]) for group in result.groups]
    return {
        "output": str(output),
        "total_queries": result.total_records,
        "retained_groups": len(result.groups),
        "dropped_no_positive": result.dropped_no_positive,
        "mean_candidates": sum(counts) / len(counts) if counts else 0.0,
    }


def resolve_settings(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(args.config)
    model = config.get("model", {})
    training = config.get("training", {})
    paths = config.get("paths", {})
    return {
        "model": args.model
        or model.get("reranker_name_or_path")
        or "Qwen/Qwen3-Reranker-0.6B",
        "output_dir": args.output_dir
        or paths.get("reranker_checkpoint")
        or "checkpoints/fcsr-rank-0.6b",
        "method": args.method or training.get("method", "lora"),
        "epochs": args.epochs
        if args.epochs is not None
        else training.get("epochs_reranker", 1),
        "gradient_accumulation_steps": args.gradient_accumulation_steps
        if args.gradient_accumulation_steps is not None
        else training.get("gradient_accumulation_steps", 16),
        "learning_rate": args.learning_rate
        if args.learning_rate is not None
        else training.get("learning_rate_reranker", 1e-5),
        "max_length": args.max_length
        if args.max_length is not None
        else model.get("max_reranker_length", 4096),
        "precision": args.precision or training.get("precision", "bf16"),
        "gradient_checkpointing": training.get("gradient_checkpointing", True),
        "lora_r": training.get("lora_r", 8),
        "lora_alpha": training.get("lora_alpha", 16),
        "lora_dropout": training.get("lora_dropout", 0.05),
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
    return {
        "groups": len(groups),
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
            "training dependencies are missing; install requirements-train.txt"
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

    if settings["method"] == "lora":
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
    elif settings["method"] != "full":
        raise ValueError("method must be lora or full")
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
    if args.dry_run:
        return summarize_groups(groups, settings)
    return train(groups, settings, args.seed)


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "prepare":
        result = run_prepare(args)
    else:
        result = run_train(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

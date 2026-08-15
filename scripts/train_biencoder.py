"""Train the FCSR bi-encoder with SR-compatible InfoNCE."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data_io import stream_jsonl
from modeling import build_biencoder_examples, info_nce_loss, last_token_pool


def create_training_progress(
    epoch: int,
    total_batches: int,
    progress_factory: Any | None = None,
) -> Any:
    if progress_factory is None:
        from tqdm import tqdm

        progress_factory = tqdm
    return progress_factory(
        total=total_batches,
        desc=f"Bi-Encoder epoch {epoch}",
        unit="batch",
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
    parser = argparse.ArgumentParser(description="Train the FCSR Qwen bi-encoder")
    parser.add_argument("--config", default="configs/fcsr.yaml")
    parser.add_argument(
        "--train-data",
        default="data/training/biencoder.jsonl.gz",
    )
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
        value = yaml.safe_load(handle) or {}
    if not isinstance(value, dict):
        raise ValueError("training config must be a YAML object")
    return value


def resolve_settings(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(args.config)
    model = config.get("model", {})
    training = config.get("training", {})
    retriever = training.get("retriever", {})
    lora = training.get("lora", {})
    paths = config.get("paths", {})
    return {
        "model": model.get("retriever", "models/Qwen3-Embedding-0.6B"),
        "output_dir": args.output_dir
        or paths.get("retriever_checkpoint")
        or "checkpoints/fcsr/retriever",
        "epochs": retriever.get("epochs", 1),
        "micro_batch_size": retriever.get("micro_batch_size", 4),
        "gradient_accumulation_steps": retriever.get(
            "gradient_accumulation_steps", 4
        ),
        "learning_rate": retriever.get("learning_rate", 2e-5),
        "temperature": retriever.get("temperature", 0.05),
        "query_max_length": retriever.get("query_max_length", 512),
        "skill_max_length": retriever.get("skill_max_length", 2048),
        "precision": training.get("precision", "bf16"),
        "gradient_checkpointing": training.get("gradient_checkpointing", True),
        "lora_r": lora.get("r", 8),
        "lora_alpha": lora.get("alpha", 16),
        "lora_dropout": lora.get("dropout", 0.05),
    }


def load_examples(args: argparse.Namespace) -> list[dict[str, Any]]:
    records = list(stream_jsonl(args.train_data))
    skills = list(stream_jsonl(args.skills))
    return build_biencoder_examples(records, skills, max_negatives=10)


def summarize_examples(
    examples: list[dict[str, Any]],
    settings: dict[str, Any],
) -> dict[str, Any]:
    sources = Counter(
        source for example in examples for source in example["negative_sources"]
    )
    negatives = [len(example["negative_skill_ids"]) for example in examples]
    type_counts = Counter(example.get("training_type", "single_skill") for example in examples)
    weighted_type_mass: Counter[str] = Counter()
    for example in examples:
        weight = float(example.get("loss_weight", 1.0))
        if weight <= 0:
            raise ValueError("biencoder loss weights must be positive")
        weighted_type_mass[example.get("training_type", "single_skill")] += weight
    return {
        "examples": len(examples),
        "training_types": dict(sorted(type_counts.items())),
        "weighted_type_mass": dict(sorted(weighted_type_mass.items())),
        "negative_sources": dict(sorted(sources.items())),
        "mean_negatives": sum(negatives) / len(negatives) if negatives else 0.0,
        "effective_query_batch_size": (
            settings["micro_batch_size"]
            * settings["gradient_accumulation_steps"]
        ),
        "settings": settings,
    }


def train(
    examples: list[dict[str, Any]],
    settings: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    try:
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
    except ImportError as exc:
        raise RuntimeError(
            "training dependencies are missing; install requirements.txt"
        ) from exc
    if not examples:
        raise ValueError("training data contains no examples")
    if settings["temperature"] <= 0:
        raise ValueError("temperature must be positive")

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
    model = AutoModel.from_pretrained(
        settings["model"],
        torch_dtype="auto",
        trust_remote_code=True,
    )
    if settings["gradient_checkpointing"]:
        enable_checkpoint_input_gradients(model)
    model = get_peft_model(
        model,
        LoraConfig(
            task_type="FEATURE_EXTRACTION",
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
    micro_batch = settings["micro_batch_size"]
    accumulation = settings["gradient_accumulation_steps"]
    batches_per_epoch = math.ceil(len(examples) / micro_batch)
    optimizer_steps = math.ceil(batches_per_epoch / accumulation) * settings["epochs"]
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
    history = []
    optimizer.zero_grad(set_to_none=True)
    global_batch = 0
    rng = random.Random(seed)

    for epoch in range(settings["epochs"]):
        order = list(range(len(examples)))
        accumulation_count = 0
        rng.shuffle(order)
        progress = create_training_progress(
            epoch=epoch + 1,
            total_batches=batches_per_epoch,
        )
        for start in range(0, len(order), micro_batch):
            batch = [examples[index] for index in order[start : start + micro_batch]]
            queries = [example["query_text"] for example in batch]
            documents = []
            positive_indices = []
            for example in batch:
                positive_indices.append(len(documents))
                documents.append(example["positive_text"])
                documents.extend(example["negative_texts"])

            with torch.autocast(
                device_type=device,
                dtype=torch.bfloat16,
                enabled=use_bf16,
            ):
                query_embeddings = _encode_train(
                    model,
                    tokenizer,
                    queries,
                    settings["query_max_length"],
                    device,
                )
                document_embeddings = _encode_train(
                    model,
                    tokenizer,
                    documents,
                    settings["skill_max_length"],
                    device,
                )
                loss = info_nce_loss(
                    query_embeddings,
                    document_embeddings,
                    torch.tensor(positive_indices, device=device),
                    settings["temperature"],
                    reduction="none",
                )
                weights = torch.tensor(
                    [float(example.get("loss_weight", 1.0)) for example in batch],
                    dtype=loss.dtype,
                    device=device,
                )
                if bool((weights <= 0).any()):
                    raise ValueError("biencoder loss weights must be positive")
                loss = (loss * weights).mean()
            (loss / accumulation).backward()
            loss_value = float(loss.detach().cpu())
            history.append(
                {
                    "epoch": epoch + 1,
                    "batch": global_batch + 1,
                    "loss": loss_value,
                }
            )
            progress.update(1)
            progress.set_postfix(loss=f"{loss_value:.4f}")
            global_batch += 1
            accumulation_count += 1
            is_boundary = accumulation_count >= accumulation
            is_last = start + micro_batch >= len(order)
            if is_boundary or is_last:
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
        **summarize_examples(examples, settings),
        "optimizer_steps": optimizer_steps,
        "loss_history": history,
    }
    (output_dir / "training_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def _encode_train(
    model: Any,
    tokenizer: Any,
    texts: list[str],
    max_length: int,
    device: str,
) -> Any:
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    outputs = model(**encoded)
    hidden_states = getattr(outputs, "last_hidden_state", outputs[0])
    return last_token_pool(hidden_states, encoded["attention_mask"])


def main() -> None:
    args = build_parser().parse_args()
    settings = resolve_settings(args)
    examples = load_examples(args)
    if args.dry_run:
        result = summarize_examples(examples, settings)
    else:
        result = train(examples, settings, args.seed)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

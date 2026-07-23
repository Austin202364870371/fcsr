"""Prepare Top-20 groups and train the FCSR listwise reranker."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Callable

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


RESUME_STATE_FILENAME = "trainer_state.pt"
RESUME_ADAPTER_DIRNAME = "adapter"


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


def checkpoint_due(processed_groups: int, accumulation_count: int, next_due: int) -> bool:
    """Return whether it is safe and time to persist a resumable checkpoint."""
    return processed_groups >= next_due and accumulation_count == 0


def resume_epoch_position(
    state: dict[str, Any],
    epoch: int,
) -> tuple[list[int], int] | None:
    """Return the unfinished shuffled order only for its saved epoch."""
    if int(state.get("epoch", -1)) != epoch:
        return None
    order = state.get("order")
    next_position = state.get("next_position")
    if not isinstance(order, list) or not isinstance(next_position, int):
        raise ValueError("checkpoint is missing a valid epoch order or next position")
    return list(order), next_position


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
    train.add_argument(
        "--checkpoint-every",
        type=int,
        default=250,
        help="save a resumable checkpoint after at least this many groups",
    )
    train.add_argument(
        "--resume-from",
        help="checkpoint directory containing trainer_state.pt and adapter/",
    )
    train.add_argument(
        "--skip-memory-preflight",
        action="store_true",
        help="skip the longest-group forward/backward memory check",
    )
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


def _checkpoint_dir(output_dir: str | Path, epoch: int, processed_groups: int) -> Path:
    return Path(output_dir) / "resume" / f"epoch-{epoch + 1:02d}-step-{processed_groups:05d}"


def _load_checkpoint(path: str | Path, torch: Any) -> tuple[Path, dict[str, Any]]:
    checkpoint_dir = Path(path)
    state_path = checkpoint_dir / RESUME_STATE_FILENAME
    adapter_dir = checkpoint_dir / RESUME_ADAPTER_DIRNAME
    if not state_path.is_file() or not adapter_dir.is_dir():
        raise FileNotFoundError(
            "resume checkpoint must contain trainer_state.pt and adapter/: "
            f"{checkpoint_dir}"
        )
    try:
        state = torch.load(state_path, map_location="cpu", weights_only=False)
    except TypeError:
        state = torch.load(state_path, map_location="cpu")
    if not isinstance(state, dict):
        raise ValueError("checkpoint trainer_state.pt must contain an object")
    return checkpoint_dir, state


def _validate_resume_settings(state: dict[str, Any], settings: dict[str, Any]) -> None:
    saved = state.get("settings")
    if not isinstance(saved, dict):
        raise ValueError("checkpoint is missing its training settings")
    keys = (
        "model",
        "method",
        "max_length",
        "gradient_accumulation_steps",
        "learning_rate",
        "precision",
        "lora_r",
        "lora_alpha",
        "lora_dropout",
    )
    changed = [key for key in keys if saved.get(key) != settings.get(key)]
    if changed:
        raise ValueError(
            "resume settings differ from the checkpoint for: " + ", ".join(changed)
        )


def _move_optimizer_state(optimizer: Any, device: str, torch: Any) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def _restore_rng_state(state: dict[str, Any], rng: random.Random, torch: Any, device: str) -> None:
    rng_state = state.get("rng_state")
    python_rng_state = state.get("python_rng_state")
    torch_rng_state = state.get("torch_rng_state")
    if rng_state is None or python_rng_state is None or torch_rng_state is None:
        raise ValueError("checkpoint is missing random number generator state")
    rng.setstate(rng_state)
    random.setstate(python_rng_state)
    torch.set_rng_state(torch_rng_state)
    if device == "cuda":
        cuda_rng_state_all = state.get("cuda_rng_state_all")
        if cuda_rng_state_all is None:
            raise ValueError("checkpoint is missing CUDA random number generator state")
        torch.cuda.set_rng_state_all(cuda_rng_state_all)


def _build_group_tensors(
    group: dict[str, Any],
    tokenizer: Any,
    prefix_tokens: list[int],
    suffix_tokens: list[int],
    max_length: int,
    device: str,
    torch: Any,
) -> tuple[Any, Any, Any]:
    tokenized = [
        tokenize_reranker_text(
            str(candidate.get("prompt", "")),
            tokenizer,
            prefix_tokens,
            suffix_tokens,
            max_length,
        )
        for candidate in group["candidates"]
    ]
    group_max_length = max(len(ids) for ids in tokenized)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    padded = [[pad_id] * (group_max_length - len(ids)) + ids for ids in tokenized]
    masks = [[0] * (group_max_length - len(ids)) + [1] * len(ids) for ids in tokenized]
    return (
        torch.tensor(padded, dtype=torch.long, device=device),
        torch.tensor(masks, dtype=torch.long, device=device),
        torch.tensor(group["positive_mask"], device=device),
    )


def _group_loss(
    model: Any,
    group: dict[str, Any],
    tokenizer: Any,
    prefix_tokens: list[int],
    suffix_tokens: list[int],
    settings: dict[str, Any],
    yes_id: int,
    no_id: int,
    device: str,
    use_bf16: bool,
    torch: Any,
) -> Any:
    input_ids, attention_mask, positive_mask = _build_group_tensors(
        group,
        tokenizer,
        prefix_tokens,
        suffix_tokens,
        settings["max_length"],
        device,
        torch,
    )
    with torch.autocast(
        device_type=device,
        dtype=torch.bfloat16,
        enabled=use_bf16,
    ):
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[:, -1, :]
        scores = logits[:, yes_id] - logits[:, no_id]
        return listwise_cross_entropy(scores, positive_mask)


def _run_memory_preflight(
    model: Any,
    groups: list[dict[str, Any]],
    tokenizer: Any,
    prefix_tokens: list[int],
    suffix_tokens: list[int],
    settings: dict[str, Any],
    yes_id: int,
    no_id: int,
    device: str,
    use_bf16: bool,
    optimizer: Any,
    torch: Any,
) -> dict[str, int]:
    def prompt_length(prompt: str) -> int:
        return len(
            tokenize_reranker_text(
                prompt,
                tokenizer,
                prefix_tokens,
                suffix_tokens,
                settings["max_length"],
            )
        )

    with tqdm(
        total=len(groups),
        desc="Reranker preflight: scanning groups",
        unit="group",
        dynamic_ncols=True,
    ) as progress:
        group_index, padded_tokens = longest_group_index(
            groups,
            prompt_length,
            progress=progress.update,
        )
    python_rng_state = random.getstate()
    torch_rng_state = torch.get_rng_state()
    cuda_rng_state_all = torch.cuda.get_rng_state_all() if device == "cuda" else None
    try:
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        loss = _group_loss(
            model,
            groups[group_index],
            tokenizer,
            prefix_tokens,
            suffix_tokens,
            settings,
            yes_id,
            no_id,
            device,
            use_bf16,
            torch,
        )
        loss.backward()
        if device == "cuda":
            torch.cuda.synchronize()
            peak_memory_bytes = int(torch.cuda.max_memory_allocated())
        else:
            peak_memory_bytes = 0
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
    result = {
        "group_index": group_index,
        "padded_tokens": padded_tokens,
        "peak_memory_mib": round(peak_memory_bytes / (1024 * 1024)),
    }
    print("Reranker memory preflight:", json.dumps(result))
    return result


def _save_training_checkpoint(
    model: Any,
    tokenizer: Any,
    optimizer: Any,
    scheduler: Any,
    output_dir: str | Path,
    settings: dict[str, Any],
    epoch: int,
    next_position: int,
    order: list[int],
    rng: random.Random,
    history: list[dict[str, Any]],
    device: str,
    torch: Any,
) -> Path:
    checkpoint_dir = _checkpoint_dir(output_dir, epoch, next_position)
    adapter_dir = checkpoint_dir / RESUME_ADAPTER_DIRNAME
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    state = {
        "version": 1,
        "settings": settings,
        "epoch": epoch,
        "next_position": next_position,
        "order": order,
        "rng_state": rng.getstate(),
        "python_rng_state": random.getstate(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all() if device == "cuda" else None,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "history": history,
    }
    state_path = checkpoint_dir / RESUME_STATE_FILENAME
    temp_path = state_path.with_suffix(".tmp")
    torch.save(state, temp_path)
    temp_path.replace(state_path)
    print(f"Reranker checkpoint saved: {checkpoint_dir}")
    return checkpoint_dir


def train(
    groups: list[dict[str, Any]],
    settings: dict[str, Any],
    seed: int,
    checkpoint_every: int = 250,
    resume_from: str | None = None,
    run_memory_preflight: bool = True,
) -> dict[str, Any]:
    try:
        import torch
        from peft import LoraConfig, PeftModel, get_peft_model
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
    if checkpoint_every <= 0:
        raise ValueError("checkpoint_every must be positive")
    if resume_from and settings["method"] != "lora":
        raise ValueError("resuming is currently supported only for --method lora")

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    resume_dir: Path | None = None
    resume_state: dict[str, Any] | None = None
    if resume_from:
        resume_dir, resume_state = _load_checkpoint(resume_from, torch)
        _validate_resume_settings(resume_state, settings)

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
        if resume_dir is not None:
            model = PeftModel.from_pretrained(
                model,
                resume_dir / RESUME_ADAPTER_DIRNAME,
                is_trainable=True,
            )
        else:
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
    history: list[dict[str, Any]] = []
    start_epoch = 0
    if resume_state is not None:
        optimizer.load_state_dict(resume_state["optimizer"])
        _move_optimizer_state(optimizer, device, torch)
        scheduler.load_state_dict(resume_state["scheduler"])
        _restore_rng_state(resume_state, rng, torch, device)
        history = list(resume_state.get("history", []))
        start_epoch = int(resume_state["epoch"])
        if start_epoch >= settings["epochs"]:
            raise ValueError("checkpoint has already completed all configured epochs")
        print(f"Reranker resume checkpoint: {resume_dir}")
    optimizer.zero_grad(set_to_none=True)

    preflight = None
    if run_memory_preflight:
        preflight = _run_memory_preflight(
            model,
            groups,
            tokenizer,
            prefix_tokens,
            suffix_tokens,
            settings,
            yes_id,
            no_id,
            device,
            use_bf16,
            optimizer,
            torch,
        )

    for epoch in range(start_epoch, settings["epochs"]):
        resumed = resume_epoch_position(resume_state, epoch) if resume_state else None
        if resumed is None:
            order = list(range(len(groups)))
            rng.shuffle(order)
            start_position = 0
        else:
            order, start_position = resumed
            if len(order) != len(groups) or not 0 <= start_position < len(order):
                raise ValueError("checkpoint order does not match the current training groups")
        accumulation_count = 0
        next_checkpoint_due = start_position + checkpoint_every
        progress = create_training_progress(
            epoch=epoch + 1,
            total_groups=len(order),
        )
        if start_position:
            progress.update(start_position)
            progress.set_postfix(resumed=start_position)
        for position in range(start_position, len(order)):
            group = groups[order[position]]
            loss = _group_loss(
                model,
                group,
                tokenizer,
                prefix_tokens,
                suffix_tokens,
                settings,
                yes_id,
                no_id,
                device,
                use_bf16,
                torch,
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
                processed_groups = position + 1
                if not is_last and checkpoint_due(
                    processed_groups,
                    accumulation_count,
                    next_checkpoint_due,
                ):
                    _save_training_checkpoint(
                        model,
                        tokenizer,
                        optimizer,
                        scheduler,
                        settings["output_dir"],
                        settings,
                        epoch,
                        processed_groups,
                        order,
                        rng,
                        history,
                        device,
                        torch,
                    )
                    next_checkpoint_due = processed_groups + checkpoint_every
        progress.close()
        resume_state = None

    output_dir = Path(settings["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    summary = {
        **summarize_groups(groups, settings),
        "optimizer_steps": optimizer_steps,
        "loss_history": history,
        "memory_preflight": preflight,
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
    return train(
        groups,
        settings,
        args.seed,
        checkpoint_every=args.checkpoint_every,
        resume_from=args.resume_from,
        run_memory_preflight=not args.skip_memory_preflight,
    )


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "prepare":
        result = run_prepare(args)
    else:
        result = run_train(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

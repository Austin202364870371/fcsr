# Reranker Training Resilience Design

## Goal

Prevent an hours-long reranker training run from losing all useful progress when a later group causes a CUDA out-of-memory failure or the server is interrupted.

## Design

The training command will run a memory preflight before optimization. It tokenizes every group with the active tokenizer and configured maximum length, chooses the group with the largest padded token length, and executes one forward/backward pass without an optimizer update. A preflight OOM fails before the full epoch begins.

Training will save a resumable checkpoint at optimizer-step boundaries every configurable number of processed groups. A checkpoint contains the LoRA adapter, optimizer and scheduler state, epoch order and next position, loss history, and random-number-generator state. Resume reloads the base model plus adapter, restores the optimizer/scheduler and continues the exact unfinished order.

## Constraints

- Preserve the existing Top-20 listwise objective and one-group micro-batch.
- Default to a 1536-token cap for the current 24GB RTX 4090 command, supplied at the CLI rather than changing the research configuration file.
- Do not save a partial gradient-accumulation window; checkpoints occur only immediately after an optimizer update.
- Keep the existing final adapter and `training_summary.json` output contract unchanged.

## User Interface

`train` gains `--checkpoint-every`, `--resume-from`, and `--skip-memory-preflight`. The default checkpoint interval is 250 groups; the actual save may be delayed to the next gradient-accumulation boundary.

Checkpoints live under `<output-dir>/resume/` and a resume command accepts that directory. The preflight prints the selected group index, padded token length, and peak allocated CUDA memory when CUDA is available.

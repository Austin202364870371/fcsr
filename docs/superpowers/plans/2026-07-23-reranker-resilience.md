# Reranker Training Resilience Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a worst-case memory preflight and resumable reranker checkpoints.

**Architecture:** Keep listwise group training unchanged. Add pure helpers for checkpoint scheduling and state validation, then use them in `scripts/train_reranker.py` to preflight the longest padded group and persist enough state to reconstruct the unfinished shuffled epoch.

**Tech Stack:** Python 3.10, PyTorch, PEFT, Transformers, unittest.

## Global Constraints

- Keep Top-20 listwise training and the current final checkpoint format.
- Persist checkpoints only after an optimizer step.
- Preserve deterministic epoch order after resume.

---

### Task 1: Add checkpoint scheduling and resume-state tests

**Files:**
- Modify: `tests/test_train_reranker_progress.py`
- Modify: `scripts/train_reranker.py`

- [x] Write tests for a checkpoint being due only at an accumulation boundary, for a resume state selecting the saved order and next position, and for a longest-group selector accepting an injected length function.
- [x] Run `python -B -m unittest tests.test_train_reranker_progress -v` and verify the new assertions fail because the helpers do not exist.
- [x] Implement the pure helpers in `scripts/train_reranker.py`.
- [x] Re-run the focused test and verify it passes.

### Task 2: Add training preflight and resumable state

**Files:**
- Modify: `scripts/train_reranker.py`
- Modify: `tests/test_train_reranker_progress.py`

- [x] Add parser flags for checkpoint interval, resume location, and skipping preflight.
- [x] Save LoRA adapter, tokenizer, optimizer/scheduler state, shuffled epoch order, RNG state and loss history at safe boundaries.
- [x] Reload those values before the training loop and continue from the saved group offset.
- [x] Run focused no-CUDA unit tests and the full test suite.

### Task 3: Document the safe 4090 command

**Files:**
- Modify: `README.md`
- Modify: `README_EN.md`

- [x] Document the automatic worst-case preflight, checkpoint location, resume command, and `--max-length 1536` command for a 24GB GPU.
- [x] Run the full unittest suite and `git diff --check`.

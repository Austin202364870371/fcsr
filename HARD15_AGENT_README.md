# Hard-15 Agent Pilot

This stage connects the frozen hard FCSR reranker output to a leakage-controlled
Agent planning input. It deliberately refuses to treat task text alone as an
executable benchmark.

## Evidence behind the implementation

The implementation follows four benchmark principles:

1. [SkillsBench](https://arxiv.org/abs/2602.12670) treats a runnable task as a
   containerized environment with fixed data, an oracle, and a deterministic
   verifier. Its [official repository](https://github.com/benchflow-ai/skillsbench)
   uses `environment/Dockerfile` and `verifier/test.sh` or
   `verifier/test_outputs.py`; the local audit checks the same minimum boundary.
2. SkillsBench uses matched conditions on the same task and environment. The
   pilot therefore freezes final FCSR `reranked_candidates` before changing Skill
   presentation or planning.
3. [ReAct](https://arxiv.org/abs/2210.03629) separates model reasoning from
   environment actions. Accordingly, an instruction Skill is context and may
   have no tool binding; only the later executor can invoke sandboxed tools.
4. [Plan-and-Solve](https://arxiv.org/abs/2305.04091) motivates producing an
   explicit plan before execution. The pilot requests a compact JSON plan whose
   Skill aliases and step dependencies can be validated before any action.

A recent controlled
[SkillsBench presentation study](https://arxiv.org/abs/2605.31408) also reports
paired, multi-trial comparisons and model-dependent presentation effects. This
supports keeping the same model, task, candidate set, and budget across later
Flat/Hierarchy/Graph conditions.

## Privacy and experimental controls

The public planner input contains only `S01` through `S20`, Skill name,
description, body, rank, and reranker score. Real IDs and `gt_skill_ids` are
written to a separate evaluation file. Never pass `evaluation.jsonl` to the LLM.

The sample is deterministic (`seed=42`) and asks for:

- 3 single-Skill tasks with complete FCSR Top-20 coverage;
- 2 single-Skill tasks with incomplete coverage;
- 5 multi-Skill tasks with complete coverage;
- 5 multi-Skill tasks with incomplete coverage.

Only tasks with a local container definition and deterministic verifier are
eligible. Missing tasks are logged rather than silently replaced.

## Current local audit

The `fcsr` repository contains query text and FCSR outputs, but not the original
task packages. The strict audit therefore marks all 75 tasks as
`missing_task_environment` and refuses to create a nominal 15-task sample. This
is expected and prevents planning-only output from being reported as task
execution evidence.

## Prepare on the server

Place or clone the official runnable task packages so the directory contains:

```text
<task-environments>/<task-id>/
  environment/Dockerfile
  verifier/test.sh            # or verifier/test_outputs.py
```

Then run:

```bash
conda activate agent-learn
export PYTHONPATH=src
python -B scripts/prepare_hard_pilot.py \
  --queries data/raw/evaluation_queries.jsonl \
  --rankings reports/hard/fcsr/reranker_hard.jsonl \
  --skills data/raw/skills_hard.jsonl \
  --task-environments /path/to/skillsbench/tasks \
  --output-dir reports/agent/hard15
```

The command always writes `environment_audit.jsonl` and `summary.json`. When the
fixed quotas can be satisfied it also writes:

- `tasks.jsonl`: anonymous Agent-visible tasks;
- `evaluation.jsonl`: private alias/GT mapping, used only after planning.

Generate DeepSeek plans only after `summary.json` reports `status: ready`:

```bash
python -B scripts/plan_hard_pilot.py \
  --tasks reports/agent/hard15/tasks.jsonl \
  --output reports/agent/hard15/plans.jsonl \
  --model deepseek-v4-flash \
  --limit 15
```

The planner loads `DEEPSEEK_API_KEY` from the project `.env`, sends only the
public file, uses temperature 0 and JSON mode, validates every returned alias,
and records prompt/completion tokens. It does not execute commands or claim task
success. End-to-end pass/fail begins only after the server-side container
executor and verifier are connected.

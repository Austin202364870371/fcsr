# FCSR

[中文说明](README.zh-CN.md)

FCSR (Function-aware Coverage Skill Retriever) is a reproducible pipeline for large-scale Agent Skill retrieval. It builds evidence-grounded Skill Contracts, derives single-skill and multi-skill synthetic training data, fine-tunes Qwen bi-encoder and reranker adapters, and evaluates retriever or full two-stage systems with a SkillRouter-compatible protocol.

Contracts are used only to construct auditable training data. At inference time, FCSR retrieves the original Skill text: `name + description + body`.

## What Is Included

| Component | Purpose |
|---|---|
| Contract extraction | Creates source-evidence-grounded Contract records from a benchmark-safe sample. |
| Single-skill data | Generates queries and safe local/semantic negatives. |
| Multi-skill data | Creates Contract-validated skill pairs/triples, then lets an LLM phrase only the validated compositions. |
| Training | Fine-tunes Qwen3 Embedding and Qwen3 Reranker with LoRA. |
| Evaluation | Runs dense, BM25, RRF hybrid, reranking, scoring, and canonical result-table rendering. |

## Repository Layout

```text
configs/                         Shared model and data defaults
data/
  samples/                       Flat sample files and sample manifest
  contracts/                     Extracted Contracts, failures, and manifest
  raw/                           Skill pools and public evaluation tasks
  pilots/                        Isolated pilot outputs
  synthetic/
    single_skill/                Single-skill queries, negatives, and manifest
    multi_skill/                 Validated candidates, queries, and manifest
  training/                      Single-pass, type-weighted mixed training data
docs/                            Research-question notes and references
jobs/                            Slurm job templates
scripts/
  build_single_skill_data.py     Contract and single-skill dataset pipeline
  build_multiskill_candidates.py Contract-guided pair/triple construction
  generate_multiskill_queries.py LLM query authoring with strict validation
  build_multiskill_training_data.py
  train_biencoder.py
  train_reranker.py
  evaluate.py
  render_evaluation_tables.py
src/                             Reusable implementation modules
tests/                           Offline regression tests
```

`data/samples/` is intentionally flat: `sample_skills.jsonl.gz` and
`manifest.json` live directly in it. Git tracks every `data/**/manifest.json`, while
large datasets, models, checkpoints, logs, caches, and generated reports stay ignored.

## Setup

Use Python 3.10 or newer. Run commands from the repository root.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Copy-Item .env.example .env
# Edit .env and set DEEPSEEK_API_KEY.
$env:PYTHONPATH = "src"
python -B -m unittest discover -s tests -v
```

On CUDA hosts, install the PyTorch build matching the host CUDA environment before installing `requirements.txt` when necessary. See [PyTorch installation](https://pytorch.org/get-started/locally/).

Contract extraction and LLM-authored queries use `deepseek-v4-flash` with thinking
disabled and JSON Output enabled. Copy `.env.example` to `.env`, set
`DEEPSEEK_API_KEY`, and never commit that file. Contract extraction sends exactly one
Skill per request and maintains a bounded pool of 16 concurrent requests. Single- and
multi-Skill query generation use the same continuously refilled request pool; a retry
occupies only the failed item's worker and never replays a successful item.
Single-Skill query prompting targets 110-140 words inside the hard 80-180-word
validation window and supplies the exact source Skill name only as forbidden negative
metadata. Final query failures preserve the rejected response for diagnosis.

## Data Pipeline

### 1. Build Single-Skill Data

```powershell
# Benchmark-safe 32k Contract sample, then evidence-grounded Contracts.
python -B scripts/build_single_skill_data.py sample `
  --sample-size 32000 --output-dir data/samples --overwrite
python -B scripts/build_single_skill_data.py contracts `
  --model deepseek-v4-flash --concurrency 16 `
  --sample data/samples/sample_skills.jsonl.gz `
  --output data/contracts/contracts.jsonl.gz `
  --failures data/contracts/failures.jsonl.gz

# Contract-grounded queries, local negatives, then GPU semantic negatives.
python -B scripts/build_single_skill_data.py queries
python -B scripts/build_single_skill_data.py local-negatives --overwrite
python -B scripts/build_single_skill_data.py semantic-negatives `
  --model models/Qwen3-Embedding-0.6B --device cuda --overwrite
```

The resulting single-skill data is stored in `data/synthetic/single_skill/`.
Local mining combines BM25, same-category, and random candidates, then removes identity,
normalized-name, exact-body, and high trigram-overlap false negatives. GPU semantic
mining uses local Qwen3 Embedding Top-50 retrieval and removes candidates whose cosine
similarity to the positive Skill is at least `0.95`. Removed candidates are retained
for audit in `semantic_fn_review.jsonl.gz`; the filtered data is then converted into
directly trainable bi-encoder and reranker records.

### 2. Build Multi-Skill Data

Candidate construction does not call an LLM. A candidate is retained only when valid Contracts, non-benchmark Skills, artifact handoff, and complementary operations support the ordered pair or triple.

```powershell
python -B scripts/build_multiskill_candidates.py

# DeepSeek writes tasks only for validated candidates.
python -B scripts/generate_multiskill_queries.py `
  --model deepseek-v4-flash --concurrency 16 --max-attempts 3 --progress
```

The generator validates JSON structure, exact positive Skill IDs and order, source hashes, subtask coverage, and the dependency DAG. It writes `queries.jsonl.gz`, `query_failures.jsonl.gz`, and `review_queue.jsonl.gz` under `data/synthetic/multi_skill/`.

### 3. Build Mixed Training Data

`mixed` stores every original query/group exactly once; there are no
threefold copies. A pair or triple still produces one bi-encoder record per distinct
positive Skill because each positive label must be trained once. The reranker keeps the
same query as one multi-label group. Negatives come only from the Easy pool and exclude
every positive Skill ID. Semantic Top-64 candidates are also compared against every
positive Skill at the `0.95` threshold; removals are saved to
`data/training/semantic_fn_review.jsonl.gz`. Each epoch shuffles all single- and
multi-Skill records together.

The default type weights are `1.5` for multi-Skill bi-encoder records and `3.0` for
multi-Skill reranker groups. They keep the on-disk distribution natural while raising
the expected multi-Skill gradient share from roughly 13–15% to 18–21% for the
bi-encoder and from roughly 6–8% to 16–21% for the reranker. These are explicit starting
points for ablation, not literature-derived constants.

```bash
python -B scripts/build_multiskill_training_data.py \
  --negative-model models/Qwen3-Embedding-0.6B \
  --biencoder-multi-loss-weight 1.5 \
  --reranker-multi-loss-weight 3.0 \
  --output-dir data/training
```

## Training

The following commands use local Hugging Face model directories and produce clearly named LoRA adapters.

```bash
python -B scripts/train_biencoder.py \
  --train-data data/training/biencoder.jsonl.gz \
  --skills data/raw/skills_easy.jsonl.gz \
  --model models/Qwen3-Embedding-0.6B \
  --output-dir checkpoints/fcsr-emb-0.6b-multiskill-weighted

python -B scripts/train_reranker.py train \
  --groups data/training/reranker.jsonl.gz \
  --skills data/raw/skills_easy.jsonl.gz \
  --model models/Qwen3-Reranker-0.6B \
  --output-dir checkpoints/fcsr-rank-0.6b-multiskill-weighted
```

For the single-skill baseline, omit the mixed build step and explicitly pass
`data/synthetic/single_skill/train_biencoder.jsonl.gz` or
`data/synthetic/single_skill/train_reranker.jsonl.gz`, with checkpoint names
`fcsr-emb-0.6b-single` and `fcsr-rank-0.6b-single`.

## Evaluation

`evaluate.py` exposes independent stages so every experiment has explicit artifacts.

```bash
# First stage: RRF candidate recall.
python -B scripts/evaluate.py hybrid \
  --queries data/raw/evaluation_queries.jsonl.gz \
  --skills data/raw/skills_hard.jsonl.gz \
  --model checkpoints/fcsr-emb-0.6b \
  --top-k 50 --fusion-depth 100 --rrf-k 60 \
  --output-predictions reports/retrieval/hard/hybrid/predictions.json \
  --output-records reports/retrieval/hard/hybrid/records.jsonl

# Second stage: rerank the Top-50 candidates, keeping Top-10 for scoring.
python -B scripts/evaluate.py rerank \
  --retrieval-records reports/retrieval/hard/hybrid/records.jsonl \
  --skills data/raw/skills_hard.jsonl.gz \
  --model checkpoints/fcsr-rank-0.6b-multiskill-weighted \
  --top-k 10 \
  --output-predictions reports/reranker/hard/rrf-base-emb-multiskill-weighted/predictions.json \
  --output-records reports/reranker/hard/rrf-base-emb-multiskill-weighted/records.jsonl

python -B scripts/evaluate.py score \
  --tasks data/raw/evaluation_queries.jsonl.gz \
  --skills data/raw/skills_hard.jsonl.gz \
  --predictions reports/reranker/hard/rrf-base-emb-multiskill-weighted/predictions.json \
  --stage reranker --tier hard \
  --output-dir reports/reranker/hard/rrf-base-emb-multiskill-weighted

# Render one final-system table and one two-stage ablation table.
python -B scripts/render_evaluation_tables.py
```

The renderer writes:

- `reports/tables/hard-baselines.md`: one final output for each system.
- `reports/tables/hard-two-stage-ablation.md`: retrieval-versus-rerank comparisons for two-stage systems.

Bold values in the final table identify numerical maxima only, not statistical significance.

## Slurm API Generation

API generation is a CPU task and must still run through Slurm. The pilot job creates
the deterministic 32k stratified sample when needed, but extracts only the first 32
Skills into an isolated output directory:

```bash
sbatch jobs/extract_contracts_deepseek_pilot.sbatch
```

After auditing the pilot, submit the separate formal job for all 32,000 Skills:

```bash
sbatch jobs/extract_contracts_deepseek.sbatch
```

Each request contains one Skill; `CONCURRENCY=16` controls only how many independent
requests are in flight. Completed records are validated and appended individually.
Existing `(skill_id, source_hash)` records are skipped, so resubmitting the formal job
resumes safely. Prompt 007 reads up to 20,000 body characters, orders fields by
importance, and enforces caps of 12
operations, 12 constraints, 10 outputs, and 8 items for every other collection;
the materializer applies the same caps and removes constraint/exclusion duplicates
that cite the same evidence. It also caps all collection items at 32, removes
input/precondition items supported only by Skill-trigger phrases, and retains exclusions
only when their evidence contains explicit negative scope language or appears directly
under an exclusion heading. Configurable feature flags are not exclusions; workflow
steps and user requests are not preconditions; near-duplicate constraints and quality
criteria are collapsed. Contract responses allow up to 6,144 output tokens. Terminal failures
record the last raw response and provider `finish_reason` when available, which makes
truncation distinguishable from malformed JSON. The formal output is
`data/contracts/contracts.jsonl.gz`.

The remaining build is submitted in dependency order:

```bash
# 1. DeepSeek single-Skill queries (16 concurrent requests).
sbatch jobs/generate_single_skill_deepseek.sbatch

# 2. Local negatives and local false-negative filtering.
sbatch jobs/mine_single_skill_local_negatives.sbatch

# 3. Qwen semantic negatives and semantic false-negative filtering.
sbatch jobs/mine_single_skill_semantic_negatives.sbatch

# 4. Convert filtered single-Skill records into reranker groups.
sbatch jobs/prepare_single_skill_reranker.sbatch

# 5. Build Contract-validated multi-Skill candidates.
sbatch jobs/build_multi_skill_candidates.sbatch

# 6. Audit a 50-candidate DeepSeek pilot.
sbatch jobs/generate_multiskill_deepseek.sbatch

# 7. Generate all validated multi-Skill queries after pilot review.
sbatch --export=ALL,FULL_RUN=1 jobs/generate_multiskill_deepseek.sbatch

# 8. Mine multi-Skill negatives, filter false negatives, and build mixed data.
sbatch jobs/build_mixed_training_data.sbatch
```

The two DeepSeek query jobs use `deepseek-v4-flash`, JSON Output, disabled thinking,
one item per request, and 16 requests in flight. The negative-mining and training-data
jobs use only local Qwen models. Add Slurm `afterok` dependencies when submitting the
sequence together; do not start a downstream stage before its manifest and artifacts
have been audited.

## Why Weighted Mixing Instead of 3x Copies

The previous 8k run contained 7,342 single-Skill and 541 validated multi-Skill queries,
or about 93.1% versus 6.9% before expansion. A 32k run should therefore be planned as
roughly 30k–31.5k single-Skill plus 2.2k–2.4k multi-Skill queries if candidate yield is
similar; the actual counts must come from the new manifests. Pair/triple positive
expansion makes the bi-encoder's raw multi-Skill record share higher than the query
share, so using one sampling multiplier for both models is not well calibrated.

Multi-task literature generally treats task proportions as an optimization policy:
interleaving tasks can reduce forgetting, and performance-aware or learned task
sampling can outperform uniform sampling. Retrieval work also warns that naive
multi-task mixing can trail task-specialized models. Accordingly, FCSR preserves each
original example once, interleaves both types by epoch shuffling, and applies separate
model-specific weights. See [Dynamic Sampling Strategies for Multi-Task Reading
Comprehension](https://aclanthology.org/2020.acl-main.86/), [Learning Task Sampling
Policy for Multitask Learning](https://aclanthology.org/2021.findings-emnlp.375/),
[Multi-Task Retrieval for Knowledge-Intensive Tasks](https://aclanthology.org/2021.acl-long.89/),
and [Improving Multitask Retrieval by Promoting Task
Specialization](https://aclanthology.org/2023.tacl-1.68/). The default weights are an
FCSR engineering choice and should be compared against `1.0/1.0` and at least one
stronger weighting setting under the same epoch and seed budget.

## Documentation

- [RQ1: Skill Retrieval](docs/rq1-skill-retrieval.md)
- [Research question map](docs/README.md)
- [Chinese README](README.zh-CN.md)

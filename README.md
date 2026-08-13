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
  contracts/                     Sampled Skills and extracted Contracts
  raw/                           Skill pools and public evaluation tasks
  synthetic/
    single_skill_v1/             Single-skill queries and training records
    multiskill_v1/               Validated candidates, LLM queries, and manifests
  training/
    multiskill3x/                Generated mixed training data (ignored by Git)
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

Large datasets, model files, checkpoints, logs, caches, and generated reports are intentionally ignored by Git. Tracked `manifest.json` files record the version, inputs, parameters, and counts of generated datasets.

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
Skill per request and maintains a bounded pool of 16 concurrent requests.

## Data Pipeline

### 1. Build Single-Skill Data

```powershell
# Benchmark-safe 32k Contract sample, then evidence-grounded Contracts.
python -B scripts/build_single_skill_data.py sample `
  --sample-size 32000 --output-dir data/contracts_32k --overwrite
python -B scripts/build_single_skill_data.py contracts `
  --model deepseek-v4-flash --concurrency 16 `
  --sample data/contracts_32k/sample_skills.jsonl.gz `
  --output data/contracts_32k/contracts.jsonl.gz `
  --failures data/contracts_32k/failures.jsonl.gz

# Contract-grounded queries, local negatives, then GPU semantic negatives.
python -B scripts/build_single_skill_data.py queries
python -B scripts/build_single_skill_data.py local-negatives --overwrite
python -B scripts/build_single_skill_data.py semantic-negatives `
  --model models/Qwen3-Embedding-0.6B --device cuda --overwrite
```

The resulting single-skill data is stored in `data/synthetic/single_skill_v1/`.

### 2. Build Multi-Skill Data

Candidate construction does not call an LLM. A candidate is retained only when valid Contracts, non-benchmark Skills, artifact handoff, and complementary operations support the ordered pair or triple.

```powershell
python -B scripts/build_multiskill_candidates.py

# DeepSeek writes tasks only for validated candidates.
python -B scripts/generate_multiskill_queries.py `
  --model deepseek-v4-flash --max-attempts 3 --progress
```

The generator validates JSON structure, exact positive Skill IDs and order, source hashes, subtask coverage, and the dependency DAG. It writes `queries.jsonl.gz`, `failures.jsonl.gz`, and `review_queue.jsonl.gz` under `data/synthetic/multiskill_v1/`.

### 3. Build Mixed Training Data

`multiskill3x` keeps all single-skill examples and deterministically repeats each original multi-skill task group three times. Each positive Skill becomes a bi-encoder example; the reranker keeps all positives in one multi-label group. Negatives are mined only from the Easy pool and exclude every positive Skill ID.

```bash
python -B scripts/build_multiskill_training_data.py \
  --negative-model models/Qwen3-Embedding-0.6B \
  --multiplier 3 \
  --output-dir data/training/multiskill3x
```

## Training

The following commands use local Hugging Face model directories and produce clearly named LoRA adapters.

```bash
python -B scripts/train_biencoder.py \
  --train-data data/training/multiskill3x/biencoder.jsonl.gz \
  --skills data/raw/skills_easy.jsonl.gz \
  --model models/Qwen3-Embedding-0.6B \
  --output-dir checkpoints/fcsr-emb-0.6b-multiskill3x

python -B scripts/train_reranker.py train \
  --groups data/training/multiskill3x/reranker.jsonl.gz \
  --skills data/raw/skills_easy.jsonl.gz \
  --model models/Qwen3-Reranker-0.6B \
  --output-dir checkpoints/fcsr-rank-0.6b-multiskill3x
```

For the single-skill baseline, omit the mixed build step and use the default inputs with `checkpoints/fcsr-emb-0.6b` and `checkpoints/fcsr-rank-0.6b`.

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
  --model checkpoints/fcsr-rank-0.6b-multiskill3x \
  --top-k 10 \
  --output-predictions reports/reranker/hard/rrf-base-emb-multiskill3x/predictions.json \
  --output-records reports/reranker/hard/rrf-base-emb-multiskill3x/records.jsonl

python -B scripts/evaluate.py score \
  --tasks data/raw/evaluation_queries.jsonl.gz \
  --skills data/raw/skills_hard.jsonl.gz \
  --predictions reports/reranker/hard/rrf-base-emb-multiskill3x/predictions.json \
  --stage reranker --tier hard \
  --output-dir reports/reranker/hard/rrf-base-emb-multiskill3x

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
sbatch jobs/extract_contracts_deepseek_32k.sbatch
```

Each request contains one Skill; `CONCURRENCY=16` controls only how many independent
requests are in flight. Completed records are validated and appended individually.
Existing `(skill_id, source_hash)` records are skipped, so resubmitting the formal job
resumes safely. The formal output is `data/contracts_32k/contracts.jsonl.gz`.

The multi-Skill API generation template is
[jobs/generate_multiskill_deepseek.sbatch](jobs/generate_multiskill_deepseek.sbatch):

```bash
sbatch jobs/generate_multiskill_deepseek.sbatch
```

## Migrating Existing Local or Server Data

The refactor changes only names, not data schemas. After `git pull`, the new directories already contain their tracked manifests, so merge only ignored data files into them:

```bash
rsync -a --exclude manifest.json data/synthetic/single_v1/ \
  data/synthetic/single_skill_v1/
rsync -a --exclude manifest.json data/synthetic/compositional_v1/ \
  data/synthetic/multiskill_v1/
mv data/synthetic/multiskill_v1/compositional_queries.jsonl.gz \
  data/synthetic/multiskill_v1/queries.jsonl.gz
mv data/training/rq1-mixed-3x data/training/multiskill3x
mv data/processed/contract_fn_review.jsonl.gz \
  data/processed/semantic_negative_review.jsonl.gz
mv data/processed/synthetic_top20.json \
  data/processed/single_skill_top20_predictions.json
mv data/processed/synthetic_top20.jsonl.gz \
  data/processed/single_skill_top20_records.jsonl.gz
```

For existing reports, rename `fcsr-base` to `fcsr-single` and `base-rerank` to `dense-base-reranker` at the same path. After checking the copied files, remove the two old synthetic directories. Regenerate the corresponding manifest after a new dataset build; do not edit record content by hand.

## Documentation

- [RQ1: Skill Retrieval](docs/rq1-skill-retrieval.md)
- [Research question map](docs/README.md)
- [Chinese README](README.zh-CN.md)

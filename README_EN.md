# FCSR: Function-aware Coverage Skill Retriever

**English** | [简体中文](README.md)

FCSR is a compact training and evaluation framework for Agent Skill retrieval over an approximately 80K-skill registry. It follows SkillRouter's public retrieval format and two-stage Bi-Encoder/Reranker design, then adds an evidence-grounded **Skill Contract** teacher for synthetic query generation.

Contracts are sidecars for sampled training positives. The deployed encoder still indexes the original `name + description + body`, so the full pool does not require 80K Contract API calls.

## Pipeline

```text
Easy natural pool
  -> benchmark-safe category x language sampling (8,000)
  -> DeepSeek Contract V2 extraction
  -> Contract-grounded query generation
  -> BM25 + same-category + random negatives
  -> Qwen semantic negatives + false-negative filters
  -> Bi-Encoder InfoNCE training
  -> trained Bi-Encoder Top-20 groups
  -> listwise Reranker training
  -> Easy/Hard retrieval, reranking, and SR-compatible scoring
```

The Hard pool is used for evaluation only. It contains the Easy natural pool plus benchmark distractors.

## Layout

```text
configs/              paths, thresholds, and RTX 4090-safe defaults
scripts/              preprocessing, training, and evaluation entry points
src/                  flat reusable modules with no nested Python package
tests/                offline unit tests with fake API clients
data/raw/             75 queries and the Easy/Hard skill pools
data/contracts/       8K sample, Contract sidecars, failures, and manifest
data/processed/       intermediate retrieval results and review records
data/synthetic/       generated queries and training records
checkpoints/           LoRA or full fine-tuning checkpoints
reports/               SR-compatible metric summaries and per-task details
```

All directory names on disk are lowercase. Indentation above only describes the hierarchy.

## 1. Local Setup

Run from the `fcsr` root on the local computer:

```powershell
python -m pip install -r requirements.txt
$env:PYTHONPATH = "src"
python -B -m unittest discover -s tests -v
```

If PyTorch is not installed locally, four numerical training tests are skipped automatically while the remaining tests still run. Installing `requirements-train.txt` enables those tests.

Required raw files:

```text
data/raw/evaluation_queries.jsonl   consolidated 75 scored tasks
data/raw/skills_easy.jsonl          78,361-skill natural pool
data/raw/skills_hard.jsonl          79,141-skill hard pool
```

## 2. Sample 8,000 Skills Locally

```powershell
python -B scripts/preprocess.py sample `
  --skills data/raw/skills_easy.jsonl `
  --tasks data/raw/evaluation_queries.jsonl `
  --sample-size 8000 --seed 42 `
  --output-dir data/contracts --overwrite
```

Outputs:

```text
data/contracts/sample_skills.jsonl
data/contracts/manifest.json
```

The sampler excludes every benchmark GT/relevance skill, removes exact content duplicates, and applies deterministic square-root allocation across category-language strata.

## 3. Extract Contract V2 with DeepSeek V4 Flash

Add the DeepSeek API key to the `.env` file in the project root:

```dotenv
DEEPSEEK_API_KEY=your-real-api-key
```

The program loads this file automatically without overriding an environment variable already exported in the terminal. `.env` is listed in `.gitignore` and must not be committed or uploaded to the server; the provided `scp` command does not include it.

The default model is [`deepseek-v4-flash`](https://api-docs.deepseek.com/news/news260424/), and the API base URL remains `https://api.deepseek.com`. All LLM preprocessing commands explicitly use non-thinking mode by default to reduce batch extraction latency; add `--thinking enabled` to opt back in. First test three records:

```powershell
python -B scripts/preprocess.py contracts --limit 3
```

Inspect `data/contracts/contracts.jsonl` and `data/contracts/failures.jsonl`. Then resume the full sample by omitting `--limit`:

```powershell
python -B scripts/preprocess.py contracts
```

The terminal progress bar shows processed and total records plus `ok` (succeeded), `skip` (already completed), and `fail` counts. Add `--no-progress` for scripts or log jobs that do not need dynamic terminal output.

A completed `(skill_id, source_hash)` is skipped automatically. Each semantic field must cite an exact original quote; offsets and hashes are computed by code and validated by `src/contract_schema.py`.

The extractor only repairs citations whose differences can be explained by Markdown markers or whitespace, then stores the recovered source text and offsets in the Contract. Optional items with no valid evidence are dropped and recorded in `extraction.warnings`; a `capability` without valid evidence still fails and is retried with the validation error. Old failure records are removed from `failures.jsonl` once a valid output exists.

## 4. Generate Contract-Grounded Queries

Use the same three-record smoke test, then resume all validated Contracts:

```powershell
python -B scripts/preprocess.py queries --limit 3
python -B scripts/preprocess.py queries
```

The command shows a `Queries` progress bar with `ok`, `skip`, and `fail` counts. Add `--no-progress` to disable dynamic output in log jobs.

Output: `data/synthetic/queries.jsonl`. Query Prompt V5 requires 80--180 English words and treats the Contract's operations, outputs, constraints, and quality criteria as an allowlist of requested work; surrounding business logic may appear only as an already-existing scenario. Queries with an invalid length, a leaked skill name, invalid JSON, or no current validated Contract are rejected or retried and recorded as failures where applicable.

## 5. Mine Local Negatives

```powershell
python -B scripts/preprocess.py local-negatives `
  --queries data/synthetic/queries.jsonl `
  --skills data/raw/skills_easy.jsonl `
  --output data/synthetic/local_negatives.jsonl `
  --seed 42 --overwrite
```

The terminal reports skill loading, BM25 index construction, and per-query mining stages. Once mining begins, it shows processed queries, speed, and ETA. Add `--no-progress` to disable dynamic output in log jobs.

After identity, normalized-name, exact-body, and character-trigram false-negative filtering, each record receives up to `3 BM25 + 2 same_category + 1 random` candidate negatives.

## 6. Upload to AutoDL

Upload the framework and locally generated artifacts. The server can keep its existing raw dataset:

```powershell
scp -P 42112 -r configs scripts src requirements.txt requirements-train.txt README.md README_EN.md `
  root@connect.westb.seetacloud.com:/root/autodl-tmp/fcsr/
scp -P 42112 -r data/contracts data/synthetic `
  root@connect.westb.seetacloud.com:/root/autodl-tmp/fcsr/data/
```

## 7. AutoDL Setup and Model Download

On AutoDL:

```bash
cd /root/autodl-tmp/fcsr
pip install -r requirements-train.txt
mkdir -p /root/autodl-tmp/models
hf download Qwen/Qwen3-Embedding-0.6B --local-dir /root/autodl-tmp/models/Qwen3-Embedding-0.6B
hf download Qwen/Qwen3-Reranker-0.6B --local-dir /root/autodl-tmp/models/Qwen3-Reranker-0.6B
```

This is the only model download step. Do not run it on the local preprocessing machine.

## 8. Mine Semantic Negatives on AutoDL

```bash
python -B scripts/preprocess.py semantic-negatives \
  --local data/synthetic/local_negatives.jsonl \
  --skills data/raw/skills_easy.jsonl \
  --model /root/autodl-tmp/models/Qwen3-Embedding-0.6B \
  --output data/synthetic/train_biencoder.jsonl \
  --review data/processed/contract_fn_review.jsonl \
  --top-k 50 --threshold 0.95 --batch-size 8 --overwrite
```

The final target composition is up to `4 semantic + 3 BM25 + 2 same_category + 1 random`. Candidates with excessively high positive-candidate embedding similarity are filtered and exported for optional human review.

## 9. Train the Bi-Encoder

Validate the training data without loading a model:

```bash
python -B scripts/train_biencoder.py --dry-run
```

Train with the default RTX 4090-safe LoRA setup:

```bash
python -B scripts/train_biencoder.py \
  --model /root/autodl-tmp/models/Qwen3-Embedding-0.6B \
  --output-dir checkpoints/fcsr-emb-0.6b
```

Defaults: one epoch, micro-batch 1, gradient accumulation 16, BF16, gradient checkpointing, and InfoNCE temperature 0.05. During training, the terminal shows the current epoch, completed batches, live loss, speed, and ETA. Use `--method full` for an approach closer to SkillRouter but with higher memory and training costs.

## 10. Build Top-20 Groups and Train the Reranker

Retrieve the synthetic training queries with the trained Bi-Encoder:

```bash
python -B scripts/evaluate.py retrieve \
  --queries data/synthetic/queries.jsonl \
  --skills data/raw/skills_easy.jsonl \
  --model checkpoints/fcsr-emb-0.6b \
  --top-k 20 --batch-size 8 \
  --output-predictions data/processed/synthetic_top20.json \
  --output-records data/processed/synthetic_top20.jsonl
```

Prepare ordered candidate groups and validate them:

```bash
python -B scripts/train_reranker.py prepare \
  --retrieval data/processed/synthetic_top20.jsonl \
  --skills data/raw/skills_easy.jsonl \
  --output data/synthetic/train_reranker.jsonl --top-k 20 --overwrite
python -B scripts/train_reranker.py train --dry-run
```

Train:

```bash
python -B scripts/train_reranker.py train \
  --model /root/autodl-tmp/models/Qwen3-Reranker-0.6B \
  --output-dir checkpoints/fcsr-rank-0.6b
```

The listwise loss assigns probability mass to all valid positives in a group and rejects groups with no positive.

## 10.1 No-Training Retrieval Baselines

The following baselines use the same 75 tasks, Easy/Hard pools, and `description=300`, `body=2500` Skill caps as FCSR. Each method writes to its own report directory so summary files cannot overwrite one another.

```bash
mkdir -p reports/baselines/{bm25,dense,hybrid}

for tier in easy hard; do
  python -B scripts/evaluate.py bm25 \
    --queries data/raw/evaluation_queries.jsonl \
    --skills data/raw/skills_${tier}.jsonl --top-k 50 \
    --output-predictions reports/baselines/bm25/retrieval_${tier}.json \
    --output-records reports/baselines/bm25/retrieval_${tier}.jsonl
  python -B scripts/evaluate.py score \
    --tasks data/raw/evaluation_queries.jsonl \
    --skills data/raw/skills_${tier}.jsonl \
    --predictions reports/baselines/bm25/retrieval_${tier}.json \
    --stage retrieval --tier ${tier} --output-dir reports/baselines/bm25

  python -B scripts/evaluate.py retrieve \
    --queries data/raw/evaluation_queries.jsonl \
    --skills data/raw/skills_${tier}.jsonl \
    --model /root/autodl-tmp/models/Qwen3-Embedding-0.6B \
    --top-k 50 --batch-size 8 --skill-max-length 2048 \
    --output-predictions reports/baselines/dense/retrieval_${tier}.json \
    --output-records reports/baselines/dense/retrieval_${tier}.jsonl
  python -B scripts/evaluate.py score \
    --tasks data/raw/evaluation_queries.jsonl \
    --skills data/raw/skills_${tier}.jsonl \
    --predictions reports/baselines/dense/retrieval_${tier}.json \
    --stage retrieval --tier ${tier} --output-dir reports/baselines/dense

  python -B scripts/evaluate.py hybrid \
    --queries data/raw/evaluation_queries.jsonl \
    --skills data/raw/skills_${tier}.jsonl \
    --model /root/autodl-tmp/models/Qwen3-Embedding-0.6B \
    --top-k 50 --fusion-depth 100 --rrf-k 60 \
    --batch-size 8 --skill-max-length 2048 \
    --output-predictions reports/baselines/hybrid/retrieval_${tier}.json \
    --output-records reports/baselines/hybrid/retrieval_${tier}.jsonl
  python -B scripts/evaluate.py score \
    --tasks data/raw/evaluation_queries.jsonl \
    --skills data/raw/skills_${tier}.jsonl \
    --predictions reports/baselines/hybrid/retrieval_${tier}.json \
    --stage retrieval --tier ${tier} --output-dir reports/baselines/hybrid
done
```

BM25 does not use a GPU. Dense and Hybrid use the base Qwen3-Embedding-0.6B model with no FCSR LoRA. Hybrid combines each method's Top-100 rankings with fixed reciprocal-rank fusion (`RRF k=60`); this parameter is not tuned on the 75 test tasks.
## 11. Evaluate Easy and Hard

For each pool, export retrieval Top-50, rerank its first 20 candidates, and then score the predictions. Example for Easy:

```bash
python -B scripts/evaluate.py retrieve \
  --queries data/raw/evaluation_queries.jsonl \
  --skills data/raw/skills_easy.jsonl \
  --model checkpoints/fcsr-emb-0.6b --top-k 50 \
  --output-predictions reports/retrieval_easy.json \
  --output-records reports/retrieval_easy.jsonl
python -B scripts/evaluate.py rerank \
  --retrieval-records reports/retrieval_easy.jsonl \
  --skills data/raw/skills_easy.jsonl \
  --model checkpoints/fcsr-rank-0.6b --top-k 20 \
  --output-predictions reports/reranker_easy.json \
  --output-records reports/reranker_easy.jsonl
python -B scripts/evaluate.py score \
  --tasks data/raw/evaluation_queries.jsonl \
  --skills data/raw/skills_easy.jsonl \
  --predictions reports/retrieval_easy.json \
  --stage retrieval --tier easy
python -B scripts/evaluate.py score \
  --tasks data/raw/evaluation_queries.jsonl \
  --skills data/raw/skills_easy.jsonl \
  --predictions reports/reranker_easy.json \
  --stage reranker --tier easy
```

For the Hard pool, replace the skill file with `skills_hard.jsonl`, set `--tier hard`, and change `*_easy.*` output names to `*_hard.*`.

Scoring follows SkillRouter: exclude `generic_only` tasks, intersect GT/relevance with the current skill pool, use graded relevance for nDCG, and report overall, single-skill, multi-skill, and FullCoverage metrics.

## Method Scope

The public [SkillRouter repository](https://github.com/zhengyanzhao1997/SkillRouter) releases the benchmark plus inference and evaluation code under the MIT license, but does not release its training preprocessing scripts. FCSR preserves its public data formats, pooling method, metrics, Top-20 candidate groups, and two-stage training conventions. The mining and training code in this project reimplements the paper-described procedure.

Budget-aware deviations from the paper:

1. Sample 8,000 positives instead of generating 37,979 synthetic training pairs.
2. Use LoRA by default while retaining the more expensive full fine-tuning option.
3. Extract Contracts only for sampled positives; suspicious false-negative pairs are exported for human review instead of invoking another paid LLM judge.
### Reranker Memory Preflight

Before the training epoch begins, `train_reranker.py train` scans every candidate group, selects the group with the longest tokenized sequence, and runs one forward/backward preflight without updating parameters. After the preflight passes, the training flow is unchanged. If it OOMs, lower `--max-length` and run again.

For a 24GB RTX 4090:

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -B scripts/train_reranker.py train \
  --model /root/autodl-tmp/models/Qwen3-Reranker-0.6B \
  --output-dir checkpoints/fcsr-rank-0.6b \
  --max-length 1536
```

# FCSR

FCSR (Full-Coverage Skill Retrieval) trains an auditable two-stage retriever for large Agent Skill libraries:

```text
Task Query
  -> BM25 + FCSR Retriever
  -> Reciprocal Rank Fusion (RRF)
  -> FCSR Reranker
  -> Top-10 Skills
```

The repository maintains the frozen Hard-pool evaluation only. See the [RQ1 document](docs/rq1-skill-retrieval.md) for the method, data statistics, and results, or [README.zh-CN.md](README.zh-CN.md) for Chinese instructions.

## Final artifacts

Two systems are retained:

- `FCSR`: the official model trained on the complete dataset, under `checkpoints/fcsr/`.
- `FCSR-Small`: the earlier small-data ablation, under `checkpoints/fcsr-small/`.

Each system contains a Qwen3-Embedding-0.6B LoRA `retriever/` and a Qwen3-Reranker-0.6B LoRA `reranker/`.

Results are organized as follows:

- `reports/baselines/hard/`: BM25, Dense, RRF, Base Reranker, and SkillRouter.
- `reports/systems/{fcsr,fcsr-small}/hard/`: stage outputs for FCSR systems.
- `reports/tables/hard-retrieval.md`: first-stage comparison.
- `reports/tables/hard-final.md`: final-system comparison.
- `reports/tables/hard-two-stage.md`: two-stage ablation.

## Layout

```text
configs/fcsr.yaml       single training configuration
data/                   raw, synthetic, and mixed training data
jobs/                   production Slurm jobs
scripts/                data, training, and evaluation entry points
src/                    core implementation
tests/                  unit tests
checkpoints/            FCSR and FCSR-Small
reports/                Hard-pool outputs and tables
```

`data/raw/skills_easy.jsonl.gz` remains the training and negative-mining pool. It is not evaluated; final evaluation uses `data/raw/skills_hard.jsonl.gz` only.

## Environment

Run every formal workload through Slurm and use a project-local environment:

```bash
conda create -p ./env-qgen python=3.10 -y
conda activate ./env-qgen
pip install -r requirements.txt
```

Store the DeepSeek key in the untracked `.env` file:

```text
DEEPSEEK_API_KEY=...
```

Generation uses `deepseek-v4-flash` with 16 concurrent requests by default.

## Data pipeline

This is the single supported reproduction order. Inspect each job before submitting the next one.

```bash
sbatch jobs/extract_contracts_deepseek.sbatch
sbatch jobs/generate_single_skill_deepseek.sbatch
sbatch jobs/mine_single_skill_local_negatives.sbatch
sbatch jobs/mine_single_skill_semantic_negatives.sbatch
sbatch jobs/prepare_single_skill_reranker.sbatch
sbatch jobs/build_multi_skill_candidates.sbatch
sbatch jobs/generate_multiskill_deepseek.sbatch
sbatch jobs/build_mixed_training_data.sbatch
```

To validate multi-Skill generation on a small subset, limit the formal command without creating a separate pilot tree:

```bash
sbatch --export=ALL,LIMIT=50 jobs/generate_multiskill_deepseek.sbatch
```

Negative mining combines BM25, same-category, random, and semantic candidates. Every candidate is checked against all positives; removed false-negative candidates are recorded in `semantic_fn_review.jsonl.gz`.

## Training

All official hyperparameters live in `configs/fcsr.yaml`; both components use LoRA:

```bash
sbatch jobs/train_fcsr_retriever.sbatch
sbatch jobs/train_fcsr_reranker.sbatch
```

The default output paths already contain the final checkpoints. Set a new `OUTPUT_DIR` for reproduction; the jobs refuse to overwrite non-empty directories.

## Hard-pool evaluation

Run each first-stage method before its reranker:

```bash
sbatch --export=ALL,FCSR_SYSTEM=fcsr,RETRIEVAL_MODE=dense jobs/evaluate_fcsr_retrieval.sbatch
sbatch --export=ALL,FCSR_SYSTEM=fcsr,RETRIEVAL_MODE=rrf jobs/evaluate_fcsr_retrieval.sbatch
sbatch --export=ALL,FCSR_SYSTEM=fcsr,RETRIEVAL_MODE=dense jobs/evaluate_fcsr_reranker.sbatch
sbatch --export=ALL,FCSR_SYSTEM=fcsr,RETRIEVAL_MODE=rrf jobs/evaluate_fcsr_reranker.sbatch
```

Use `FCSR_SYSTEM=fcsr-small` for the scale ablation. The frozen protocol uses first-stage Top-50, RRF depth 100, `rrf_k=60`, reranker Top-20, and final Top-10 Skills.

Render the canonical tables with:

```bash
sbatch jobs/render_reports.sbatch
```

## Validation

```bash
bash -n jobs/*.sbatch
PYTHONPATH=.:src ./env-qgen/bin/python -m unittest discover -s tests -v
```

Git excludes large data, checkpoints, logs, and reports. Small manifests remain tracked to preserve provenance and validation metadata.

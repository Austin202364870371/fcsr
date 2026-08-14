#!/bin/bash
#SBATCH --job-name=fcsr-mixed-data
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --exclude=GPU0,GPU1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/mixed-data-%j.out
#SBATCH --error=logs/mixed-data-%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$PWD}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/data/training/mixed}"
MODEL_DIR="${MODEL_DIR:-$PROJECT_DIR/models/Qwen3-Embedding-0.6B}"

cd "$PROJECT_DIR"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$PROJECT_DIR/env-qgen"
mkdir -p "$PROJECT_DIR/logs" "$OUTPUT_DIR"

python -B scripts/build_multiskill_training_data.py \
  --single-biencoder data/synthetic/single_skill/train_biencoder.jsonl.gz \
  --single-reranker data/synthetic/single_skill/train_reranker.jsonl.gz \
  --multiskill-queries data/synthetic/multi_skill/queries.jsonl.gz \
  --skills data/raw/skills_easy.jsonl.gz \
  --negative-model "$MODEL_DIR" \
  --semantic-top-k 64 \
  --semantic-fn-threshold 0.95 \
  --semantic-review "$OUTPUT_DIR/semantic_fn_review.jsonl.gz" \
  --biencoder-multi-loss-weight 1.5 \
  --reranker-multi-loss-weight 3.0 \
  --seed 42 \
  --batch-size 8 \
  --query-max-length 512 \
  --skill-max-length 2048 \
  --device cuda \
  --output-dir "$OUTPUT_DIR" \
  --overwrite

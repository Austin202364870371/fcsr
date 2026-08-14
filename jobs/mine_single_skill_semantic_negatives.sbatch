#!/bin/bash
#SBATCH --job-name=fcsr-single-sem-neg
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --exclude=GPU0,GPU1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/single-sem-neg-%j.out
#SBATCH --error=logs/single-sem-neg-%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$PWD}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/data/synthetic/single_skill}"
MODEL_DIR="${MODEL_DIR:-$PROJECT_DIR/models/Qwen3-Embedding-0.6B}"

cd "$PROJECT_DIR"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$PROJECT_DIR/env-qgen"
mkdir -p "$PROJECT_DIR/logs" "$OUTPUT_DIR"

python -B scripts/build_single_skill_data.py semantic-negatives \
  --local "$OUTPUT_DIR/local_negatives.jsonl.gz" \
  --skills data/raw/skills_easy.jsonl.gz \
  --output "$OUTPUT_DIR/train_biencoder.jsonl.gz" \
  --review "$OUTPUT_DIR/semantic_fn_review.jsonl.gz" \
  --model "$MODEL_DIR" \
  --top-k 50 \
  --threshold 0.95 \
  --batch-size 8 \
  --query-max-length 512 \
  --skill-max-length 2048 \
  --device cuda \
  --overwrite

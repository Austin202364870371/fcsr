#!/bin/bash
#SBATCH --job-name=fcsr-single-reranker-data
#SBATCH --qos=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=logs/single-reranker-data-%j.out
#SBATCH --error=logs/single-reranker-data-%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$PWD}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/data/synthetic/single_skill}"

cd "$PROJECT_DIR"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$PROJECT_DIR/env-qgen"
mkdir -p "$PROJECT_DIR/logs" "$OUTPUT_DIR"

python -B scripts/build_single_skill_data.py reranker-groups \
  --retrieval "$OUTPUT_DIR/train_biencoder.jsonl.gz" \
  --skills data/raw/skills_easy.jsonl.gz \
  --output "$OUTPUT_DIR/train_reranker.jsonl.gz" \
  --top-k 20 \
  --overwrite

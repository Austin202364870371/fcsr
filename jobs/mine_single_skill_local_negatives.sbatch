#!/bin/bash
#SBATCH --job-name=fcsr-single-local-neg
#SBATCH --qos=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/single-local-neg-%j.out
#SBATCH --error=logs/single-local-neg-%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$PWD}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/data/synthetic/single_skill}"

cd "$PROJECT_DIR"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$PROJECT_DIR/env-qgen"
mkdir -p "$PROJECT_DIR/logs" "$OUTPUT_DIR"

python -B scripts/build_single_skill_data.py local-negatives \
  --queries "$OUTPUT_DIR/queries.jsonl.gz" \
  --skills data/raw/skills_easy.jsonl.gz \
  --output "$OUTPUT_DIR/local_negatives.jsonl.gz" \
  --seed 42 \
  --overlap-threshold 0.85 \
  --overwrite

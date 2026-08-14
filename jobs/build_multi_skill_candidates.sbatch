#!/bin/bash
#SBATCH --job-name=fcsr-multi-candidates
#SBATCH --qos=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/multi-candidates-%j.out
#SBATCH --error=logs/multi-candidates-%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$PWD}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/data/synthetic/multi_skill}"

cd "$PROJECT_DIR"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$PROJECT_DIR/env-qgen"
mkdir -p "$PROJECT_DIR/logs" "$OUTPUT_DIR"

python -B scripts/build_multiskill_candidates.py \
  --contracts data/contracts/contracts.jsonl.gz \
  --queries data/synthetic/single_skill/queries.jsonl.gz \
  --tasks data/raw/evaluation_queries.jsonl.gz \
  --output "$OUTPUT_DIR/candidates.jsonl.gz" \
  --rejections "$OUTPUT_DIR/candidate_rejections.jsonl.gz" \
  --manifest "$OUTPUT_DIR/manifest.json" \
  --overwrite

#!/bin/bash
#SBATCH --job-name=fcsr-single-qgen
#SBATCH --qos=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=08:00:00
#SBATCH --output=logs/single-qgen-%j.out
#SBATCH --error=logs/single-qgen-%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$PWD}"
SAMPLE_DIR="${SAMPLE_DIR:-$PROJECT_DIR/data/contracts_32k}"
CONTRACT_DIR="${CONTRACT_DIR:-$PROJECT_DIR/data/contracts_32k_prompt006}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/data/synthetic/single_skill_v1}"
CONCURRENCY="${CONCURRENCY:-16}"

cd "$PROJECT_DIR"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$PROJECT_DIR/env-qgen"
mkdir -p "$PROJECT_DIR/logs" "$OUTPUT_DIR"

python -B scripts/build_single_skill_data.py queries \
  --model deepseek-v4-flash \
  --sample "$SAMPLE_DIR/sample_skills.jsonl.gz" \
  --contracts "$CONTRACT_DIR/contracts.jsonl.gz" \
  --output "$OUTPUT_DIR/queries.jsonl.gz" \
  --failures "$OUTPUT_DIR/query_failures.jsonl.gz" \
  --concurrency "$CONCURRENCY" \
  --max-new-tokens 1024 \
  --max-attempts 3 \
  --backoff 2

#!/bin/bash
#SBATCH --job-name=fcsr-contracts-ds-pilot
#SBATCH --qos=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/contracts-ds-pilot-%j.out
#SBATCH --error=logs/contracts-ds-pilot-%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$PWD}"
SAMPLE_DIR="${SAMPLE_DIR:-$PROJECT_DIR/data/contracts_32k}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/data/contracts_deepseek_prompt004_pilot}"
LIMIT="${LIMIT:-32}"
CONCURRENCY="${CONCURRENCY:-16}"

cd "$PROJECT_DIR"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$PROJECT_DIR/env-qgen"
mkdir -p "$PROJECT_DIR/logs" "$OUTPUT_DIR"

if [[ ! -f "$SAMPLE_DIR/sample_skills.jsonl.gz" ]]; then
  python -B scripts/build_single_skill_data.py sample \
    --sample-size 32000 \
    --output-dir "$SAMPLE_DIR" \
    --overwrite
fi

python -B scripts/build_single_skill_data.py contracts \
  --model deepseek-v4-flash \
  --sample "$SAMPLE_DIR/sample_skills.jsonl.gz" \
  --output "$OUTPUT_DIR/contracts.jsonl.gz" \
  --failures "$OUTPUT_DIR/failures.jsonl.gz" \
  --concurrency "$CONCURRENCY" \
  --limit "$LIMIT" \
  --max-new-tokens 6144 \
  --max-attempts 3 \
  --backoff 2

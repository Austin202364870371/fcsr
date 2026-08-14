#!/bin/bash
#SBATCH --job-name=fcsr-contracts-deepseek
#SBATCH --qos=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=logs/contracts-deepseek-%j.out
#SBATCH --error=logs/contracts-deepseek-%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$PWD}"
SAMPLE_DIR="${SAMPLE_DIR:-$PROJECT_DIR/data/samples}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/data/contracts}"
CONCURRENCY="${CONCURRENCY:-16}"

cd "$PROJECT_DIR"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$PROJECT_DIR/env-qgen"
mkdir -p "$PROJECT_DIR/logs" "$SAMPLE_DIR" "$OUTPUT_DIR"

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
  --max-new-tokens 6144 \
  --max-attempts 3 \
  --backoff 2

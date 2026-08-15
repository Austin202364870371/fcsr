#!/bin/bash
#SBATCH --job-name=fcsr-qgen-deepseek
#SBATCH --qos=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=06:00:00
#SBATCH --output=logs/qgen-deepseek-%j.out
#SBATCH --error=logs/qgen-deepseek-%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$PWD}"
CANDIDATE_DIR="${CANDIDATE_DIR:-$PROJECT_DIR/data/synthetic/multi_skill}"
CONTRACT_DIR="${CONTRACT_DIR:-$PROJECT_DIR/data/contracts}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/data/synthetic/multi_skill}"
LIMIT="${LIMIT:-}"
OVERWRITE="${OVERWRITE:-0}"
CONCURRENCY="${CONCURRENCY:-16}"

cd "$PROJECT_DIR"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$PROJECT_DIR/env-qgen"
mkdir -p "$PROJECT_DIR/logs" "$OUTPUT_DIR"

arguments=(
  --candidates "$CANDIDATE_DIR/candidates.jsonl.gz"
  --contracts "$CONTRACT_DIR/contracts.jsonl.gz"
  --output "$OUTPUT_DIR/queries.jsonl.gz"
  --failures "$OUTPUT_DIR/query_failures.jsonl.gz"
  --review-queue "$OUTPUT_DIR/review_queue.jsonl.gz"
  --manifest "$OUTPUT_DIR/manifest.json"
  --model deepseek-v4-flash
  --concurrency "$CONCURRENCY"
  --max-attempts 3
)
if [[ -n "$LIMIT" ]]; then
  arguments+=(--limit "$LIMIT")
fi
if [[ "$OVERWRITE" == "1" ]]; then
  arguments+=(--overwrite)
fi

python -B scripts/generate_multiskill_queries.py "${arguments[@]}"

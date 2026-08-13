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
LIMIT="${LIMIT:-50}"
OVERWRITE="${OVERWRITE:-0}"

cd "$PROJECT_DIR"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$PROJECT_DIR/env-qgen"
mkdir -p "$PROJECT_DIR/logs"

arguments=(
  --model deepseek-v4-flash
  --concurrency 16
  --limit "$LIMIT"
)
if [[ "$OVERWRITE" == "1" ]]; then
  arguments+=(--overwrite)
fi

python -B scripts/generate_multiskill_queries.py "${arguments[@]}"

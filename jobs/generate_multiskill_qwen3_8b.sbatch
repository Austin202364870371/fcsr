#!/bin/bash
#SBATCH --job-name=fcsr-qgen-pilot
#SBATCH --qos=fast
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/data-nfs/gpu3/u13256401368/fcsr/logs/qgen-pilot-%j.out
#SBATCH --error=/data-nfs/gpu3/u13256401368/fcsr/logs/qgen-pilot-%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/data-nfs/gpu3/u13256401368/fcsr}"
LIMIT="${LIMIT:-50}"
OVERWRITE="${OVERWRITE:-0}"

cd "$PROJECT_DIR"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$PROJECT_DIR/env-qgen"

export HF_HOME="$PROJECT_DIR/.cache/huggingface"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

mkdir -p "$PROJECT_DIR/logs"

arguments=(
  --model "$PROJECT_DIR/models/Qwen3-8B"
  --limit "$LIMIT"
)
if [[ "$OVERWRITE" == "1" ]]; then
  arguments+=(--overwrite)
fi

python -B scripts/generate_multiskill_queries.py "${arguments[@]}"

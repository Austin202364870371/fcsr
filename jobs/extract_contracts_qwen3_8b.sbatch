#!/bin/bash
#SBATCH --job-name=fcsr-contracts-qwen3-8b
#SBATCH --qos=fast
#SBATCH --nodelist=GPU2,GPU3,GPU4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=logs/contracts-qwen3-8b-%j.out
#SBATCH --error=logs/contracts-qwen3-8b-%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$PWD}"
MODEL_PATH="${MODEL_PATH:-$PROJECT_DIR/models/Qwen3-8B}"
SAMPLE_PATH="${SAMPLE_PATH:-$PROJECT_DIR/data/contracts/sample_skills.jsonl.gz}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/data/contracts_local_qwen3_8b}"
LIMIT="${LIMIT:-32}"

cd "$PROJECT_DIR"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$PROJECT_DIR/env-qgen"

export HF_HOME="$PROJECT_DIR/.cache/huggingface"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

mkdir -p "$PROJECT_DIR/logs" "$OUTPUT_DIR"

arguments=(
  contracts
  --provider local
  --model "$MODEL_PATH"
  --device cuda
  --sample "$SAMPLE_PATH"
  --output "$OUTPUT_DIR/contracts.jsonl.gz"
  --failures "$OUTPUT_DIR/failures.jsonl.gz"
  --max-new-tokens 3072
  --max-attempts 3
  --backoff 0
)
if [[ -n "$LIMIT" ]]; then
  arguments+=(--limit "$LIMIT")
fi

python -B scripts/build_single_skill_data.py "${arguments[@]}"

#!/bin/bash
#SBATCH --job-name=fcsr-train-reranker
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --exclude=GPU0,GPU1,GPU4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/train-reranker-%j.out
#SBATCH --error=logs/train-reranker-%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$PWD}"
CONFIG="${CONFIG:-$PROJECT_DIR/configs/fcsr.yaml}"
TRAIN_GROUPS="${TRAIN_GROUPS:-$PROJECT_DIR/data/training/reranker.jsonl.gz}"
SKILLS="${SKILLS:-$PROJECT_DIR/data/raw/skills_easy.jsonl.gz}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/checkpoints/fcsr/reranker}"
SEED="${SEED:-42}"

cd "$PROJECT_DIR"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$PROJECT_DIR/env-qgen"

for path in "$CONFIG" "$TRAIN_GROUPS" "$SKILLS"; do
  if [[ ! -f "$path" ]]; then
    echo "Required input file does not exist: $path" >&2
    exit 1
  fi
done
if [[ -d "$OUTPUT_DIR" ]] && [[ -n "$(find "$OUTPUT_DIR" -mindepth 1 -print -quit)" ]]; then
  echo "Checkpoint directory is not empty; choose a new OUTPUT_DIR: $OUTPUT_DIR" >&2
  exit 1
fi

export HF_HOME="${HF_HOME:-$PROJECT_DIR/.cache/huggingface}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
mkdir -p "$PROJECT_DIR/logs" "$HF_HOME" "$OUTPUT_DIR"

echo "Starting FCSR reranker training"
echo "node=$(hostname) output=$OUTPUT_DIR"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python -u -B scripts/train_reranker.py \
  --config "$CONFIG" \
  --groups "$TRAIN_GROUPS" \
  --skills "$SKILLS" \
  --output-dir "$OUTPUT_DIR" \
  --seed "$SEED"

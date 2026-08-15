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
CONFIG="${CONFIG:-$PROJECT_DIR/configs/model_qwen3_0_6b.yaml}"
TRAIN_GROUPS="${TRAIN_GROUPS:-$PROJECT_DIR/data/training/reranker.jsonl.gz}"
SKILLS="${SKILLS:-$PROJECT_DIR/data/raw/skills_easy.jsonl.gz}"
MODEL_DIR="${MODEL_DIR:-$PROJECT_DIR/models/Qwen3-Reranker-0.6B}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/checkpoints/fcsr-rank-0.6b-multiskill-weighted}"
METHOD="${METHOD:-lora}"
EPOCHS="${EPOCHS:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-16}"
LEARNING_RATE="${LEARNING_RATE:-0.00001}"
MAX_LENGTH="${MAX_LENGTH:-4096}"
PRECISION="${PRECISION:-bf16}"
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
if [[ ! -d "$MODEL_DIR" ]]; then
  echo "Local reranker model directory does not exist: $MODEL_DIR" >&2
  exit 1
fi
if [[ -d "$OUTPUT_DIR" ]] && [[ -n "$(find "$OUTPUT_DIR" -mindepth 1 -print -quit)" ]]; then
  echo "Checkpoint directory is not empty; choose a new OUTPUT_DIR: $OUTPUT_DIR" >&2
  exit 1
fi

export HF_HOME="${HF_HOME:-$PROJECT_DIR/.cache/huggingface}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
mkdir -p "$PROJECT_DIR/logs" "$HF_HOME" "$OUTPUT_DIR"

echo "Starting FCSR reranker training"
echo "node=$(hostname) model=$MODEL_DIR output=$OUTPUT_DIR"
echo "epochs=$EPOCHS accumulation=$GRADIENT_ACCUMULATION_STEPS max_length=$MAX_LENGTH precision=$PRECISION"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python -u -B scripts/train_reranker.py train \
  --config "$CONFIG" \
  --groups "$TRAIN_GROUPS" \
  --skills "$SKILLS" \
  --model "$MODEL_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --method "$METHOD" \
  --epochs "$EPOCHS" \
  --gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS" \
  --learning-rate "$LEARNING_RATE" \
  --max-length "$MAX_LENGTH" \
  --precision "$PRECISION" \
  --seed "$SEED"

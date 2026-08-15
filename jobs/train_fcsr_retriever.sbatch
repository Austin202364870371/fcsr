#!/bin/bash
#SBATCH --job-name=fcsr-train-biencoder
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --exclude=GPU0,GPU1,GPU4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/train-fcsr-retriever-%j.out
#SBATCH --error=logs/train-fcsr-retriever-%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$PWD}"
CONFIG="${CONFIG:-$PROJECT_DIR/configs/model_qwen3_0_6b.yaml}"
TRAIN_DATA="${TRAIN_DATA:-$PROJECT_DIR/data/training/biencoder.jsonl.gz}"
SKILLS="${SKILLS:-$PROJECT_DIR/data/raw/skills_easy.jsonl.gz}"
MODEL_DIR="${MODEL_DIR:-$PROJECT_DIR/models/Qwen3-Embedding-0.6B}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/checkpoints/fcsr/retriever}"
METHOD="${METHOD:-lora}"
EPOCHS="${EPOCHS:-1}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-4}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"
LEARNING_RATE="${LEARNING_RATE:-0.00002}"
TEMPERATURE="${TEMPERATURE:-0.05}"
QUERY_MAX_LENGTH="${QUERY_MAX_LENGTH:-512}"
SKILL_MAX_LENGTH="${SKILL_MAX_LENGTH:-2048}"
PRECISION="${PRECISION:-bf16}"
SEED="${SEED:-42}"

cd "$PROJECT_DIR"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$PROJECT_DIR/env-qgen"

for path in "$CONFIG" "$TRAIN_DATA" "$SKILLS"; do
  if [[ ! -f "$path" ]]; then
    echo "Required input file does not exist: $path" >&2
    exit 1
  fi
done
if [[ ! -d "$MODEL_DIR" ]]; then
  echo "Local embedding model directory does not exist: $MODEL_DIR" >&2
  exit 1
fi
if [[ -d "$OUTPUT_DIR" ]] && [[ -n "$(find "$OUTPUT_DIR" -mindepth 1 -print -quit)" ]]; then
  echo "Checkpoint directory is not empty; choose a new OUTPUT_DIR: $OUTPUT_DIR" >&2
  exit 1
fi

export HF_HOME="${HF_HOME:-$PROJECT_DIR/.cache/huggingface}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
mkdir -p "$PROJECT_DIR/logs" "$HF_HOME" "$OUTPUT_DIR"

echo "Starting FCSR retriever training"
echo "node=$(hostname) model=$MODEL_DIR output=$OUTPUT_DIR"
echo "epochs=$EPOCHS micro_batch=$MICRO_BATCH_SIZE accumulation=$GRADIENT_ACCUMULATION_STEPS precision=$PRECISION"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python -u -B scripts/train_biencoder.py \
  --config "$CONFIG" \
  --train-data "$TRAIN_DATA" \
  --skills "$SKILLS" \
  --model "$MODEL_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --method "$METHOD" \
  --epochs "$EPOCHS" \
  --micro-batch-size "$MICRO_BATCH_SIZE" \
  --gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS" \
  --learning-rate "$LEARNING_RATE" \
  --temperature "$TEMPERATURE" \
  --query-max-length "$QUERY_MAX_LENGTH" \
  --skill-max-length "$SKILL_MAX_LENGTH" \
  --precision "$PRECISION" \
  --seed "$SEED"

#!/bin/bash
#SBATCH --job-name=fcsr-eval-dense
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --exclude=GPU0,GPU1,GPU4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/eval-hard-fcsr-dense-%j.out
#SBATCH --error=logs/eval-hard-fcsr-dense-%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$PWD}"
QUERIES="${QUERIES:-$PROJECT_DIR/data/raw/evaluation_queries.jsonl.gz}"
SKILLS="${SKILLS:-$PROJECT_DIR/data/raw/skills_hard.jsonl.gz}"
MODEL_DIR="${MODEL_DIR:-$PROJECT_DIR/checkpoints/fcsr-emb-0.6b-multiskill-weighted}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/reports/retrieval/hard/fcsr-multiskill-weighted}"
BATCH_SIZE="${BATCH_SIZE:-8}"

cd "$PROJECT_DIR"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$PROJECT_DIR/env-qgen"

for path in "$QUERIES" "$SKILLS"; do
  if [[ ! -f "$path" ]]; then
    echo "Required input file does not exist: $path" >&2
    exit 1
  fi
done
if [[ ! -f "$MODEL_DIR/adapter_config.json" ]] || [[ ! -f "$MODEL_DIR/adapter_model.safetensors" ]]; then
  echo "FCSR embedding checkpoint is incomplete: $MODEL_DIR" >&2
  exit 1
fi
if [[ -d "$OUTPUT_DIR" ]] && [[ -n "$(find "$OUTPUT_DIR" -mindepth 1 -print -quit)" ]]; then
  echo "Evaluation output directory is not empty: $OUTPUT_DIR" >&2
  exit 1
fi

export HF_HOME="${HF_HOME:-$PROJECT_DIR/.cache/huggingface}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
unset TRANSFORMERS_CACHE
mkdir -p "$PROJECT_DIR/logs" "$HF_HOME" "$OUTPUT_DIR"

echo "Evaluating FCSR dense retrieval on the Hard pool"
echo "node=$(hostname) model=$MODEL_DIR output=$OUTPUT_DIR batch_size=$BATCH_SIZE"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python -u -B scripts/evaluate.py retrieve \
  --queries "$QUERIES" \
  --skills "$SKILLS" \
  --model "$MODEL_DIR" \
  --top-k 50 \
  --batch-size "$BATCH_SIZE" \
  --query-max-length 512 \
  --skill-max-length 2048 \
  --device cuda \
  --output-predictions "$OUTPUT_DIR/predictions.json" \
  --output-records "$OUTPUT_DIR/records.jsonl"

python -u -B scripts/evaluate.py score \
  --tasks "$QUERIES" \
  --skills "$SKILLS" \
  --predictions "$OUTPUT_DIR/predictions.json" \
  --stage retrieval \
  --tier hard \
  --output-dir "$OUTPUT_DIR"

echo "Dense retrieval evaluation completed: $OUTPUT_DIR"

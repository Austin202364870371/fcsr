#!/bin/bash
#SBATCH --job-name=fcsr-render-reports
#SBATCH --qos=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=logs/render-reports-%j.out
#SBATCH --error=logs/render-reports-%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$PWD}"

cd "$PROJECT_DIR"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$PROJECT_DIR/env-qgen"

mkdir -p "$PROJECT_DIR/logs" "$PROJECT_DIR/reports/tables"

PYTHONPATH=src python -u -B scripts/render_evaluation_tables.py \
  --reports-dir reports \
  --output-dir reports/tables

echo "Canonical Hard-pool reports rendered under reports/tables"

#!/bin/bash
#SBATCH --job-name=hubert_rawnet2_eval
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

# Usage:
#   sbatch scripts/run_hubert_rawnet2.sbatch /absolute/path/to/signal-laundering-deepfake
# If no path is passed, uses current directory.

PROJECT_DIR="${1:-$PWD}"
cd "$PROJECT_DIR"

mkdir -p logs

module purge
module load python/3.11.14

if [ ! -d ".venv" ]; then
  uv venv .venv
fi

source .venv/bin/activate

uv pip install --upgrade pip
uv pip install -r requirements.txt

# Optional but recommended for stable cache location on HPC.
export HF_HOME="${HF_HOME:-$PROJECT_DIR/.cache/huggingface}"
mkdir -p "$HF_HOME"

python run.py \
  --model rawnet2 \
  --frontend_model hubert \
  --frontend_ckpt facebook/hubert-base-ls960 \
  --data_root data/LA/LA \
  --depth 0

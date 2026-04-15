#!/bin/bash
#SBATCH --job-name=eval_suite
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j_%a.out
#SBATCH --error=logs/%x_%j_%a.err

# ─────────────────────────────────────────────────────────────────────────────
# Usage:
#   sbatch scripts/run_eval.sbatch aasist
#   sbatch scripts/run_eval.sbatch wav2vec2 --run_cka
#   sbatch scripts/run_eval.sbatch rawnet2
#
# All arguments after the model name are forwarded to eval_suite.py verbatim.
# Example with custom subset for testing:
#   sbatch scripts/run_eval.sbatch aasist --max_eval 500
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

MODEL="${1:?ERROR: pass model name as first argument, e.g. aasist}"
# All remaining args forwarded to eval_suite.py
EXTRA_ARGS="${@:2}"

# Resolve project directory (script is in PROJECT_DIR/scripts/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

mkdir -p logs

module purge
module load python/3.11.14   # adjust to your HPC module name

# Create venv if needed
if [ ! -d ".venv" ]; then
    uv venv .venv
fi
source .venv/bin/activate

# Install / sync dependencies
uv pip install --upgrade pip
uv pip install -r requirements.txt

# Hugging Face cache — keep inside project dir to avoid home quota issues
export HF_HOME="${HF_HOME:-$PROJECT_DIR/.cache/huggingface}"
mkdir -p "$HF_HOME"

echo "========================================"
echo "  Model:      $MODEL"
echo "  Extra args: $EXTRA_ARGS"
echo "  GPU:        $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "  Node:       $(hostname)"
echo "  Start:      $(date)"
echo "========================================"

python eval_suite.py \
    --model "$MODEL" \
    --data_root data/ASVspoof2019/LA \
    $EXTRA_ARGS

echo "========================================"
echo "  Done:  $(date)"
echo "========================================"

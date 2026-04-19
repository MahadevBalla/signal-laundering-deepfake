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

PROJECT_DIR="/home/mahadevballa23/scratch_space/signal-laundering-deepfake"
cd "$PROJECT_DIR"

if [ ! -d "logs" ]; then
    echo "ERROR: logs directory is missing. Create it manually before submitting this job."
    exit 1
fi

module purge
module load python/3.11.14   # adjust to your HPC module name

if [ ! -f ".venv/bin/activate" ]; then
    echo "ERROR: missing virtual environment at $PROJECT_DIR/.venv"
    echo "Create it manually and install requirements before submitting this job."
    exit 1
fi
source .venv/bin/activate

# Hugging Face cache — keep inside project dir to avoid home quota issues
export HF_HOME="${HF_HOME:-$PROJECT_DIR/.cache/huggingface}"
if [ ! -d "$HF_HOME" ]; then
    echo "ERROR: HF_HOME directory '$HF_HOME' is missing."
    echo "Create it manually or set HF_HOME to an existing directory before submitting this job."
    exit 1
fi

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

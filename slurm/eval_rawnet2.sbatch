#!/bin/bash
#SBATCH --job-name=eval_rawnet2
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=logs/eval_rawnet2_%j.out
#SBATCH --error=logs/eval_rawnet2_%j.err

# ─────────────────────────────────────────────────────────────────────────────
# Full laundering evaluation grid for RawNet2 (raw audio).
# Runs all 3 pipelines (N, M, P) x 3 strengths (L, M, H) x 4 depths (0-3).
#
# Outputs -> outputs/eval_suite/rawnet2/
#
# Submit:
#   sbatch slurm/eval_rawnet2.sbatch
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

EXTRA_ARGS="${@:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

mkdir -p logs

module purge
module load python/3.11.14

if [ ! -d ".venv" ]; then
    python -m venv .venv
fi
source .venv/bin/activate
pip install --upgrade pip -q
pip install -r requirements.txt -q
if [ ! -f "external/aasist/main.py" ]; then
    echo "ERROR: external/aasist is missing. Submodules must be initialized before submitting this job."
    echo "Initialize submodules outside this job script before submission."
    exit 1
fi

export HF_HOME="${HF_HOME:-$PROJECT_DIR/.cache/huggingface}"
mkdir -p "$HF_HOME"

echo "========================================================"
echo "  Job:   RawNet2 - Full Laundering Eval Grid"
echo "  GPU:   $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo N/A)"
echo "  Node:  $(hostname)"
echo "  Start: $(date)"
echo "========================================================"

python eval_suite.py \
    --model    rawnet2 \
    --data_root data/ASVspoof2019/LA \
    --pipelines N M P \
    --strengths L M H \
    --depths 0 1 2 3 \
    $EXTRA_ARGS

echo "========================================================"
echo "  Done: $(date)"
echo "  Results: outputs/eval_suite/rawnet2/master_results.csv"
echo "========================================================"

#!/bin/bash
#SBATCH --job-name=eval_aasist
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=logs/eval_aasist_%j.out
#SBATCH --error=logs/eval_aasist_%j.err

# ─────────────────────────────────────────────────────────────────────────────
# Full laundering evaluation grid for AASIST (raw audio).
# Runs all 3 pipelines (N, M, P) x 3 strengths (L, M, H) x 4 depths (0-3).
# Supports resume: already-completed conditions are skipped automatically.
#
# Outputs -> outputs/eval_suite/aasist/
#   master_results.csv          all EER / tDCF results
#   summary/                    collapse curves, strength heatmaps, AURC
#   <pipeline>/<strength>/k<d>/ per-condition DET curves + per-attack EER
#
# Submit:
#   sbatch slurm/eval_aasist.sbatch
#
# Optional extra args (appended to eval_suite.py):
#   sbatch slurm/eval_aasist.sbatch --max_eval 1000
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

EXTRA_ARGS="${@:-}"

PROJECT_DIR="/home/mahadevballa23/scratch_space/signal-laundering-deepfake"
cd "$PROJECT_DIR"

if [ ! -d "logs" ]; then
    echo "ERROR: logs directory is missing. Create it manually before submitting this job."
    exit 1
fi

module purge
module load python/3.11.14

if [ ! -f ".venv/bin/activate" ]; then
    echo "ERROR: missing virtual environment at $PROJECT_DIR/.venv"
    echo "Create it manually and install requirements before submitting this job."
    exit 1
fi
source .venv/bin/activate
if [ ! -f "external/aasist/main.py" ]; then
    echo "ERROR: external/aasist is missing. Submodules must be initialized before submitting this job."
    echo "Initialize submodules outside this job script before submission."
    exit 1
fi

export HF_HOME="${HF_HOME:-$PROJECT_DIR/.cache/huggingface}"
if [ ! -d "$HF_HOME" ]; then
    echo "ERROR: HF_HOME directory '$HF_HOME' is missing."
    echo "Create it manually or set HF_HOME to an existing directory before submitting this job."
    exit 1
fi

echo "========================================================"
echo "  Job:   AASIST - Full Laundering Eval Grid"
echo "  GPU:   $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo N/A)"
echo "  Node:  $(hostname)"
echo "  Start: $(date)"
echo "========================================================"

python eval_suite.py \
    --model    aasist \
    --data_root data/LA \
    --pipelines N M P \
    --strengths L M H \
    --depths 0 1 2 3 \
    $EXTRA_ARGS

echo "========================================================"
echo "  Done: $(date)"
echo "  Results: outputs/eval_suite/aasist/master_results.csv"
echo "========================================================"

#!/bin/bash
#SBATCH --job-name=train_aasist
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=logs/train_aasist_%j.out
#SBATCH --error=logs/train_aasist_%j.err

# ─────────────────────────────────────────────────────────────────────────────
# Train AASIST (raw audio, end-to-end)
# AASIST uses pretrained weights from the external submodule.
# This job validates weights exist and runs a full clean+launder eval pass
# to confirm the model is ready (AASIST doesn't need a training loop here;
# it loads pretrained weights from external/aasist/models/weights/AASIST.pth).
#
# If you want to fine-tune AASIST, extend this script with your fine-tune
# entry-point. As-is, it verifies setup and runs dry-run evaluation.
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

PROJECT_DIR="/home/mahadevballa23/scratch_space/signal-laundering-deepfake"
cd "$PROJECT_DIR"

if [ ! -d "logs" ]; then
    echo "ERROR: logs directory is missing. Create it manually before submitting this job."
    exit 1
fi
if [ ! -d "models" ]; then
    echo "ERROR: models directory is missing. Create it manually before submitting this job."
    exit 1
fi

module purge
module load python/3.11.14        # adjust to your HPC module name

# ── Virtual environment ───────────────────────────────────────────────────────
if [ ! -f ".venv/bin/activate" ]; then
    echo "ERROR: missing virtual environment at $PROJECT_DIR/.venv"
    echo "Create it manually and install requirements before submitting this job."
    exit 1
fi
source .venv/bin/activate

# ── Verify external/aasist is already present (no in-job fetch) ─────────────
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
echo "  Job:   AASIST - Raw Audio (pretrained weights check)"
echo "  GPU:   $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo N/A)"
echo "  Node:  $(hostname)"
echo "  Start: $(date)"
echo "========================================================"

# Verify weights are present
WEIGHTS="external/aasist/models/weights/AASIST.pth"
if [ ! -f "$WEIGHTS" ]; then
    echo "ERROR: Pretrained weights not found at $WEIGHTS"
    echo "Download them from the AASIST repo / paper release and place them there."
    exit 1
fi
echo "Weights found: $WEIGHTS"

# Sanity-check dry run (200 samples, clean baseline only)
python eval_suite.py \
    --model aasist \
    --data_root data/LA \
    --depths 0 \
    --dry_run \
    --max_eval 200

echo "========================================================"
echo "  AASIST setup verified. Run eval_aasist.sbatch for"
echo "  the full laundering evaluation grid."
echo "  Done: $(date)"
echo "========================================================"

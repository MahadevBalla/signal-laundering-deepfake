#!/bin/bash
#SBATCH --job-name=train_rawnet2
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=logs/train_rawnet2_%j.out
#SBATCH --error=logs/train_rawnet2_%j.err

# ─────────────────────────────────────────────────────────────────────────────
# Train RawNet2 (raw audio, end-to-end)
# Like AASIST, RawNet2 uses pretrained weights from the external submodule:
#   external/aasist/models/weights/RawNet2.pth
# This script verifies the weights and runs a dry-run eval to confirm setup.
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
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
if [ ! -d ".venv" ]; then
    python -m venv .venv
fi
source .venv/bin/activate
pip install --upgrade pip -q
pip install -r requirements.txt -q

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
echo "  Job:   RawNet2 - Raw Audio (pretrained weights check)"
echo "  GPU:   $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo N/A)"
echo "  Node:  $(hostname)"
echo "  Start: $(date)"
echo "========================================================"

WEIGHTS="external/aasist/models/weights/RawNet2.pth"
if [ ! -f "$WEIGHTS" ]; then
    echo "ERROR: Pretrained weights not found at $WEIGHTS"
    echo "Download them from the AASIST/RawNet2 repo release and place them there."
    exit 1
fi
echo "Weights found: $WEIGHTS"

# Sanity-check dry run
python eval_suite.py \
    --model rawnet2 \
    --data_root data/ASVspoof2019/LA \
    --depths 0 \
    --dry_run \
    --max_eval 200

echo "========================================================"
echo "  RawNet2 setup verified. Run eval_rawnet2.sbatch for"
echo "  the full laundering evaluation grid."
echo "  Done: $(date)"
echo "========================================================"

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
# Train RawNet2 from scratch using external/aasist/main.py.
#
# Custom config:
#   external/aasist/config/RawNet2_scratch_signal.conf
#
# Dataset root expected by this config:
#   data/LA/
#     ASVspoof2019_LA_train/flac/
#     ASVspoof2019_LA_dev/flac/
#     ASVspoof2019_LA_eval/flac/
#     ASVspoof2019_LA_cm_protocols/
#     ASVspoof2019_LA_asv_scores/
#
# Output directory passed to trainer:
#   outputs/rawnet2_scratch/
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

PROJECT_DIR="/home/mahadevballa23/scratch_space/signal-laundering-deepfake"
cd "$PROJECT_DIR"

RAWNET2_CONFIG="external/aasist/config/RawNet2_scratch_signal.conf"
RAWNET2_DATA_ROOT="data/LA"
RAWNET2_OUTPUT_DIR="outputs/rawnet2_scratch"

if [ ! -d "logs" ]; then
    echo "ERROR: logs directory is missing. Create it manually before submitting this job."
    exit 1
fi
if [ ! -d "outputs" ]; then
    echo "ERROR: outputs directory is missing. Create it manually before submitting this job."
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
if [ ! -f "$RAWNET2_CONFIG" ]; then
    echo "ERROR: RawNet2 config not found at $RAWNET2_CONFIG"
    exit 1
fi

# Dataset sanity checks for ASVspoof2019 LA layout
if [ ! -d "$RAWNET2_DATA_ROOT/ASVspoof2019_LA_train/flac" ]; then
    echo "ERROR: missing $RAWNET2_DATA_ROOT/ASVspoof2019_LA_train/flac"
    exit 1
fi
if [ ! -d "$RAWNET2_DATA_ROOT/ASVspoof2019_LA_dev/flac" ]; then
    echo "ERROR: missing $RAWNET2_DATA_ROOT/ASVspoof2019_LA_dev/flac"
    exit 1
fi
if [ ! -d "$RAWNET2_DATA_ROOT/ASVspoof2019_LA_eval/flac" ]; then
    echo "ERROR: missing $RAWNET2_DATA_ROOT/ASVspoof2019_LA_eval/flac"
    exit 1
fi
if [ ! -f "$RAWNET2_DATA_ROOT/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt" ]; then
    echo "ERROR: missing train protocol file in $RAWNET2_DATA_ROOT/ASVspoof2019_LA_cm_protocols"
    exit 1
fi
if [ ! -f "$RAWNET2_DATA_ROOT/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt" ]; then
    echo "ERROR: missing dev protocol file in $RAWNET2_DATA_ROOT/ASVspoof2019_LA_cm_protocols"
    exit 1
fi
if [ ! -f "$RAWNET2_DATA_ROOT/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt" ]; then
    echo "ERROR: missing eval protocol file in $RAWNET2_DATA_ROOT/ASVspoof2019_LA_cm_protocols"
    exit 1
fi
if [ ! -f "$RAWNET2_DATA_ROOT/ASVspoof2019_LA_asv_scores/ASVspoof2019.LA.asv.eval.gi.trl.scores.txt" ]; then
    echo "ERROR: missing ASV score file in $RAWNET2_DATA_ROOT/ASVspoof2019_LA_asv_scores"
    exit 1
fi

export HF_HOME="${HF_HOME:-$PROJECT_DIR/.cache/huggingface}"
if [ ! -d "$HF_HOME" ]; then
    echo "ERROR: HF_HOME directory '$HF_HOME' is missing."
    echo "Create it manually or set HF_HOME to an existing directory before submitting this job."
    exit 1
fi

echo "========================================================"
echo "  Job:   RawNet2 - From Scratch Training"
echo "  Config:  $RAWNET2_CONFIG"
echo "  Data:    $RAWNET2_DATA_ROOT"
echo "  Output:  $RAWNET2_OUTPUT_DIR"
echo "  GPU:   $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo N/A)"
echo "  Node:  $(hostname)"
echo "  Start: $(date)"
echo "========================================================"

python external/aasist/main.py \
    --config "$RAWNET2_CONFIG" \
    --output_dir "$RAWNET2_OUTPUT_DIR"

echo "========================================================"
echo "  RawNet2 scratch training finished."
echo "  Trained checkpoints are under $RAWNET2_OUTPUT_DIR"
echo "  Done: $(date)"
echo "========================================================"

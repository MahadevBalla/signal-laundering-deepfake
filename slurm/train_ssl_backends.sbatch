#!/bin/bash
#SBATCH --job-name=train_ssl_%a
#SBATCH --partition=a40
#SBATCH --qos=a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=36:00:00
#SBATCH --output=logs/train_ssl_%a_%j.out
#SBATCH --error=logs/train_ssl_%a_%j.err
#SBATCH --array=0-5   # 6 combos: wav2vec2/hubert/wavlm x aasist/rawnet2

# ─────────────────────────────────────────────────────────────────────────────
# Train all SSL frontend + backend combinations using a SLURM array job.
#
#   Index  SSL Model   Backend    Saved weight
#   -----  ---------   -------    ----------------------------------------
#     0    wav2vec2    aasist     models/wav2vec2_aasist_weighted.pth
#     1    wav2vec2    rawnet2    models/wav2vec2_rawnet2_weighted.pth
#     2    hubert      aasist     models/hubert_aasist_weighted.pth
#     3    hubert      rawnet2    models/hubert_rawnet2_weighted.pth
#     4    wavlm       aasist     models/wavlm_aasist_weighted.pth
#     5    wavlm       rawnet2    models/wavlm_rawnet2_weighted.pth
#
# PRE-REQUISITES (run on login node before submitting):
#   1. mkdir -p logs models .cache/huggingface
#   2. git submodule update --init --recursive
#   3. Pre-download HF models (see README / login node script)
#
# Submit all:
#   sbatch slurm/train_ssl_backends.sbatch
#
# Submit subset (e.g. wav2vec2 only):
#   sbatch --array=0-1 slurm/train_ssl_backends.sbatch
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── Array index → (model, backend) mapping ───────────────────────────────────
SSL_MODELS=(wav2vec2 wav2vec2 hubert hubert wavlm wavlm)
BACKENDS=(aasist rawnet2 aasist rawnet2 aasist rawnet2)

IDX=${SLURM_ARRAY_TASK_ID}
MODEL="${SSL_MODELS[$IDX]}"
BACKEND="${BACKENDS[$IDX]}"

PROJECT_DIR="/home/gopalpatil/22d0296/projects/signal-laundering-deepfake"
VENV_DIR="$PROJECT_DIR/.venv"
cd "$PROJECT_DIR"

# ── Create logs/ and models/ if missing ──────────────────────────────────────
mkdir -p logs models

# ── Activate Spack and load Anaconda (matches eval scripts) ──────────────────
source /lustre-flash/apps/spack/share/spack/setup-env.sh
spack load anaconda3@2023.09-0
eval "$(conda shell.bash hook)"

# ── Activate pre-built venv ───────────────────────────────────────────────────
if [ ! -f "$VENV_DIR/bin/activate" ]; then
    echo "ERROR: Virtual environment not found at $VENV_DIR"
    echo "Build it on the login node first:"
    echo "  source /lustre-flash/apps/spack/share/spack/setup-env.sh"
    echo "  spack load anaconda3@2023.09-0"
    echo "  eval \"$(conda shell.bash hook)\""
    echo "  python -m venv $VENV_DIR"
    echo "  source $VENV_DIR/bin/activate"
    echo "  pip install -r $PROJECT_DIR/requirements.txt"
    exit 1
fi
source "$VENV_DIR/bin/activate"
echo ">>> Python: $(which python) | $(python --version)"

# ── Install requirements ──────────────────────────────────────────────────────
echo ">>> Installing requirements..."
pip install --upgrade pip --quiet
pip install -r "$PROJECT_DIR/requirements.txt" --quiet
echo ">>> Requirements installed."

# ── Verify AASIST submodule ───────────────────────────────────────────────────
if [ ! -f "external/aasist/main.py" ]; then
    echo "ERROR: external/aasist submodule is missing."
    echo "Run on login node: git submodule update --init --recursive"
    exit 1
fi

# ── HuggingFace cache — MUST be pre-downloaded on login node ─────────────────
export HF_HOME="$PROJECT_DIR/.cache/huggingface"
export TRANSFORMERS_OFFLINE=1       # ← block any live download attempts on compute node
export HF_DATASETS_OFFLINE=1
mkdir -p "$HF_HOME"

# ── Prevent CPU thread oversubscription across 8 workers ─────────────────────
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

echo "========================================================"
echo "  Job:      SSL Backend Training"
echo "  Task ID:  $IDX"
echo "  Frontend: $MODEL (frozen)"
echo "  Backend:  $BACKEND (trained)"
echo "  GPU:      $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo N/A)"
echo "  Node:     $(hostname)"
echo "  Start:    $(date)"
echo "========================================================"

python train_ssl_backend.py \
    --model      "$MODEL" \
    --mode       weighted \
    --backend    "$BACKEND" \
    --data_root  data/LA \
    --epochs     50 \
    --patience   10 \
    --lr         1e-4 \
    --batch_size 32 \
    --embed_dim  768 \
    --num_layers 12

echo "========================================================"
echo "  Done:   $(date)"
echo "  Saved:  models/${MODEL}_${BACKEND}_weighted.pth"
echo "========================================================"
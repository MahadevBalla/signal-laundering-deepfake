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
#SBATCH --array=0-8   # 9 combos: 3 SSL frontends x 3 backends (ffn/aasist/rawnet2)

# ─────────────────────────────────────────────────────────────────────────────
# Train all SSL frontend + backend combinations using a SLURM array job.
#
#   Index  SSL Model   Backend    Saved weight
#   -----  ---------   -------    ----------------------------------------
#     0    wav2vec2    ffn        models/wav2vec2_ffn_weighted.pth
#     1    wav2vec2    aasist     models/wav2vec2_aasist_weighted.pth
#     2    wav2vec2    rawnet2    models/wav2vec2_rawnet2_weighted.pth
#     3    hubert      ffn        models/hubert_ffn_weighted.pth
#     4    hubert      aasist     models/hubert_aasist_weighted.pth
#     5    hubert      rawnet2    models/hubert_rawnet2_weighted.pth
#     6    wavlm       ffn        models/wavlm_ffn_weighted.pth
#     7    wavlm       aasist     models/wavlm_aasist_weighted.pth
#     8    wavlm       rawnet2    models/wavlm_rawnet2_weighted.pth
#
# PRE-REQUISITES (run on login node before submitting):
#   1. mkdir -p logs models data/ssl_cache .cache/huggingface
#   2. git submodule update --init --recursive
#   3. Pre-download HF models (see README / login node script)
#
# SSL frontend embedding cache:
#   Frozen-SSL sequence outputs are cached under data/ssl_cache/<model>/{train,dev}
#   and reused across epochs and across the ffn/aasist/rawnet2 array tasks for
#   the SAME SSL model (since the cache is keyed by extract_layers, and all
#   weighted-mode tasks request all 12 layers). Running task 0 (wav2vec2 ffn)
#   first will populate the cache so tasks 1-2 (wav2vec2 aasist/rawnet2) skip
#   the SSL forward pass entirely after epoch 1. If you submit the full
#   array=0-8 at once, tasks for the same SSL model may run concurrently and
#   write the cache redundantly on first epoch — this is safe (last-write-wins
#   on a per-utterance file) but offers no speedup until one task finishes
#   epoch 1. For max cache reuse, submit per-model batches sequentially:
#     sbatch --array=0-2 slurm/train_ssl_backends.sbatch   # wav2vec2
#     sbatch --array=3-5 slurm/train_ssl_backends.sbatch   # hubert
#     sbatch --array=6-8 slurm/train_ssl_backends.sbatch   # wavlm
#
# Submit all:
#   sbatch slurm/train_ssl_backends.sbatch
#
# Submit subset (e.g. wav2vec2 only):
#   sbatch --array=0-2 slurm/train_ssl_backends.sbatch
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── Array index → (model, backend) mapping ───────────────────────────────────
SSL_MODELS=(wav2vec2 wav2vec2 wav2vec2 hubert hubert hubert wavlm wavlm wavlm)
BACKENDS=(ffn aasist rawnet2 ffn aasist rawnet2 ffn aasist rawnet2)

IDX=${SLURM_ARRAY_TASK_ID}
MODEL="${SSL_MODELS[$IDX]}"
BACKEND="${BACKENDS[$IDX]}"

PROJECT_DIR="/home/gopalpatil/22d0296/projects/signal-laundering-deepfake"
VENV_DIR="$PROJECT_DIR/.venv"
CACHE_DIR="$PROJECT_DIR/data/ssl_cache"
cd "$PROJECT_DIR"

# ── Create logs/, models/, cache dir if missing ──────────────────────────────
mkdir -p logs models "$CACHE_DIR"

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
echo "  Cache:    $CACHE_DIR/$MODEL/{train,dev}"
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
    --num_layers 12 \
    --cache_dir  "$CACHE_DIR"

echo "========================================================"
echo "  Done:   $(date)"
echo "  Saved:  models/${MODEL}_${BACKEND}_weighted.pth"
echo "========================================================"

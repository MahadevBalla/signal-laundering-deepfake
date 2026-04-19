#!/bin/bash
#SBATCH --job-name=train_ssl_%a
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=36:00:00
#SBATCH --output=logs/train_ssl_%a_%j.out
#SBATCH --error=logs/train_ssl_%a_%j.err
#SBATCH --array=0-8    # 9 combos: 3 SSL frontends x 3 backends

# ─────────────────────────────────────────────────────────────────────────────
# Train all SSL frontend + backend combinations using a SLURM array job.
# Each array task maps to one (ssl_model, backend) pair:
#
#   Index  SSL Model   Backend    Saved weight
#   -----  ---------   -------    -----------------
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
# Submit:
#   sbatch slurm/train_ssl_backends.sbatch
#
# Submit only a subset (e.g. wav2vec2 only):
#   sbatch --array=0-2 slurm/train_ssl_backends.sbatch
#
# Training config: Adam lr=1e-4, batch=32, CE loss, 50 epochs,
# early stopping patience=10 on dev loss. SSL frontend is FROZEN.
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── Array index → (model, backend) mapping ───────────────────────────────────
SSL_MODELS=(wav2vec2 wav2vec2 wav2vec2 hubert hubert hubert wavlm wavlm wavlm)
BACKENDS=(ffn aasist rawnet2 ffn aasist rawnet2 ffn aasist rawnet2)

IDX=${SLURM_ARRAY_TASK_ID}
MODEL="${SSL_MODELS[$IDX]}"
BACKEND="${BACKENDS[$IDX]}"

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

if [ ! -f "external/aasist/main.py" ]; then
    echo "ERROR: external/aasist is missing. Submodules must be initialized before submitting this job."
    echo "Initialize submodules outside this job script before submission."
    exit 1
fi

# ── HuggingFace cache ─────────────────────────────────────────────────────────
export HF_HOME="${HF_HOME:-$PROJECT_DIR/.cache/huggingface}"
if [ ! -d "$HF_HOME" ]; then
    echo "ERROR: HF_HOME directory '$HF_HOME' is missing."
    echo "Create it manually or set HF_HOME to an existing directory before submitting this job."
    exit 1
fi

echo "========================================================"
echo "  Job:     SSL Backend Training"
echo "  Task ID: $IDX"
echo "  Frontend (frozen): $MODEL"
echo "  Backend (trained): $BACKEND"
echo "  GPU:     $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo N/A)"
echo "  Node:    $(hostname)"
echo "  Start:   $(date)"
echo "========================================================"

python train_ssl_backend.py \
    --model    "$MODEL" \
    --mode     weighted \
    --backend  "$BACKEND" \
    --data_root data/ASVspoof2019/LA \
    --epochs   50 \
    --patience 10 \
    --lr       1e-4 \
    --batch_size 32 \
    --embed_dim  768 \
    --num_layers 12

echo "========================================================"
echo "  Done: $(date)"
echo "  Saved: models/${MODEL}_${BACKEND}_weighted.pth"
echo "========================================================"

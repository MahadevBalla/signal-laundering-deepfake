#!/bin/bash
#SBATCH --job-name=stab_w2v2_aaist_%a
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=07:00:00
#SBATCH --output=logs/stab_%a_%j.out
#SBATCH --error=logs/stab_%a_%j.err
#SBATCH --array=0-4

# ─────────────────────────────────────────────────────────────────────────────
# Stability-regularised laundering-aware training — lambda_stab sweep.
# Trains wav2vec2+aasist with Pipeline N, strength M, depths 1-3.
# Warm-starts from the clean-trained checkpoint (models/wav2vec2_aasist_weighted.pth).
#
# Array index → lambda_stab:
#   0 → 0.01
#   1 → 0.05
#   2 → 0.10  (recommended starting point)
#   3 → 0.50
#   4 → 1.00
#
# PRE-REQUISITES (run on login node before submitting):
#   1. Train clean SSL backend:
#        sbatch slurm/train_ssl_backends.sbatch  # or --array=1 for wav2vec2_aasist only
#   2. Verify weights exist:
#        ls models/wav2vec2_aasist_weighted.pth
#   3. mkdir -p logs data/ssl_cache
#
# Submit all 5 lambda values:
#   sbatch slurm/train_stab.sbatch
#
# Submit single lambda (e.g. lambda_stab=0.10, index 2):
#   sbatch --array=2 slurm/train_stab.sbatch
#
# After training, evaluate on full laundering grid:
#   python eval_suite.py --model wav2vec2_aasist \
#       --weights outputs/stab_training/wav2vec2_aasist/<run_id>/best_laund.pth
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

LAMBDAS=(0.01 0.05 0.10 0.50 1.00)
LAMBDA_STAB="${LAMBDAS[$SLURM_ARRAY_TASK_ID]}"

MODEL="wav2vec2"
BACKEND="aasist"
INIT_WEIGHTS="models/wav2vec2_aasist_weighted.pth"
RUN_ID="${MODEL}_${BACKEND}_lam${LAMBDA_STAB}_lc0.5_pipN_d123"

PROJECT_DIR="/home/mahadevballa23/scratch_space/signal-laundering-deepfake"
VENV_DIR="$PROJECT_DIR/.venv"
CACHE_DIR="$PROJECT_DIR/data/ssl_cache"

cd "$PROJECT_DIR"
mkdir -p logs "$CACHE_DIR"

# ── Environment ───────────────────────────────────────────────────────────────
if [ ! -f "$VENV_DIR/bin/activate" ]; then
    echo "ERROR: Missing venv at $VENV_DIR"
    exit 1
fi
source "$VENV_DIR/bin/activate"

echo ">>> Python: $(which python) | $(python --version)"
python -c "import torch; print(f'>>> torch {torch.__version__} | CUDA {torch.cuda.is_available()}')"

# ── Submodule check ───────────────────────────────────────────────────────────
if [ ! -f "external/aasist/main.py" ]; then
    echo "ERROR: external/aasist submodule missing."
    echo "Run on login node: git submodule update --init --recursive"
    exit 1
fi

# ── Weights pre-flight check ──────────────────────────────────────────────────
if [ ! -f "$INIT_WEIGHTS" ]; then
    echo "ERROR: Clean-trained weights not found at $INIT_WEIGHTS"
    echo "Train the clean SSL backend first:"
    echo "  sbatch slurm/train_ssl_backends.sbatch"
    echo "Or run without --init_weights to train from scratch (remove the flag below)."
    exit 1
fi

# ── HuggingFace offline ───────────────────────────────────────────────────────
export HF_HOME="$PROJECT_DIR/.cache/huggingface"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
mkdir -p "$HF_HOME"

# ── Thread settings ───────────────────────────────────────────────────────────
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

echo "========================================================"
echo "  Job:          Stability Training (lambda sweep)"
echo "  Model:        $MODEL + $BACKEND"
echo "  lambda_stab:  $LAMBDA_STAB  (array task $SLURM_ARRAY_TASK_ID)"
echo "  Init weights: $INIT_WEIGHTS"
echo "  GPU:          $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo N/A)"
echo "  Node:         $(hostname)"
echo "  Start:        $(date)"
echo "========================================================"

python train_stab.py \
    --model              "$MODEL" \
    --backend            "$BACKEND" \
    --init_weights       "$INIT_WEIGHTS" \
    --pipelines          N \
    --depths             1 2 3 \
    --strength           M \
    --eval_depth         3 \
    --lambda_stab        "$LAMBDA_STAB" \
    --lambda_clean       0.5 \
    --epochs             50 \
    --patience           10 \
    --eval_every         5 \
    --lr                 1e-4 \
    --weight_decay       1e-4 \
    --batch_size         32 \
    --embed_dim          768 \
    --num_layers         12 \
    --data_root          data/LA \
    --precomputed_root   data/precomputed \
    --cache_dir          "$CACHE_DIR" \
    --output_dir         outputs/stab_training \
    --run_id             "$RUN_ID" \
    --resume

echo "========================================================"
echo "  Finished: $(date)"
echo "========================================================"

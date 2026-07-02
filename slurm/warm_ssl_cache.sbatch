#!/bin/bash
#SBATCH --job-name=warm_ssl_cache
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=logs/warm_ssl_cache_%j.out
#SBATCH --error=logs/warm_ssl_cache_%j.err

# ─────────────────────────────────────────────────────────────────────────────
# Warm the SSL embedding cache for all 27 precomputed laundered variants
# (3 pipelines × 3 strengths × 3 depths) so that eval_stab.sbatch runs with
# zero wav2vec2 transformer forward passes.
#
# This job should be submitted AFTER precompute_laundering.sbatch completes.
# Use SLURM job dependencies to enforce ordering:
#
#   # Submit the full dependency chain in one go:
#   PRECOMP=$(sbatch --parsable slurm/precompute_laundering.sbatch)
#   WARM=$(sbatch --parsable --dependency=afterok:$PRECOMP slurm/warm_ssl_cache.sbatch)
#   sbatch --dependency=afterok:$WARM slurm/eval_stab.sbatch
#
# Manual submission (after verifying precomputed audio is present):
#   sbatch slurm/warm_ssl_cache.sbatch
#
# What it does:
#   Runs warm_ssl_cache.py, which loads each precomputed variant from
#   data/precomputed/ through the frozen wav2vec2 frontend (all 12 layers)
#   and writes per-utterance embeddings to:
#       data/ssl_cache/wav2vec2/eval_{pipeline}_d{depth}_{strength}/<utt_id>.pt
#
# After this job completes, eval_stab.sbatch will:
#   1. Load static audio from data/precomputed/  (no FFmpeg)
#   2. Load SSL embeddings from data/ssl_cache/  (no wav2vec2)
#   3. Push embeddings through the tiny AASIST backend only
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

MODEL="wav2vec2"

PROJECT_DIR="/home/mahadevballa23/scratch_space/signal-laundering-deepfake"
VENV_DIR="$PROJECT_DIR/.venv"
PRECOMPUTED_ROOT="$PROJECT_DIR/data/precomputed"
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
    exit 1
fi

# ── Verify precomputed audio exists ───────────────────────────────────────────
if [ ! -d "$PRECOMPUTED_ROOT" ]; then
    echo "ERROR: precomputed_root not found: $PRECOMPUTED_ROOT"
    echo "Submit precompute_laundering.sbatch first and wait for it to complete."
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
echo "  Job:             SSL Cache Warmer"
echo "  Model:           $MODEL"
echo "  Precomputed:     $PRECOMPUTED_ROOT"
echo "  Cache output:    $CACHE_DIR/$MODEL/eval_{pipeline}_d{depth}_{strength}/"
echo "  Variants:        N/M/P × L/M/H × d1/d2/d3 = 27 total"
echo "  GPU:             $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo N/A)"
echo "  Node:            $(hostname)"
echo "  Start:           $(date)"
echo "========================================================"

python warm_ssl_cache.py \
    --model              "$MODEL" \
    --precomputed_root   "$PRECOMPUTED_ROOT" \
    --cache_dir          "$CACHE_DIR" \
    --pipelines          N M P \
    --strengths          L M H \
    --depths             1 2 3 \
    --batch_size         8 \
    --num_workers        "$SLURM_CPUS_PER_TASK"

echo "========================================================"
echo "  Finished: $(date)"
echo "  Cache:    $CACHE_DIR/$MODEL/eval_*/"
echo ""
echo "  Next step — submit eval_stab.sbatch:"
echo "    sbatch slurm/eval_stab.sbatch"
echo "========================================================"

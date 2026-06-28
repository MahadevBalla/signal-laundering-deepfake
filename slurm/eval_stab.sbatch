#!/bin/bash
#SBATCH --job-name=eval_stab_%a
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=36:00:00
#SBATCH --output=logs/eval_stab_%a_%j.out
#SBATCH --error=logs/eval_stab_%a_%j.err
#SBATCH --array=0-2

set -euo pipefail

# ─────────────────────────────────────────────────────────────────────────────
# Evaluate stability-trained wav2vec2+AASIST models over the full laundering
# grid (N/M/P × L/M/H × k=0-3).
#
# Array index → lambda_stab:
#   0 → 0.01
#   1 → 0.10
#   2 → 1.00
#
# Submit:
#   sbatch slurm/eval_stab.sbatch
#
# Single lambda:
#   sbatch --array=2 slurm/eval_stab.sbatch
# ─────────────────────────────────────────────────────────────────────────────

LAMBDAS=(0.01 0.10 1.00)
LAMBDA_STAB="${LAMBDAS[$SLURM_ARRAY_TASK_ID]}"

MODEL="wav2vec2_aasist"
RUN_ID="wav2vec2_aasist_lam${LAMBDA_STAB}_lc0.5_pipN_d123"

PROJECT_DIR="/home/mahadevballa23/scratch_space/signal-laundering-deepfake"
VENV_DIR="$PROJECT_DIR/.venv"
CACHE_DIR="$PROJECT_DIR/data/ssl_cache"

CHECKPOINT="$PROJECT_DIR/outputs/stab_training/wav2vec2_aasist/${RUN_ID}/best_laund.pth"

OUTPUT_DIR="$PROJECT_DIR/outputs/eval_suite_stab/${RUN_ID}"

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

# ── Check checkpoint ──────────────────────────────────────────────────────────
if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: checkpoint not found:"
    echo "  $CHECKPOINT"
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
echo "  Job:        Stability Evaluation"
echo "  Model:      $MODEL"
echo "  lambda:     $LAMBDA_STAB"
echo "  Checkpoint: $CHECKPOINT"
echo "  GPU:        $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo N/A)"
echo "  Node:       $(hostname)"
echo "  Start:      $(date)"
echo "========================================================"

python eval_suite.py \
    --model "$MODEL" \
    --weights "$CHECKPOINT" \
    --output_dir "$OUTPUT_DIR" \
    --data_root data/LA \
    --pipelines N M P \
    --strengths L M H \
    --depths 0 1 2 3 \
    --run_cka \
    --cka_max_eval 1000 \
    --cache_dir "$CACHE_DIR"

echo "========================================================"
echo "  Finished: $(date)"
echo "  Results:  $OUTPUT_DIR"
echo "========================================================"
#!/bin/bash
#SBATCH --job-name=eval_stab_%a
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/eval_stab_%a_%j.out
#SBATCH --error=logs/eval_stab_%a_%j.err
#SBATCH --array=0-4

set -euo pipefail

# ─────────────────────────────────────────────────────────────────────────────
# Evaluate stability-trained wav2vec2+AASIST models over the full laundering
# grid (N/M/P × L/M/H × k=0-3) in on-the-fly mode (no precomputed cache).
#
# Array index → lambda_stab:
#   0 → 0.01
#   1 → 0.05
#   2 → 0.10
#   3 → 0.50
#   4 → 1.00
#
# ON-THE-FLY MODE (no --precomputed_root):
#   DataLoader workers launder audio on the CPU in parallel while the GPU
#   runs inference.  Key settings preventing GPU idle:
#     - 6 DataLoader workers (out of 8 allocated CPUs)
#     - pin_memory + persistent_workers + prefetch_factor already set in
#       ssl_eval_wrapper.py
#     - OMP/MKL/OPENBLAS/VECLIB/NUMEXPR all pinned to 1 thread so worker
#       processes do not contend for CPU cores
#     - torch.amp.autocast FP16 inference (set in ssl_eval_wrapper.py)
#     - torch.backends.cudnn.benchmark (set in eval_suite.py)
#
# Submit all lambdas:
#   sbatch slurm/eval_stab.sbatch
#
# Single lambda:
#   sbatch --array=2 slurm/eval_stab.sbatch
# ─────────────────────────────────────────────────────────────────────────────

LAMBDAS=(0.01 0.05 0.10 0.50 1.00)
LAMBDA_STAB="${LAMBDAS[$SLURM_ARRAY_TASK_ID]}"

MODEL="wav2vec2_aasist"
RUN_ID="wav2vec2_aasist_lam${LAMBDA_STAB}_lc0.5_pipN_d123"

PROJECT_DIR="/home/mahadevballa23/scratch_space/signal-laundering-deepfake"
VENV_DIR="$PROJECT_DIR/.venv"
CHECKPOINT="$PROJECT_DIR/outputs/stab_training/wav2vec2_aasist/${RUN_ID}/best_laund.pth"
OUTPUT_DIR="$PROJECT_DIR/outputs/eval_suite_stab/${RUN_ID}"

cd "$PROJECT_DIR"
mkdir -p logs "$OUTPUT_DIR"

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

# ── CPU / Worker Tuning ───────────────────────────────────────────────────────
# Pin every threading library to 1 thread per DataLoader worker process.
# Without this, workers spawn NumPy/SciPy sub-threads that compete for the
# same allocated CPUs and degrade laundering throughput. num_workers is now
# derived at runtime from the SLURM-granted CPU affinity (see
# ssl_eval_wrapper.py), not hardcoded.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

echo "========================================================"
echo "  Job:        Stability Evaluation (on-the-fly mode)"
echo "  Model:      $MODEL"
echo "  lambda:     $LAMBDA_STAB"
echo "  Checkpoint: $CHECKPOINT"
echo "  Output:     $OUTPUT_DIR"
echo "  GPU:        $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo N/A)"
echo "  Node:       $(hostname)"
echo "  Start:      $(date)"
echo "========================================================"

# ── Run Evaluation (Zero Cache / On-the-Fly) ──────────────────────────────────
python eval_suite.py \
    --model "$MODEL" \
    --weights "$CHECKPOINT" \
    --output_dir "$OUTPUT_DIR" \
    --data_root data/LA \
    --pipelines N M P \
    --strengths L M H \
    --depths 0 1 2 3 \
    --run_cka \
    --cka_max_eval 1000

echo "========================================================"
echo "  Finished: $(date)"
echo "  Results:  $OUTPUT_DIR"
echo "========================================================"
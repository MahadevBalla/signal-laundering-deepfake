#!/bin/bash
#SBATCH --job-name=full_pipeline
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

# Usage examples:
#   sbatch scripts/run_full_pipeline.sbatch
#   sbatch scripts/run_full_pipeline.sbatch wav2vec2 aasist white
#   sbatch scripts/run_full_pipeline.sbatch hubert ffn white --dry-run

MODEL="${1:-wav2vec2}"
BACKEND="${2:-aasist}"
NOISE_MODE="${3:-white}"
shift $(( $# >= 3 ? 3 : $# ))
EXTRA_ARGS=("$@")

PROJECT_DIR="/home/mahadevballa23/scratch_space/signal-laundering-deepfake"
cd "$PROJECT_DIR"

module purge
module load python/3.11.14

if [ ! -f ".venv/bin/activate" ]; then
    echo "ERROR: missing virtual environment at $PROJECT_DIR/.venv"
    echo "Create it manually and install requirements before submitting this job."
    exit 1
fi
source .venv/bin/activate

export HF_HOME="${HF_HOME:-$PROJECT_DIR/.cache/huggingface}"
if [ ! -d "$HF_HOME" ]; then
    echo "ERROR: HF_HOME directory '$HF_HOME' is missing."
    echo "Create it manually or set HF_HOME to an existing directory before submitting this job."
    exit 1
fi

echo "========================================"
echo "  Model:      $MODEL"
echo "  Backend:    $BACKEND"
echo "  Noise:      $NOISE_MODE"
echo "  Extra args: ${EXTRA_ARGS[*]:-<none>}"
echo "  Project:    $PROJECT_DIR"
echo "  GPU:        $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "  Node:       $(hostname)"
echo "  Start:      $(date)"
echo "========================================"

python scripts/run_full_pipeline.py \
    --model "$MODEL" \
    --backend "$BACKEND" \
    --noise-mode "$NOISE_MODE" \
    --sync-submodules \
    --run-cka \
    "${EXTRA_ARGS[@]}"

echo "========================================"
echo "  Done:  $(date)"
echo "========================================"

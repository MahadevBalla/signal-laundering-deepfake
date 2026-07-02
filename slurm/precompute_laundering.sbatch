#!/bin/bash
#SBATCH --job-name=precompute_%a
#SBATCH --partition=general
#SBATCH --cpus-per-task=24
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/precompute_%a_%j.out
#SBATCH --error=logs/precompute_%a_%j.err
#SBATCH --array=0-8

# ─────────────────────────────────────────────────────────────────────────────
# Precompute all 9 laundering variants (3 pipelines × 3 strengths), depths 1-3.
#
# Array index → (pipeline, strength):
#   0 → N L    3 → M L    6 → P L
#   1 → N M    4 → M M    7 → P M
#   2 → N H    5 → M H    8 → P H
#
# Outputs written to:
#   data/precomputed/<pipeline>_d<depth>_<strength>/
#
# Submit all 9 variants in parallel:
#   sbatch slurm/precompute_laundering.sbatch
#
# Submit a single variant (e.g. N/M, index 1):
#   sbatch --array=1 slurm/precompute_laundering.sbatch
#
# PRE-REQUISITE for eval_stab.sbatch:
#   This job must complete before warm_ssl_cache.sbatch is submitted.
#   Use SLURM job dependencies:
#     PRECOMP=$(sbatch --parsable slurm/precompute_laundering.sbatch)
#     sbatch --dependency=afterok:$PRECOMP slurm/warm_ssl_cache.sbatch
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# 9-element flat arrays: index i → PIPELINES[i], STRENGTHS[i]
PIPELINES=(N N N M M M P P P)
STRENGTHS=(L M H L M H L M H)

PIPELINE="${PIPELINES[$SLURM_ARRAY_TASK_ID]}"
STRENGTH="${STRENGTHS[$SLURM_ARRAY_TASK_ID]}"

PROJECT_DIR="/home/mahadevballa23/scratch_space/signal-laundering-deepfake"
VENV_DIR="$PROJECT_DIR/.venv"

cd "$PROJECT_DIR"
mkdir -p logs

# ── Environment ───────────────────────────────────────────────────────────────
if [ ! -f "$VENV_DIR/bin/activate" ]; then
    echo "ERROR: Missing venv at $VENV_DIR"
    exit 1
fi
source "$VENV_DIR/bin/activate"

echo ">>> Python: $(which python) | $(python --version)"

echo "========================================================"
echo "  Job:       Precompute Laundering"
echo "  Task ID:   $SLURM_ARRAY_TASK_ID"
echo "  Pipeline:  $PIPELINE"
echo "  Strength:  $STRENGTH"
echo "  Depths:    1 2 3"
echo "  Node:      $(hostname)"
echo "  Start:     $(date)"
echo "========================================================"

python precompute_laundering.py \
    --data_root   data/LA \
    --pipeline    "$PIPELINE" \
    --depths      1 2 3 \
    --strength    "$STRENGTH" \
    --output_root data/precomputed \
    --config_dir  configs \
    --workers     "$SLURM_CPUS_PER_TASK"

echo "========================================================"
echo "  Finished: $(date)"
echo "  Output:   data/precomputed/${PIPELINE}_d{1,2,3}_${STRENGTH}/"
echo "========================================================"

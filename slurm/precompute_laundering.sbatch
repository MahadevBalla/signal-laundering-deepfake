#!/bin/bash
#SBATCH --job-name=precompute
#SBATCH --partition=general
#SBATCH --cpus-per-task=24
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/precompute_%j.out
#SBATCH --error=logs/precompute_%j.err

set -euo pipefail

PROJECT_DIR="/home/mahadevballa23/scratch_space/signal-laundering-deepfake"
VENV_DIR="$PROJECT_DIR/.venv"

cd "$PROJECT_DIR"
source "$VENV_DIR/bin/activate"

python precompute_laundering.py \
    --data_root data/ASVspoof2019/LA \
    --output_root data/precomputed \
    --config_dir configs \
    --pipeline N \
    --depths 1 2 3 \
    --strength M \
    --workers "$SLURM_CPUS_PER_TASK"

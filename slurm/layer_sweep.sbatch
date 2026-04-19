#!/bin/bash
#SBATCH --job-name=layer_sweep_%a
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=20:00:00
#SBATCH --output=logs/layer_sweep_%a_%j.out
#SBATCH --error=logs/layer_sweep_%a_%j.err
#SBATCH --array=0-2    # one task per SSL model: 0=wav2vec2, 1=hubert, 2=wavlm

# ─────────────────────────────────────────────────────────────────────────────
# Layer sweep: for each SSL model, train a single-layer FFN probe for each
# transformer layer (0-11) and plot EER vs. layer index.
#
# Uses layer_sweep.py (the high-level orchestrator), which calls
# train_ssl_backend.py internally for each layer with --mode single.
# Running as a 3-task array keeps each model's 12 layers on one GPU.
#
# Array tasks:
#   0 -> wav2vec2    (trains layers 0-11 sequentially, ~7h total)
#   1 -> hubert      (trains layers 0-11 sequentially, ~7h total)
#   2 -> wavlm       (trains layers 0-11 sequentially, ~7h total)
#
# Outputs per model:
#   models/<model>_ffn_layer{0-11}.pth
#   outputs/sweep/<model>/<model>_layer_sweep.json
#   outputs/sweep/<model>/<model>_layer_eer.png
#
# Submit all 3 models:
#   sbatch slurm/layer_sweep.sbatch
#
# Submit only wav2vec2 (task 0):
#   sbatch --array=0 slurm/layer_sweep.sbatch
#
# Dry-run (2 epochs, 200 utterances per layer):
#   sbatch --array=0 slurm/layer_sweep.sbatch --dry_run
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

SSL_MODELS=(wav2vec2 hubert wavlm)
IDX=${SLURM_ARRAY_TASK_ID}
MODEL="${SSL_MODELS[$IDX]}"

EXTRA_ARGS="${@:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

mkdir -p logs models

module purge
module load python/3.11.14        # adjust to your HPC module name

if [ ! -d ".venv" ]; then
    python -m venv .venv
fi
source .venv/bin/activate
pip install --upgrade pip -q
pip install -r requirements.txt -q
git submodule update --init --recursive

export HF_HOME="${HF_HOME:-$PROJECT_DIR/.cache/huggingface}"
mkdir -p "$HF_HOME"

echo "========================================================"
echo "  Job:     Layer Sweep"
echo "  Task ID: $IDX  ->  $MODEL  (layers 0-11)"
echo "  GPU:     $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo N/A)"
echo "  Node:    $(hostname)"
echo "  Start:   $(date)"
echo "========================================================"

python layer_sweep.py \
    --model      "$MODEL" \
    --data_root  data/ASVspoof2019/LA \
    --layers     0 1 2 3 4 5 6 7 8 9 10 11 \
    --epochs     20 \
    --lr         1e-4 \
    --batch_size 32 \
    $EXTRA_ARGS

echo "========================================================"
echo "  Done: $(date)"
echo "  JSON: outputs/sweep/${MODEL}/${MODEL}_layer_sweep.json"
echo "  Plot: outputs/sweep/${MODEL}/${MODEL}_layer_eer.png"
echo "========================================================"

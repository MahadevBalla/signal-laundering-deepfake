#!/bin/bash
#SBATCH --job-name=layer_sweep_%a
#SBATCH --partition=general
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
#   sbatch --array=0 slurm/layer_sweep.sbatch
#   # then set LAYER_SWEEP_EXTRA_ARGS="--dry_run" below before submit
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

SSL_MODELS=(wav2vec2 hubert wavlm)
IDX=${SLURM_ARRAY_TASK_ID}
MODEL="${SSL_MODELS[$IDX]}"

# Slurm does not forward extra CLI args passed to sbatch.
# Set this before submission when needed (example: "--dry_run").
LAYER_SWEEP_EXTRA_ARGS=""

PROJECT_DIR="/home/mahadevballa23/scratch_space/signal-laundering-deepfake"
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

export HF_HOME="${HF_HOME:-$PROJECT_DIR/.cache/huggingface}"
if [ ! -d "$HF_HOME" ]; then
    echo "ERROR: HF_HOME directory '$HF_HOME' is missing."
    echo "Create it manually or set HF_HOME to an existing directory before submitting this job."
    exit 1
fi

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
    $LAYER_SWEEP_EXTRA_ARGS

echo "========================================================"
echo "  Done: $(date)"
echo "  JSON: outputs/sweep/${MODEL}/${MODEL}_layer_sweep.json"
echo "  Plot: outputs/sweep/${MODEL}/${MODEL}_layer_eer.png"
echo "========================================================"

#!/bin/bash
#SBATCH --job-name=stab_dryrun
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/stab_dryrun_%j.out
#SBATCH --error=logs/stab_dryrun_%j.err

set -euo pipefail

PROJECT_DIR="/home/mahadevballa23/scratch_space/signal-laundering-deepfake"
VENV_DIR="$PROJECT_DIR/.venv"
CACHE_DIR="$PROJECT_DIR/data/ssl_cache"

cd "$PROJECT_DIR"
mkdir -p logs "$CACHE_DIR"

if [ ! -f "$VENV_DIR/bin/activate" ]; then
    echo "ERROR: Missing venv at $VENV_DIR"
    exit 1
fi
source "$VENV_DIR/bin/activate"

echo ">>> Python: $(which python) | $(python --version)"
python -c "import torch; print(f'>>> torch {torch.__version__} | CUDA {torch.cuda.is_available()}')"
python -c "import numpy; print(f'>>> numpy {numpy.__version__}')"

# ── Submodule check ───────────────────────────────────────────────────────────
if [ ! -f "external/aasist/main.py" ]; then
    echo "ERROR: external/aasist submodule missing."
    exit 1
fi

# ── Init weights check ────────────────────────────────────────────────────────
if [ ! -f "models/wav2vec2_aasist_weighted.pth" ]; then
    echo "ERROR: models/wav2vec2_aasist_weighted.pth not found."
    echo "Copy it from local: scp backup_v1_phase2/models/wav2vec2_aasist_weighted.pth <hpc>:$(pwd)/models/"
    exit 1
fi

# ── HuggingFace offline ───────────────────────────────────────────────────────
export HF_HOME="$PROJECT_DIR/.cache/huggingface"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# ── Thread settings ───────────────────────────────────────────────────────────
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

echo "========================================================"
echo "  Job:    Stab Training DRY RUN"
echo "  GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo N/A)"
echo "  Node:   $(hostname)"
echo "  Start:  $(date)"
echo "========================================================"

python train_stab.py \
    --model         wav2vec2 \
    --backend       aasist \
    --init_weights  models/wav2vec2_aasist_weighted.pth \
    --pipelines     N \
    --depths        1 2 3 \
    --strength      M \
    --eval_depth    3 \
    --lambda_stab   0.1 \
    --lambda_clean  0.5 \
    --epochs        50 \
    --patience      10 \
    --eval_every    5 \
    --lr            1e-4 \
    --weight_decay  1e-4 \
    --batch_size    32 \
    --embed_dim     768 \
    --num_layers    12 \
    --data_root     data/LA \
    --cache_dir     "$CACHE_DIR" \
    --output_dir    outputs/stab_training \
    --dry_run

echo "========================================================"
echo "  Dry run complete: $(date)"
echo "========================================================"

# ── Verify expected outputs exist ─────────────────────────────────────────────
echo ""
echo "Checking expected outputs..."
RUN_DIR=$(ls -td outputs/stab_training/wav2vec2_aasist/*/ 2>/dev/null | head -1)
if [ -z "$RUN_DIR" ]; then
    echo "FAIL: No run directory created under outputs/stab_training/wav2vec2_aasist/"
    exit 1
fi
echo "Run dir: $RUN_DIR"

for f in config.json training_log.csv best_laund.pth best_clean.pth final.pth; do
    if [ -f "$RUN_DIR/$f" ]; then
        echo "  OK  $f"
    else
        echo "  MISSING  $f"
    fi
done

LW_COUNT=$(ls "$RUN_DIR/layer_weights/"*.json 2>/dev/null | wc -l)
DS_COUNT=$(ls "$RUN_DIR/dev_scores/"*.npz 2>/dev/null | wc -l)
echo "  layer_weights snapshots: $LW_COUNT"
echo "  dev_scores npz files:    $DS_COUNT"
echo ""
echo "If all files are OK above, submit the full lambda sweep:"
echo "  sbatch slurm/train_stab_wav2vec2_aasist.sbatch"

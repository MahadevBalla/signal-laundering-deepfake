#!/bin/bash
#SBATCH --job-name=stab_train_%a
#SBATCH --partition=a40
#SBATCH --qos=a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/stab_train_%a_%j.out
#SBATCH --error=logs/stab_train_%a_%j.err
#SBATCH --array=0-4

# Stability-regularized laundering-aware training
# Array index → lambda_stab
# 0 → 0.01
# 1 → 0.05
# 2 → 0.10
# 3 → 0.50
# 4 → 1.00
# Submit:
# sbatch slurm/train_stab.sbatch
# Single lambda:
# sbatch --array=2 slurm/train_stab.sbatch

set -euo pipefail

LAMBDAS=(0.01 0.05 0.10 0.50 1.00)
LAMBDA_STAB="${LAMBDAS[$SLURM_ARRAY_TASK_ID]}"

MODEL="wav2vec2"
BACKEND="aasist"

PROJECT_DIR="/home/gopalpatil/22d0296/projects/signal-laundering-deepfake"
VENV_DIR="$PROJECT_DIR/.venv"
CACHE_DIR="$PROJECT_DIR/data/ssl_cache"

cd "$PROJECT_DIR"

mkdir -p logs
mkdir -p "$CACHE_DIR"

source /lustre-flash/apps/spack/share/spack/setup-env.sh
spack load anaconda3@2023.09-0
eval "$(conda shell.bash hook)"

if [ ! -f "$VENV_DIR/bin/activate" ]; then
echo "ERROR: Missing venv at $VENV_DIR"
exit 1
fi

source "$VENV_DIR/bin/activate"

echo ">>> Python: $(which python)"
echo ">>> Version: $(python --version)"

export HF_HOME="$PROJECT_DIR/.cache/huggingface"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

mkdir -p "$HF_HOME"

echo "========================================================"
echo " Stability Training"
echo " Model        : $MODEL"
echo " Backend      : $BACKEND"
echo " lambda_stab  : $LAMBDA_STAB"
echo " Node         : $(hostname)"
echo " GPU          : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo N/A)"
echo " Start        : $(date)"
echo "========================================================"

python train_stab.py
--model "$MODEL"
--backend "$BACKEND"
--pipelines N
--depths 1 2 3
--strength M
--lambda_stab "$LAMBDA_STAB"
--lambda_clean 0.5
--epochs 50
--patience 10
--eval_every 5
--lr 1e-4
--batch_size 32
--embed_dim 768
--num_layers 12
--cache_dir "$CACHE_DIR"

echo "========================================================"
echo " Finished: $(date)"
echo "========================================================"

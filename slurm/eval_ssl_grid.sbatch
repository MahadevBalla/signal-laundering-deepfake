#!/bin/bash
#SBATCH --job-name=eval_ssl_%a
#SBATCH --partition=a40
#SBATCH --qos=a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=36:00:00
#SBATCH --output=logs/eval_ssl_%a_%j.out
#SBATCH --error=logs/eval_ssl_%a_%j.err
#SBATCH --array=0-8

set -euo pipefail

SSL_MODEL_KEYS=(
    wav2vec2 wav2vec2_aasist wav2vec2_rawnet2
    hubert hubert_aasist hubert_rawnet2_ssl
    wavlm wavlm_aasist wavlm_rawnet2
)

IDX=${SLURM_ARRAY_TASK_ID}
MODEL="${SSL_MODEL_KEYS[$IDX]}"
EXTRA_ARGS="${@:-}"

PROJECT_DIR="/home/gopalpatil/22d0296/projects/signal-laundering-deepfake"
VENV_DIR="$PROJECT_DIR/.venv"
cd "$PROJECT_DIR"
mkdir -p logs

# ── Spack + Anaconda (matches eval_rawnet2 / eval_aasist) ────────────────────
source /lustre-flash/apps/spack/share/spack/setup-env.sh
spack load anaconda3@2023.09-0
eval "$(conda shell.bash hook)"

# ── Activate venv ─────────────────────────────────────────────────────────────
if [ ! -f "$VENV_DIR/bin/activate" ]; then
    echo "ERROR: venv missing. Build on login node first."
    exit 1
fi
source "$VENV_DIR/bin/activate"
echo ">>> Python: $(which python) | $(python --version)"
python -c "import torch; print(f'>>> torch {torch.__version__} | CUDA {torch.cuda.is_available()}')"

# ── Submodule check ───────────────────────────────────────────────────────────
if [ ! -f "external/aasist/main.py" ]; then
    echo "ERROR: external/aasist missing. Run: git submodule update --init --recursive"
    exit 1
fi

# ── HuggingFace offline ───────────────────────────────────────────────────────
export HF_HOME="$PROJECT_DIR/.cache/huggingface"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
mkdir -p "$HF_HOME"

# ── Prevent CPU thread oversubscription ──────────────────────────────────────
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

# ── Weights check ─────────────────────────────────────────────────────────────
python - <<EOF
import sys
from pathlib import Path
from src.models.model_config import WEIGHTS
w = WEIGHTS.get("$MODEL")
if w and not Path(w).exists():
    print(f"ERROR: weights not found: {w}")
    print("Run train_ssl_backends.sbatch first.")
    sys.exit(1)
print(f"Weights OK: {w}")
EOF

echo "========================================================"
echo "  Job:     SSL Eval Grid"
echo "  Task ID: $IDX  |  Model: $MODEL"
echo "  GPU:     $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo N/A)"
echo "  Node:    $(hostname)  |  Start: $(date)"
echo "========================================================"

python eval_suite.py \
    --model     "$MODEL" \
    --data_root data/LA \
    --pipelines N M P \
    --strengths L M H \
    --depths    0 1 2 3 \
    --run_cka \
    --cka_max_eval 1000 \
    $EXTRA_ARGS

echo "========================================================"
echo "  Done: $(date)"
echo "  Results: outputs/eval_suite/$MODEL/master_results.csv"
echo "========================================================"

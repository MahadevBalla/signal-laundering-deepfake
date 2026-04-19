#!/bin/bash
#SBATCH --job-name=eval_ssl_%a
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=36:00:00
#SBATCH --output=logs/eval_ssl_%a_%j.out
#SBATCH --error=logs/eval_ssl_%a_%j.err
#SBATCH --array=0-8    # 9 SSL model variants

# ─────────────────────────────────────────────────────────────────────────────
# Full laundering evaluation grid for all SSL + backend model combinations.
# Requires trained weights (run train_ssl_backends.sbatch first).
#
# Array index -> model key mapping:
#   0  wav2vec2             (FFN weighted backend)
#   1  wav2vec2_aasist      (AASIST backend)
#   2  wav2vec2_rawnet2     (RawNet2 backend)
#   3  hubert               (FFN weighted backend)
#   4  hubert_aasist        (AASIST backend)
#   5  hubert_rawnet2_ssl   (RawNet2 backend)
#   6  wavlm                (FFN weighted backend)
#   7  wavlm_aasist         (AASIST backend)
#   8  wavlm_rawnet2        (RawNet2 backend)
#
# Each task also runs CKA representation drift analysis (--run_cka).
#
# Submit all 9:
#   sbatch slurm/eval_ssl_grid.sbatch
#
# Submit only wav2vec2 variants:
#   sbatch --array=0-2 slurm/eval_ssl_grid.sbatch
#
# Submit with extra args (e.g. limit eval size):
#   sbatch slurm/eval_ssl_grid.sbatch --max_eval 2000
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

SSL_MODEL_KEYS=(
    wav2vec2
    wav2vec2_aasist
    wav2vec2_rawnet2
    hubert
    hubert_aasist
    hubert_rawnet2_ssl
    wavlm
    wavlm_aasist
    wavlm_rawnet2
)

IDX=${SLURM_ARRAY_TASK_ID}
MODEL="${SSL_MODEL_KEYS[$IDX]}"

EXTRA_ARGS="${@:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

mkdir -p logs

module purge
module load python/3.11.14

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
mkdir -p "$HF_HOME"

echo "========================================================"
echo "  Job:     SSL Eval Grid"
echo "  Task ID: $IDX"
echo "  Model:   $MODEL"
echo "  GPU:     $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo N/A)"
echo "  Node:    $(hostname)"
echo "  Start:   $(date)"
echo "========================================================"

# Check weights before launching to give a clean error message
python - <<EOF
import sys
from pathlib import Path
from src.models.model_config import WEIGHTS
model = "$MODEL"
w = WEIGHTS.get(model)
if w and not Path(w).exists():
    print(f"ERROR: weights not found: {w}")
    print("Run train_ssl_backends.sbatch first.")
    sys.exit(1)
print(f"Weights OK: {w}")
EOF

python eval_suite.py \
    --model    "$MODEL" \
    --data_root data/ASVspoof2019/LA \
    --pipelines N M P \
    --strengths L M H \
    --depths 0 1 2 3 \
    --run_cka \
    --cka_max_eval 1000 \
    $EXTRA_ARGS

echo "========================================================"
echo "  Done: $(date)"
echo "  Results: outputs/eval_suite/$MODEL/master_results.csv"
echo "========================================================"

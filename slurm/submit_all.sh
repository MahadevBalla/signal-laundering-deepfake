#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# submit_all.sh
#
# Orchestrate the full experiment pipeline by submitting all SLURM jobs
# in the correct dependency order:
#
#   Stage 0 (no deps):  Verify AASIST + RawNet2 pretrained weights
#   Stage 1 (no deps):  Train all SSL + backend combinations (array job)
#   Stage 1 (no deps):  Layer sweep - single-layer FFN probes (array job)
#   Stage 2 (after 0):  Eval AASIST full grid
#   Stage 2 (after 0):  Eval RawNet2 full grid
#   Stage 3 (after 1):  Eval all SSL models full grid + CKA (array job)
#
# Usage:
#   chmod +x slurm/submit_all.sh
#   ./slurm/submit_all.sh
#
# Dry-run (print commands without submitting):
#   DRY_RUN=1 ./slurm/submit_all.sh
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(dirname "$SCRIPT_DIR")"

DRY_RUN="${DRY_RUN:-0}"

_sbatch() {
    if [ "$DRY_RUN" = "1" ]; then
        echo "[DRY-RUN] sbatch $*"
        echo "FAKE_JOB_ID"
    else
        sbatch "$@" | awk '{print $NF}'
    fi
}

echo "================================================"
echo "  Signal Laundering - Full HPC Submission"
echo "  $(date)"
echo "================================================"

# Stage 0: Raw audio model checks
echo ""
echo "Stage 0: Verify pretrained raw-audio weights"

JOB_AASIST_CHECK=$(_sbatch slurm/train_aasist.sbatch)
echo "  AASIST  check job: $JOB_AASIST_CHECK"

JOB_RN2_CHECK=$(_sbatch slurm/train_rawnet2.sbatch)
echo "  RawNet2 check job: $JOB_RN2_CHECK"

# Stage 1: SSL backend training
echo ""
echo "Stage 1a: Train SSL backends (array 0-8)"
JOB_SSL_TRAIN=$(_sbatch slurm/train_ssl_backends.sbatch)
echo "  SSL train array job: $JOB_SSL_TRAIN"

echo ""
echo "Stage 1b: Layer sweep (array 0-2)"
JOB_SWEEP=$(_sbatch slurm/layer_sweep.sbatch)
echo "  Layer sweep array job: $JOB_SWEEP"

# Stage 2: Raw audio evaluations
echo ""
echo "Stage 2a: Eval AASIST full grid (after Stage 0 AASIST check)"
JOB_EVAL_AASIST=$(_sbatch --dependency=afterok:${JOB_AASIST_CHECK} slurm/eval_aasist.sbatch)
echo "  AASIST eval job: $JOB_EVAL_AASIST"

echo ""
echo "Stage 2b: Eval RawNet2 full grid (after Stage 0 RawNet2 check)"
JOB_EVAL_RN2=$(_sbatch --dependency=afterok:${JOB_RN2_CHECK} slurm/eval_rawnet2.sbatch)
echo "  RawNet2 eval job: $JOB_EVAL_RN2"

# Stage 3: SSL evaluations
echo ""
echo "Stage 3: Eval all SSL models + CKA (after Stage 1a SSL training)"
JOB_EVAL_SSL=$(_sbatch --dependency=afterok:${JOB_SSL_TRAIN} slurm/eval_ssl_grid.sbatch)
echo "  SSL eval array job: $JOB_EVAL_SSL"

echo ""
echo "================================================"
echo "  All jobs submitted."
echo ""
echo "  Monitor:  squeue -u \$USER"
echo "  Logs:     logs/"
echo "  Results:  outputs/eval_suite/<model>/master_results.csv"
echo "================================================"

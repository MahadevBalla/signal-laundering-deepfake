# SLURM Batch Scripts

Complete set of HPC batch scripts for training and evaluation.
Place this `slurm/` directory at the root of the project.

---

## Quick Start

```bash
# 1. Make the orchestrator executable
chmod +x slurm/submit_all.sh

# 2. Dry-run to preview all sbatch commands
DRY_RUN=1 ./slurm/submit_all.sh

# 3. Submit everything with automatic dependencies
./slurm/submit_all.sh
```

---

## Before You Submit - Checklist

| Item | What to change |
|---|---|
| `module load python/3.11.14` | Change to your HPC's actual Python module (`module avail python`) |
| `--partition=gpu` | Change to your GPU partition name (e.g. `a100`, `dgx`, `gpu-short`) |
| `data/ASVspoof2019/LA` | Must point to your ASVspoof2019-LA dataset on the cluster |
| AASIST/RawNet2 weights | Download to `external/aasist/models/weights/` before Stage 0 |
| `--mem=` | Tune if your nodes have different RAM; SSL models need >=64G |

---

## Script Reference

### Training Scripts

#### `train_aasist.sbatch`
Verifies AASIST pretrained weights exist and runs a 200-sample dry-run eval.
AASIST does not require a training loop - it uses weights from the external submodule.

```bash
sbatch slurm/train_aasist.sbatch
```

#### `train_rawnet2.sbatch`
Same as above for RawNet2 (`external/aasist/models/weights/RawNet2.pth`).

```bash
sbatch slurm/train_rawnet2.sbatch
```

#### `train_ssl_backends.sbatch` - Array job (0-8)
Trains all 9 SSL frontend x backend combinations with a frozen SSL frontend.

| Array Task | SSL Frontend | Backend | Saved Weight |
|---|---|---|---|
| 0 | wav2vec2 | FFN (weighted) | `models/wav2vec2_ffn_weighted.pth` |
| 1 | wav2vec2 | AASIST | `models/wav2vec2_aasist_weighted.pth` |
| 2 | wav2vec2 | RawNet2 | `models/wav2vec2_rawnet2_weighted.pth` |
| 3 | hubert | FFN (weighted) | `models/hubert_ffn_weighted.pth` |
| 4 | hubert | AASIST | `models/hubert_aasist_weighted.pth` |
| 5 | hubert | RawNet2 | `models/hubert_rawnet2_weighted.pth` |
| 6 | wavlm | FFN (weighted) | `models/wavlm_ffn_weighted.pth` |
| 7 | wavlm | AASIST | `models/wavlm_aasist_weighted.pth` |
| 8 | wavlm | RawNet2 | `models/wavlm_rawnet2_weighted.pth` |

```bash
# Submit all 9
sbatch slurm/train_ssl_backends.sbatch

# Submit only wav2vec2 variants (tasks 0-2)
sbatch --array=0-2 slurm/train_ssl_backends.sbatch
```

Training config: Adam lr=1e-4, batch=32, CE loss, 50 epochs, early stopping patience=10.

#### `layer_sweep.sbatch` - Array job (0-2)
Runs `layer_sweep.py` for each SSL model. Trains a single-layer FFN probe for all 12
transformer layers and plots EER vs. layer. Use this before `train_ssl_backends.sbatch`
to identify the most discriminative layer (Objective 2: Quantify Representation Drift).

| Array Task | Model | Outputs |
|---|---|---|
| 0 | wav2vec2 | `outputs/sweep/wav2vec2/wav2vec2_layer_eer.png` |
| 1 | hubert | `outputs/sweep/hubert/hubert_layer_eer.png` |
| 2 | wavlm | `outputs/sweep/wavlm/wavlm_layer_eer.png` |

```bash
sbatch slurm/layer_sweep.sbatch
sbatch --array=0 slurm/layer_sweep.sbatch          # wav2vec2 only
sbatch --array=0 slurm/layer_sweep.sbatch --dry_run  # 2 epochs, 200 utterances
```

---

### Evaluation Scripts

All eval scripts run the full laundering grid:
- **3 pipelines**: N (noise-based), M (music-based), P (playback-based)
- **3 strengths**: L (low), M (medium), H (high)
- **4 depths**: 0 (clean), 1, 2, 3

Results go to `outputs/eval_suite/<model>/master_results.csv`.
Supports resume - already-completed conditions are skipped automatically.

#### `eval_aasist.sbatch`

```bash
sbatch slurm/eval_aasist.sbatch
sbatch slurm/eval_aasist.sbatch --max_eval 1000   # faster subset
```

#### `eval_rawnet2.sbatch`

```bash
sbatch slurm/eval_rawnet2.sbatch
```

#### `eval_ssl_grid.sbatch` - Array job (0-8)
Evaluates all 9 SSL model variants over the full grid, plus CKA representation drift analysis.

| Array Task | Model Key | Backend |
|---|---|---|
| 0 | `wav2vec2` | FFN |
| 1 | `wav2vec2_aasist` | AASIST |
| 2 | `wav2vec2_rawnet2` | RawNet2 |
| 3 | `hubert` | FFN |
| 4 | `hubert_aasist` | AASIST |
| 5 | `hubert_rawnet2_ssl` | RawNet2 |
| 6 | `wavlm` | FFN |
| 7 | `wavlm_aasist` | AASIST |
| 8 | `wavlm_rawnet2` | RawNet2 |

```bash
sbatch slurm/eval_ssl_grid.sbatch
sbatch --array=0-2 slurm/eval_ssl_grid.sbatch       # wav2vec2 only
```

> **Requires**: trained weights from `train_ssl_backends.sbatch` must exist first.

---

## Dependency Graph

```
Stage 0 (no deps):
  train_aasist.sbatch   --+
  train_rawnet2.sbatch  --+--> Stage 2
  train_ssl_backends.sbatch --> Stage 3
  layer_sweep.sbatch       (independent, informational)

Stage 2 (after Stage 0):
  eval_aasist.sbatch
  eval_rawnet2.sbatch

Stage 3 (after train_ssl_backends):
  eval_ssl_grid.sbatch
```

---

## Output Structure

```
outputs/
└── eval_suite/
    └── <model>/
        ├── master_results.csv          <- all EER / tDCF values
        ├── eval_suite.log
        ├── clean/k0/                   <- clean baseline plots
        ├── N/L/k1/ ... N/H/k3/        <- per-condition DET curves
        ├── M/ ... P/ ...
        ├── cka/                        <- CKA + cosine drift heatmaps (SSL only)
        └── summary/
            ├── collapse_curves.png
            ├── strength_heatmap_k1.png
            ├── strength_heatmap_k3.png
            ├── aurc_comparison.png
            └── <model>_framework_metrics.json

models/
├── wav2vec2_ffn_weighted.pth
├── wav2vec2_aasist_weighted.pth
├── wav2vec2_rawnet2_weighted.pth
├── hubert_ffn_weighted.pth
├── hubert_aasist_weighted.pth
├── hubert_rawnet2_weighted.pth
├── wavlm_ffn_weighted.pth
├── wavlm_aasist_weighted.pth
├── wavlm_rawnet2_weighted.pth
└── (layer sweep) <model>_ffn_layer{0-11}.pth
```

---

## Monitoring

```bash
watch -n 30 squeue -u $USER
tail -f logs/train_ssl_3_<jobid>.out
srun --jobid=<JOBID> --pty nvidia-smi
```

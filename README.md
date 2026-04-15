# Signal Laundering as a Threat Model for Deepfake Audio Detection

This project studies how deepfake audio detectors break down when audio passes through real-world signal chains — codec compression, VoIP transmission, and physical replay — before reaching the detector. We call these chains **laundering pipelines** and evaluate detectors across increasing depths and strengths of laundering.

Models evaluated: **AASIST**, **RawNet2** (waveform-based), and **Wav2Vec2**, **HuBERT**, **WavLM** (SSL frontends with trainable backends: FFN, AASIST-style, or RawNet2-style).

## Setup

Clone with submodules (includes the AASIST/RawNet2 codebase):

```bash
git clone --recurse-submodules https://github.com/MahadevBalla/signal-laundering-deepfake.git
cd signal-laundering-deepfake
```

Install dependencies:

```bash
uv venv .venv && source .venv/bin/activate
uv sync
# OR
uv pip install -r requirements.txt
```

For a one-command end-to-end run of the SSL frontend + backend + laundering evaluation pipeline, use:

```bash
python scripts/run_full_pipeline.py --model wav2vec2 --backend aasist --install-deps --sync-submodules --noise-mode white --run-cka
```

This runner can:

- install Python dependencies,
- initialize missing git submodules,
- validate your dataset layout,
- optionally fetch SPIB and QUT noise data,
- run layer-wise analysis,
- train the selected weighted backend,
- run the full laundering evaluation suite with summary plots and optional CKA.

The default and recommended stable path is `--noise-mode white`, which uses the built-in white-noise fallback and does not require external noise downloads.

## Dataset

We use the **ASVspoof 2019 Logical Access (LA)** dataset.

Download from: [https://datashare.ed.ac.uk/handle/10283/3336](https://datashare.ed.ac.uk/handle/10283/3336)

Place it at `data/ASVspoof2019/LA` so the directory looks like:

```md
data/ASVspoof2019/LA/
  ASVspoof2019_LA_train/flac/
  ASVspoof2019_LA_dev/flac/
  ASVspoof2019_LA_eval/flac/
  ASVspoof2019_LA_cm_protocols/
  ASVspoof2019_LA_asv_scores/
```

Optional: place SPIB and QUT-NOISE files under `data/noise/` for realistic noise in Pipelines N and P. Without these files, white noise is used as the default fallback.

Expected layout:

```md
data/noise/
  SPIB/
    *.mat
  QUT-NOISE/
    QUT-NOISE/
      *.wav
      labels/
      impulses/
```

This dataset is not auto-downloaded by the bootstrap script because it is distributed separately from the repo and may require manual access/acceptance steps.

You can fetch both datasets into the expected locations with:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/download_noise.ps1
```

Or, cross-platform:

```bash
python scripts/noise_setup.py --datasets SPIB QUT
```

The script downloads:

- SPIB noise files from `http://spib.linse.ufsc.br/noise.html` into `data/noise/SPIB/`
- QUT-NOISE archives from `https://research.qut.edu.au/saivt/databases/qut-noise-databases-and-protocols/` and extracts them under `data/noise/QUT-NOISE/`

If you download manually, unzip all `QUT-NOISE*.zip` archives into the same `data/noise/QUT-NOISE/` directory.

## Models and pre-trained weights

**AASIST and RawNet2** come with pre-trained weights in `external/aasist/models/weights/`. No training needed — go straight to evaluation.

**SSL models** (Wav2Vec2, HuBERT, WavLM) use frozen HuggingFace encoders. You train only the backend on top (FFN, AASIST-style, or RawNet2-style). The SSL encoder itself is never trained.

## SSL backend training

### Step 1 — find the best layer (optional but recommended)

This trains an FFN on each of the 12 transformer layers separately and plots EER per layer, replicating the layer-wise analysis from El Kheir et al. (NAACL 2025). The paper's finding — lower layers tend to be more discriminative — may or may not hold on your setup.

```bash
# Smoke test first (fast, ~5 min)
python layer_sweep.py --model wav2vec2 --dry_run

# Full sweep (~4h on GPU per model)
python layer_sweep.py --model wav2vec2
python layer_sweep.py --model hubert
python layer_sweep.py --model wavlm
```

Results saved to `outputs/sweep/<model>_layer_sweep.json` and a bar chart PNG.

### Step 2 — train weighted backend model(s)

This trains a model that learns to weight all 12 SSL layers jointly.

```bash
# FFN backend (default)
python train_ssl_backend.py --model wav2vec2 --mode weighted --backend ffn
python train_ssl_backend.py --model hubert   --mode weighted --backend ffn
python train_ssl_backend.py --model wavlm    --mode weighted --backend ffn

# Optional backend variants
python train_ssl_backend.py --model wav2vec2 --mode weighted --backend aasist
python train_ssl_backend.py --model wav2vec2 --mode weighted --backend rawnet2
```

Weights are saved to:

- `models/<model>_ffn_weighted.pth`
- `models/<model>_aasist_weighted.pth`
- `models/<model>_rawnet2_weighted.pth`

If you prefer a specific layer from the sweep:

```bash
python train_ssl_backend.py --model wav2vec2 --mode single --layer 3
```

## Running evaluations

### Quick single condition

```bash
python run.py --model aasist --depth 0                          # clean baseline
python run.py --model aasist --pipeline M --depth 2 --strength H
```

### Full laundering evaluation suite

Evaluates all pipeline × strength × depth combinations sequentially. Laundering is applied on-the-fly — no laundered audio is ever written to disk. Results are appended to `outputs/eval_suite/<model>/master_results.csv` after each condition, so a crashed job can be safely resumed.

```bash
# Smoke test (200 utterances)
python eval_suite.py --model aasist --dry_run
python eval_suite.py --model wav2vec2 --dry_run --run_cka

# Full run
python eval_suite.py --model aasist
python eval_suite.py --model rawnet2
python eval_suite.py --model wav2vec2 --run_cka
python eval_suite.py --model hubert   --run_cka
python eval_suite.py --model wavlm    --run_cka
```

`--run_cka` computes CKA and cosine similarity across transformer layers at each laundering depth (SSL models only). Uses ~74 MB RAM peak and saves heatmap PNGs alongside the JSON results.

### On the HPC cluster (Slurm)

```bash
sbatch scripts/run_eval.sbatch aasist
sbatch scripts/run_eval.sbatch rawnet2
sbatch scripts/run_eval.sbatch wav2vec2 --run_cka
sbatch scripts/run_eval.sbatch hubert   --run_cka
sbatch scripts/run_eval.sbatch wavlm    --run_cka
```

## Laundering pipelines

| Pipeline | Stages | Models real-world threat |
| --- | --- | --- |
| **N** — Network | Opus codec → packet loss + LP filter → additive noise | VoIP, phone calls |
| **M** — Media | MP3 → AAC → Opus re-encoding | Social media, streaming platforms |
| **P** — Physical | Speaker EQ → room reverberation (RIR) → mic HP + noise | Replay attacks |

Each pipeline has three strengths (**L**, **M**, **H**) and four depths (**0**–**3**, where 0 = clean and depth k means the first k stages are applied cumulatively).

## Outputs

After a full run, `outputs/eval_suite/<model>/` contains:

```md
master_results.csv          all EER and tDCF results
summary/
  <model>_collapse_curves.png     EER vs depth per pipeline
  <model>_strength_heatmap_k1.png EER by pipeline x strength at depth 1
  <model>_strength_heatmap_k3.png EER by pipeline x strength at depth 3
  aurc_comparison.png             AURC bar chart
  <model>_framework_metrics.json  AURC, collapse depth kc, collapse strength lc
cka/
  <pipeline>_<strength>_cka.json
  <pipeline>_<strength>_cosine.json
  cka_heatmap_*.png
  cosine_drift_*.png
```

## Repository structure

```md
signal-laundering-deepfake/
├── configs/                         # YAML configs (SSL + laundering params)
│   ├── wav2vec2_probe.yaml
│   ├── hubert_probe.yaml
│   ├── wavlm_probe.yaml
│   ├── N_params.yaml
│   ├── M_params.yaml
│   └── P_params.yaml
│
├── data/                            # datasets (not tracked fully)
│   ├── ASVspoof2019/
│   │   └── LA/
│   │       ├── ASVspoof2019_LA_train/
│   │       ├── ASVspoof2019_LA_dev/
│   │       ├── ASVspoof2019_LA_eval/
│   │       └── ASVspoof2019_LA_cm_protocols/
│   └── noise/                       # optional noise files
│
├── external/                        # submodules / external code
│   ├── aasist/                      # AASIST + RawNet2 (with pretrained weights)
│   └── asvspoof2021/                # evaluation tools (if used)
│
├── models/                          # trained SSL backend weights
│   ├── wav2vec2_*_weighted.pth
│   ├── hubert_*_weighted.pth
│   └── wavlm_*_weighted.pth
│
├── outputs/                         # experiment outputs
│   ├── eval_suite/
│   │   ├── aasist/
│   │   ├── rawnet2/
│   │   ├── wav2vec2/
│   │   ├── hubert/
│   │   └── wavlm/
│   └── sweep/                       # layer sweep results
│
├── scripts/                         # HPC / automation scripts
│   ├── run_eval.sbatch
│   └── run_hubert_rawnet2.sbatch
│
├── src/                             # core source code
│   ├── evaluation/                  # metrics, plots, CKA
│   │   ├── metrics.py
│   │   ├── plots.py
│   │   ├── cka.py
│   │   └── results_writer.py
│   │
│   ├── laundering/                  # laundering pipelines
│   │   ├── core.py
│   │   ├── pipeline_N.py
│   │   ├── pipeline_M.py
│   │   ├── pipeline_P.py
│   │   └── utils.py
│   │
│   └── models/                      # model wrappers + SSL framework
│       ├── registry.py              # model name → wrapper mapping
│       ├── model_config.py          # config + weight resolution
│       ├── dataset.py               # ASVspoof dataset loader
│       ├── backends.py              # FFN / AASIST / RawNet2 backends
│       ├── ssl_frontend.py          # frozen SSL encoders (layer access)
│       ├── ssl_eval_wrapper.py      # SSL model evaluation wrapper
│       ├── ssl_probe_wrapper.py     # legacy probe wrapper
│       ├── aasist_wrapper.py        # raw AASIST wrapper
│       ├── rawnet2_wrapper.py       # raw RawNet2 + HuBERT-RawNet2
│       └── hubert_frontend.py       # HuBERT-specific frontend utils
│
├── train_ssl_backend.py             # train SSL + backend models
├── train_probe.py                   # legacy linear probe training
├── layer_sweep.py                   # layer-wise SSL analysis
├── eval_suite.py                    # full laundering evaluation
├── run.py                           # single-condition evaluation
│
├── pyproject.toml
├── requirements.txt
├── uv.lock
├── .gitmodules
└── README.md
```

# Signal Laundering as a Threat Model for Deepfake Audio Detection

This repository studies **signal laundering** as a structured threat model for deepfake audio detection.

## Setup

Clone with submodules:

```bash
git clone --recurse-submodules https://github.com/MahadevBalla/signal-laundering-deepfake.git
```

If already cloned:

```bash
git submodule update --init --recursive
```

## Dataset

Experiments use the **ASVspoof 2019 Logical Access (LA)** dataset.

Download from: [https://datashare.ed.ac.uk/handle/10283/3336](https://datashare.ed.ac.uk/handle/10283/3336)

## Run with HuBERT + RawNet2

The project now supports a `hubert-rawnet2` mode that applies HuBERT feature
extraction before RawNet2 scoring.

```bash
python run.py --model hubert-rawnet2 --data_root data/LA/LA --depth 0
```

Single command for explicit frontend -> backend chaining:

```bash
python run.py --model rawnet2 --frontend_model hubert --frontend_ckpt facebook/hubert-base-ls960 --data_root data/LA/LA --depth 0
```

Notes:
- First run downloads the HuBERT checkpoint from Hugging Face.
- RawNet2 weights are still loaded from `external/aasist/config/RawNet2_baseline.conf`.

## Run on SPIT GPU Cluster (Slurm)

Use the provided Slurm job script and submit with one command:

```bash
sbatch scripts/run_hubert_rawnet2.sbatch /absolute/path/to/signal-laundering-deepfake
```

If you are already in the project directory:

```bash
sbatch scripts/run_hubert_rawnet2.sbatch
```

The script handles:
- module load (`python/3.11.14`)
- virtual environment creation with `uv`
- dependency install from `requirements.txt`
- HuBERT frontend + RawNet2 backend execution

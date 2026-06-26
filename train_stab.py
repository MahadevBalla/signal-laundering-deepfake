"""
train_stab.py
=============
Laundering-aware backend training with embedding stability regularisation.

Loss formulation
----------------
  L_bce_laund = CrossEntropy(classify(h_agg(X^k)),  y)   # detect laundered
  L_bce_clean = CrossEntropy(classify(h_agg(x^0)),  y)   # detect clean (anti-forgetting)
  L_stab      = MSE(pool(h_agg(x^0)), pool(h_agg(X^k)))  # representation stability
  L_total = L_bce_laund + λ_clean · L_bce_clean + λ_stab · L_stab

Where h_agg = Σ softmax(α_j) h_j  (weighted sum of frozen SSL layers).

Gradient pathway for L_stab
----------------------------
  SSL encoder frozen → raw SSL layer outputs h_j have no gradient.
  The states_* tensors are produced under torch.no_grad() and are leaf
  tensors. However h_agg = Σ softmax(α_j) h_j depends on learnable α_j
  (layer_weights), so PyTorch tracks the gradient through w=softmax(α_j)
  even when the stacked layer tensor has no grad_fn. L_stab therefore
  correctly pushes α_j toward layers whose h_j is stable under laundering,
  matching the lower-layer CKA findings from Phase 2.

Why aggregate_and_classify bypasses backend.forward()
------------------------------------------------------
  We need h_agg as an explicit intermediate for L_stab. Calling
  backend.forward() returns only logits and discards h_agg. The outer
  backend classes (WeightedAggregationBackend, SSLWithAASIST, SSLWithRawNet2)
  do exactly _weighted_sum → sub-backend and nothing else before that step.
  aggregate_and_classify replicates this split. If pre-aggregation transforms
  are ever added to those classes, this function MUST be updated.

Checkpoint selection
--------------------
  Primary:   best_laund.pth  — best checkpoint by laundered dev EER
                               Use this for eval_suite.py --weights
  Secondary: best_clean.pth  — best checkpoint by clean dev EER
                               Forgetting comparison baseline

Early stopping
--------------
  patience_ctr increments only on eval epochs (every --eval_every epochs)
  where dev_eer_laund did NOT improve. Non-eval epochs leave it unchanged.
  With --patience 10 and --eval_every 5, stopping triggers after 10
  consecutive non-improving eval events (up to 50 epochs of no progress).

Outputs (outputs/stab_training/<model>_<backend>/<run_id>/)
------------------------------------------------------------
  config.json                all hyperparameters + run metadata
  training_log.csv           per-epoch: all loss components, EERs, patience
  best_laund.pth             best by laundered EER  (use with eval_suite.py)
  best_clean.pth             best by clean EER  (forgetting comparison)
  final.pth                  last epoch weights
  checkpoints/epoch_N.pth    periodic full checkpoints (weights + optimizer + state)
  layer_weights/epoch_N.json α_j snapshots for tracking shift during training
  dev_scores/best_laund.npz  score arrays at the best_laund checkpoint
  dev_scores/best_clean.npz  score arrays at the best_clean checkpoint

Usage
-----
  # Standard experiment (wav2vec2+aasist, Pipeline N, λ_stab=0.1)
  python train_stab.py --model wav2vec2 --backend aasist --pipelines N

  # Lambda sweep (run one per job on HPC)
  for lam in 0.01 0.05 0.1 0.5 1.0:
      python train_stab.py --model wav2vec2 --backend aasist --lambda_stab $lam

  # Dry run (200 utterances, 2 epochs, fast verification)
  python train_stab.py --model wav2vec2 --backend aasist --dry_run

  # Fine-tune from clean-trained checkpoint (optional)
  python train_stab.py --model wav2vec2 --backend aasist \\
      --init_weights models/wav2vec2_aasist_weighted.pth

  # Evaluate trained model on full laundering grid
  python eval_suite.py --model wav2vec2_aasist \\
      --weights outputs/stab_training/wav2vec2_aasist/<run_id>/best_laund.pth
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# project imports
AASIST_ROOT = Path(__file__).resolve().parent / "external" / "aasist"
if str(AASIST_ROOT) not in sys.path:
    sys.path.insert(0, str(AASIST_ROOT))

from src.models.ssl_frontend import SSLFrontend
from src.models.embedding_cache import EmbeddingCache
from src.models.backends import (
    WeightedAggregationBackend,
    SSLWithAASIST,
    SSLWithRawNet2,
    _weighted_sum,
)
from src.laundering import LaunderingEngine
from src.evaluation.metrics import compute_eer
from src.models.dataset import WavDataset, PairedLaunderDataset

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", module="transformers")

from transformers.utils import logging as hf_logging

hf_logging.set_verbosity_error()

LABEL_MAP = {"bonafide": 0, "spoof": 1}
_MAX_PIPELINE_DEPTH = 3  # N/M/P each have exactly 3 stages

# Maps (ssl_model, backend) → eval_suite.py --model registry key.
# ffn backend uses the bare model name; aasist/rawnet2 use compound names.
# hubert+rawnet2 carries the _ssl suffix to distinguish from the waveform
# HuBERT-RawNet2 hybrid (hubert-rawnet2) in registry.py.
_REGISTRY_KEY: dict[tuple[str, str], str] = {
    ("wav2vec2", "ffn"): "wav2vec2",
    ("wav2vec2", "aasist"): "wav2vec2_aasist",
    ("wav2vec2", "rawnet2"): "wav2vec2_rawnet2",
    ("hubert", "ffn"): "hubert",
    ("hubert", "aasist"): "hubert_aasist",
    ("hubert", "rawnet2"): "hubert_rawnet2_ssl",
    ("wavlm", "ffn"): "wavlm",
    ("wavlm", "aasist"): "wavlm_aasist",
    ("wavlm", "rawnet2"): "wavlm_rawnet2",
}


# Configuration dataclass — bundles per-experiment training constants so
# train_epoch() stays within a sane parameter count.
@dataclass
class TrainConfig:
    """Immutable per-run training settings passed into train_epoch."""

    engine: LaunderingEngine | None  # None when using PairedLaunderDataset
    pipelines: list[str]
    depths: list[int]
    strength: str
    lambda_stab: float
    lambda_clean: float
    rng: np.random.Generator
    use_cache: bool


# Argument parsing
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Model
    p.add_argument("--model", required=True, choices=["wav2vec2", "hubert", "wavlm"])
    p.add_argument("--backend", default="aasist", choices=["ffn", "aasist", "rawnet2"])
    p.add_argument(
        "--init_weights",
        default=None,
        help="Optional path to a clean-trained checkpoint to fine-tune from.",
    )

    # Laundering during training
    p.add_argument(
        "--pipelines",
        nargs="+",
        default=["N"],
        choices=["N", "M", "P"],
        help="Pipelines to sample during training. Each batch draws one uniformly.",
    )
    p.add_argument(
        "--depths",
        nargs="+",
        type=int,
        default=[1, 2, 3],
        help=(
            "Laundering depths to sample per batch. Must be in [1, 3]. "
            "Depth 0 (clean) is excluded — it is handled separately via L_bce_clean."
        ),
    )
    p.add_argument(
        "--strength",
        default="M",
        choices=["L", "M", "H"],
        help="Fixed laundering strength during training.",
    )
    p.add_argument(
        "--eval_depth",
        type=int,
        default=None,
        help=(
            "Laundering depth used for dev EER and checkpoint selection. "
            "Default: max(--depths). Must be in [1, 3]. "
            "The eval pipeline is always --pipelines[0]."
        ),
    )

    # Loss weights
    p.add_argument(
        "--lambda_stab",
        type=float,
        default=0.1,
        help="Weight for L_stab. Recommended sweep: [0.01, 0.05, 0.1, 0.5, 1.0].",
    )
    p.add_argument(
        "--lambda_clean",
        type=float,
        default=0.5,
        help="Weight for L_bce_clean (anti-forgetting). 0.0 disables it.",
    )

    # Optimiser
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)

    # Training schedule
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument(
        "--patience",
        type=int,
        default=10,
        help=(
            "Early stopping patience: number of consecutive eval events "
            "(each occurring every --eval_every epochs) without improvement "
            "on dev_eer_laund before stopping."
        ),
    )
    p.add_argument(
        "--eval_every",
        type=int,
        default=5,
        help="Compute dev EER every N epochs. Last epoch always evaluated.",
    )
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--embed_dim", type=int, default=768)
    p.add_argument("--num_layers", type=int, default=12)

    # Data
    p.add_argument("--data_root", default="data/ASVspoof2019/LA")
    p.add_argument("--config_dir", default="configs")
    p.add_argument("--max_train", type=int, default=None)
    p.add_argument("--max_dev", type=int, default=None)

    # Caching
    p.add_argument("--cache_dir", default="data/ssl_cache")
    p.add_argument("--no_cache", action="store_true")

    # Outputs and checkpointing
    p.add_argument("--output_dir", default="outputs/stab_training")
    p.add_argument(
        "--run_id",
        default=None,
        help="Unique run ID. Auto-generated from hyperparams if not set.",
    )
    p.add_argument(
        "--ckpt_every",
        type=int,
        default=10,
        help="Save a full checkpoint (weights + optimizer + state) every N epochs.",
    )
    p.add_argument(
        "--lw_every",
        type=int,
        default=5,
        help="Save SSL layer weight snapshot every N epochs.",
    )

    # Precomputed laundering (Patch 1)
    p.add_argument(
        "--precomputed_root",
        default=None,
        help=(
            "Root directory containing precomputed laundering datasets produced "
            "by precompute_laundering.py. Expected sub-directories: "
            "<PIPELINE>_d<DEPTH>_M/ (e.g. N_d1_M, N_d2_M, N_d3_M). "
            "When set, online laundering is fully disabled and the "
            "LaunderingEngine is not instantiated."
        ),
    )

    # Misc
    p.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from the most recent checkpoint in the run dir.",
    )
    p.add_argument(
        "--dry_run",
        action="store_true",
        help="Use 200 utterances and 2 epochs for fast end-to-end verification.",
    )
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _validate_args(args: argparse.Namespace, eval_depth: int) -> None:
    """Catch invalid argument combinations before any compute starts."""
    bad_depths = [d for d in args.depths if d < 1 or d > _MAX_PIPELINE_DEPTH]
    if bad_depths:
        raise ValueError(
            f"--depths must be in [1, {_MAX_PIPELINE_DEPTH}]. Got: {bad_depths}. "
            "Depth 0 is always the clean baseline handled via L_bce_clean; "
            "do not include it in --depths."
        )
    if eval_depth < 1 or eval_depth > _MAX_PIPELINE_DEPTH:
        raise ValueError(
            f"--eval_depth must be in [1, {_MAX_PIPELINE_DEPTH}]. Got: {eval_depth}."
        )
    if eval_depth not in args.depths:
        warnings.warn(
            f"--eval_depth={eval_depth} is not in --depths={args.depths}. "
            "Checkpoint selection will use a laundering depth the model was "
            "never trained on. This may be intentional (held-out eval condition) "
            "but must be documented explicitly.",
            stacklevel=3,
        )


# Logging
def setup_logging(run_dir: Path) -> logging.Logger:
    log = logging.getLogger("train_stab")
    log.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S")
    fh = logging.FileHandler(run_dir / "train_stab.log", mode="a")
    fh.setFormatter(fmt)
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    if not log.handlers:
        log.addHandler(fh)
        log.addHandler(ch)
    return log


# Core training utilities
def aggregate_and_classify(
    backend: nn.Module,
    layer_states: dict[int, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Expose h_agg and logits separately so L_stab can act on h_agg.

    This manually replicates what backend.forward() does, because we need
    the intermediate h_agg = Σ softmax(α_j) h_j for the stability loss.
    Calling backend.forward() would discard h_agg.

    The outer backend classes do exactly _weighted_sum → sub-backend and
    nothing else before that step (verified against backends.py). If
    pre-aggregation transforms are ever added, this function MUST be updated.

    torch.compile() wraps the backend in OptimizedModule. We use
    getattr(backend, "_orig_mod", backend) to recover the original module for
    isinstance checks. Gradient flow is unaffected: layer_weights is the same
    parameter tensor regardless of which wrapper we access it through.

    Note on gradient flow: layer_states tensors carry no grad_fn (produced
    under torch.no_grad()). However h_agg depends on raw.layer_weights
    (a learnable parameter), so PyTorch correctly tracks the gradient through
    w=softmax(layer_weights). L_stab produces a valid gradient wrt α_j.

    Returns:
        h_agg:  [B, T, D]  weighted-aggregated SSL features
        logits: [B, 2]     classifier output (class 0 = bonafide)
    """
    raw = getattr(backend, "_orig_mod", backend)
    h_agg = _weighted_sum(layer_states, raw.layer_weights)  # [B, T, D]

    if isinstance(raw, WeightedAggregationBackend):
        logits = raw.ffn(h_agg)
    elif isinstance(raw, SSLWithAASIST):
        logits = raw.backend(h_agg)
    elif isinstance(raw, SSLWithRawNet2):
        logits = raw.backend(h_agg)
    else:
        raise TypeError(
            f"Unsupported backend type after _orig_mod unwrap: {type(raw).__name__}. "
            "Expected WeightedAggregationBackend, SSLWithAASIST, or SSLWithRawNet2. "
            "Update aggregate_and_classify() if a new backend type is added."
        )
    return h_agg, logits


def stability_loss(h_clean: torch.Tensor, h_laund: torch.Tensor) -> torch.Tensor:
    """
    L_stab = MSE between mean-pooled weighted aggregations.

    h_clean, h_laund: [B, T, D].
    Mean-pool over T → utterance-level representations [B, D], then MSE.

    Mean pooling is consistent with the CKA/cosine analysis in Phase 2
    and is invariant to minor sequence-length differences introduced by
    laundering (e.g. codec resampling adding/removing a few frames).
    The MSE objective is scale-sensitive, but L_bce simultaneously enforces
    class-conditional discrimination, preventing representational collapse
    to a constant. Monitor SCI from score_stats.json post-training to verify.
    """
    return F.mse_loss(h_clean.mean(dim=1), h_laund.mean(dim=1))


def _num_workers() -> int:
    """Return a safe DataLoader num_workers bounded by available CPUs."""
    return min(8, os.cpu_count() or 1)


def train_epoch(   
    backend: nn.Module,
    frontend: SSLFrontend,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    cfg: TrainConfig,
    device: str,
) -> dict:
    """
    One training epoch.

    Laundering parameters are bundled in cfg (TrainConfig) rather than passed
    individually, keeping the signature within a manageable parameter count.

    Two data paths are supported:

    Precomputed path (cfg.engine is None):
        loader yields (clean_wav, laund_wav, utt_id, src, key) 5-tuples from
        PairedLaunderDataset. clean and laundered correspond to the same
        utterance by construction, so alignment is guaranteed under shuffle=True.

    Online path (cfg.engine is not None):
        loader yields the standard WavDataset 4-tuple (clean_wav, utt_id, src, key).
        A laundering function is sampled per batch and applied on CPU.

    Both paths produce identical downstream code from the SSL forward pass
    onward, so all loss components (L_bce_laund, L_bce_clean, L_stab) are
    computed identically.

    Returns dict of batch-averaged loss components and laundered accuracy.
    """
    backend.train()
    sum_bce_laund = sum_bce_clean = sum_stab = sum_total = 0.0
    n_correct = n_total = 0

    use_precomputed = cfg.engine is None

    for batch in tqdm(loader, desc="train", leave=False):
        if use_precomputed:
            # PairedLaunderDataset — 5-tuple, alignment guaranteed by __getitem__.
            batch_x, batch_laund_cpu, utt_ids, _srcs, keys = batch
            batch_x = batch_x.to(device, non_blocking=True)
            batch_laund = batch_laund_cpu.to(device, non_blocking=True)
        else:
            # WavDataset — 4-tuple. Sample condition and launder on CPU.
            batch_x, utt_ids, _srcs, keys = batch
            batch_x = batch_x.to(device, non_blocking=True)
            # Sample laundering condition for this batch uniformly at random.
            # rng state is checkpointed so the corruption sequence is
            # reproducible across resumes.
            pipeline = str(cfg.rng.choice(cfg.pipelines))
            depth = int(cfg.rng.choice(cfg.depths))
            launder_fn = cfg.engine.get_batch_fn(pipeline, depth, cfg.strength)
            # depth >= 1 is guaranteed by _validate_args, so launder_fn is never
            # None. Laundering runs on CPU (ffmpeg/scipy roundtrips). Output is
            # always float32 (core.py guarantees this via
            # torch.from_numpy(...).float()).
            batch_laund = launder_fn(batch_x.cpu()).to(device, non_blocking=True)

        labels = torch.tensor(
            [LABEL_MAP[k] for k in keys], dtype=torch.long, device=device
        )

        with torch.no_grad():
            if cfg.use_cache:
                states_clean = frontend.forward_with_ids(batch_x, list(utt_ids))
            else:
                states_clean = frontend(batch_x)
            states_laund = frontend(batch_laund)

        h_agg_clean, logits_clean = aggregate_and_classify(backend, states_clean)
        h_agg_laund, logits_laund = aggregate_and_classify(backend, states_laund)

        loss_bce_laund = criterion(logits_laund, labels)
        loss_bce_clean = criterion(logits_clean, labels)
        loss_stab = stability_loss(h_agg_clean, h_agg_laund)
        loss_total = (
            loss_bce_laund
            + cfg.lambda_clean * loss_bce_clean
            + cfg.lambda_stab * loss_stab
        )

        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()

        B = labels.size(0)
        sum_bce_laund += loss_bce_laund.item() * B
        sum_bce_clean += loss_bce_clean.item() * B
        sum_stab += loss_stab.item() * B
        sum_total += loss_total.item() * B
        n_correct += (logits_laund.argmax(1) == labels).sum().item()
        n_total += B

    N = max(n_total, 1)
    return {
        "loss_bce_laund": sum_bce_laund / N,
        "loss_bce_clean": sum_bce_clean / N,
        "loss_stab": sum_stab / N,
        "loss_total": sum_total / N,
        "acc_laund": n_correct / N,
    }


@torch.no_grad()
def evaluate_dev(
    backend: nn.Module,
    frontend: SSLFrontend,
    loader: DataLoader,
    criterion: nn.Module,
    engine: LaunderingEngine | None,
    device: str,
    pipeline: str,
    depth: int,
    strength: str,
    lambda_clean: float,
    use_cache: bool,
) -> dict:
    """
    Single pass through the dev set. Returns loss, accuracy, EERs, score arrays.

    The laundering condition (pipeline, depth, strength) is fixed across all
    eval calls so the dev EER trajectory is directly comparable across epochs.

    Clean and laundered scores are collected in one pass to avoid two iterations
    of the dev set per eval interval.

    Score convention: logits[:, 0] is the bonafide class logit, consistent with
    ssl_eval_wrapper.py. EER is computed from label-separated score arrays
    (via the key field), so sign convention does not affect correctness.

    Dev loss omits L_stab intentionally — L_stab is a training regulariser.
    Including it in dev loss would mix regularisation strength (lambda_stab)
    into the checkpoint selection metric, making comparisons across lambda
    sweeps incomparable.

    Two data paths (mirrors train_epoch):
      Precomputed (engine is None): loader yields 5-tuple from PairedLaunderDataset.
      Online (engine is not None):  loader yields 4-tuple; laundering applied per batch.
    """
    backend.eval()

    use_precomputed = engine is None
    if not use_precomputed:
        launder_fn = engine.get_batch_fn(pipeline, depth, strength)

    sum_loss = n_correct = n_total = 0
    bona_clean_scores: list[float] = []
    spoof_clean_scores: list[float] = []
    bona_laund_scores: list[float] = []
    spoof_laund_scores: list[float] = []

    for batch in tqdm(loader, desc=f"dev ({pipeline} k={depth})", leave=False):
        if use_precomputed:
            batch_x, batch_laund_cpu, utt_ids, _srcs, keys = batch
            batch_x = batch_x.to(device, non_blocking=True)
            batch_laund = batch_laund_cpu.to(device, non_blocking=True)
        else:
            batch_x, utt_ids, _srcs, keys = batch
            batch_x = batch_x.to(device, non_blocking=True)
            batch_laund = launder_fn(batch_x.cpu()).to(device, non_blocking=True)

        labels = torch.tensor(
            [LABEL_MAP[k] for k in keys], dtype=torch.long, device=device
        )

        if use_cache:
            states_clean = frontend.forward_with_ids(batch_x, list(utt_ids))
        else:
            states_clean = frontend(batch_x)
        states_laund = frontend(batch_laund)

        _, logits_clean = aggregate_and_classify(backend, states_clean)
        _, logits_laund = aggregate_and_classify(backend, states_laund)

        loss = criterion(logits_laund, labels) + lambda_clean * criterion(
            logits_clean, labels
        )
        sum_loss += loss.item() * labels.size(0)
        n_correct += (logits_laund.argmax(1) == labels).sum().item()
        n_total += labels.size(0)

        scores_clean = logits_clean[:, 0].cpu().numpy()
        scores_laund = logits_laund[:, 0].cpu().numpy()
        for score_c, score_l, key in zip(scores_clean, scores_laund, keys):
            if key == "bonafide":
                bona_clean_scores.append(float(score_c))
                bona_laund_scores.append(float(score_l))
            else:
                spoof_clean_scores.append(float(score_c))
                spoof_laund_scores.append(float(score_l))

    N = max(n_total, 1)

    def _eer(bona: list[float], spoof: list[float]) -> float:
        if not bona or not spoof:
            return float("nan")
        eer, _ = compute_eer(
            np.array(bona, dtype=np.float32),
            np.array(spoof, dtype=np.float32),
        )
        return float(eer * 100)

    return {
        "dev_loss": sum_loss / N,
        "dev_acc_laund": n_correct / N,
        "dev_eer_clean": _eer(bona_clean_scores, spoof_clean_scores),
        "dev_eer_laund": _eer(bona_laund_scores, spoof_laund_scores),
        "scores": {
            "bona_clean": np.array(bona_clean_scores, dtype=np.float32),
            "spoof_clean": np.array(spoof_clean_scores, dtype=np.float32),
            "bona_laund": np.array(bona_laund_scores, dtype=np.float32),
            "spoof_laund": np.array(spoof_laund_scores, dtype=np.float32),
        },
    }


# Artifact helpers
def save_layer_weights(backend: nn.Module, epoch: int, lw_dir: Path) -> None:
    """Save SSL layer importance weights (softmax-normalised α_j) to JSON."""
    raw = getattr(backend, "_orig_mod", backend)
    if not hasattr(raw, "get_layer_weights"):
        return
    weights = raw.get_layer_weights()
    payload = {
        "epoch": epoch,
        "weights": {str(k): round(float(v), 8) for k, v in sorted(weights.items())},
        "top_layer": int(max(weights, key=weights.__getitem__)),
        "weight_sum": round(float(sum(weights.values())), 8),
    }
    with open(lw_dir / f"epoch_{epoch:04d}.json", "w") as f:
        json.dump(payload, f, indent=2)


def save_dev_scores(scores: dict, label: str, scores_dir: Path) -> None:
    """
    Save dev score arrays to a named compressed NPZ file.

    Called only at best_laund and best_clean checkpoints to produce two
    stable, named files rather than per-epoch accumulations. These are used
    for DET curves and SCI figures in the paper.

    Load with: data = np.load(path); bona = data["bona_clean"]
    """
    np.savez_compressed(
        scores_dir / f"{label}.npz",
        bona_clean=scores["bona_clean"],
        spoof_clean=scores["spoof_clean"],
        bona_laund=scores["bona_laund"],
        spoof_laund=scores["spoof_laund"],
    )


def append_training_log(log_path: Path, row: dict) -> None:
    """Append one epoch's metrics to the training CSV log."""
    fieldnames = [
        "epoch",
        "loss_total",
        "loss_bce_laund",
        "loss_bce_clean",
        "loss_stab",
        "acc_laund",
        "dev_loss",
        "dev_acc_laund",
        "dev_eer_clean",
        "dev_eer_laund",
        "best_dev_eer_laund",
        "best_dev_eer_clean",
        "patience_ctr",
        "elapsed_s",
        "timestamp_utc",
    ]
    write_header = not log_path.exists()
    with open(log_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def save_weights_only(backend: nn.Module, path: Path) -> None:
    """Save only model weights (no optimizer state) for use with eval_suite.py."""
    raw = getattr(backend, "_orig_mod", backend)
    torch.save(raw.state_dict(), path)


def save_full_checkpoint(
    backend: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    state: dict,
    path: Path,
) -> None:
    """Save full checkpoint (weights + optimizer + training state) for resume."""
    raw = getattr(backend, "_orig_mod", backend)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": raw.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "state": state,
        },
        path,
    )


def load_checkpoint(
    backend: nn.Module,
    optimizer: torch.optim.Optimizer,
    path: Path,
    device: str,
) -> tuple[int, dict]:
    """
    Load a full checkpoint for resume. Returns (resumed_epoch, saved_state).

    weights_only=False is required here because the optimizer state dict
    contains Python objects (e.g. torch.optim.Adam moment tensors packaged
    with non-tensor metadata) that torch.load cannot reconstruct in
    weights-only mode. This file is written by save_full_checkpoint above
    and is never sourced from an untrusted location.
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)  # noqa: S614
    raw = getattr(backend, "_orig_mod", backend)
    raw.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt["epoch"], ckpt.get("state", {})


# Setup helpers — extracted from main() to reduce cognitive complexity
def _build_frontends(
    model: str,
    extract_layers: list[int],
    device: str,
    cache_dir: str,
    use_cache: bool,
) -> tuple[SSLFrontend, SSLFrontend]:
    """Construct frozen SSL frontends for train and dev splits."""
    train_cache = EmbeddingCache(cache_dir, model, "train") if use_cache else None
    dev_cache = EmbeddingCache(cache_dir, model, "dev") if use_cache else None
    frontend_train = SSLFrontend(
        model_type=model,
        extract_layers=extract_layers,
        device=device,
        cache=train_cache,
    )
    frontend_dev = SSLFrontend(
        model_type=model,
        extract_layers=extract_layers,
        device=device,
        cache=dev_cache,
    )
    for fe in (frontend_train, frontend_dev):
        fe.eval()
        for param in fe.parameters():
            param.requires_grad_(False)
    return frontend_train, frontend_dev


def _build_backend(
    backend_name: str,
    num_layers: int,
    embed_dim: int,
    device: str,
) -> nn.Module:
    """Construct the trainable backend module."""
    if backend_name == "ffn":
        return WeightedAggregationBackend(
            num_layers=num_layers, embed_dim=embed_dim
        ).to(device)
    if backend_name == "aasist":
        return SSLWithAASIST(num_layers=num_layers, embed_dim=embed_dim).to(device)
    if backend_name == "rawnet2":
        return SSLWithRawNet2(num_layers=num_layers, embed_dim=embed_dim).to(device)
    raise ValueError(f"Unknown backend: {backend_name}")


def _load_init_weights(
    backend: nn.Module,
    init_weights: str | None,
    device: str,
    log: logging.Logger,
) -> None:
    """Optionally warm-start backend from a clean-trained checkpoint."""
    if not init_weights:
        return
    iw_path = Path(init_weights)
    if not iw_path.exists():
        log.warning(f"[init_weights] Not found: {iw_path} — training from scratch")
        return
    state = torch.load(iw_path, map_location=device, weights_only=True)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    prefix = "_orig_mod."
    state = {
        (k[len(prefix) :] if k.startswith(prefix) else k): v for k, v in state.items()
    }
    missing, unexpected = backend.load_state_dict(state, strict=False)
    if missing:
        log.warning(f"[init_weights] Missing keys: {missing}")
    if unexpected:
        log.warning(f"[init_weights] Unexpected keys: {unexpected}")
    log.info(f"[init_weights] Warm-started from: {iw_path}")


def _build_dataloaders(
    data_root: Path,
    max_train: int | None,
    max_dev: int | None,
    batch_size: int,
    device: str,
    log: logging.Logger,
) -> tuple[DataLoader, DataLoader]:
    """Construct clean train and dev DataLoaders (online laundering path)."""
    train_ds = WavDataset(data_root, "LA", split="train", max_len=64000)
    dev_ds = WavDataset(data_root, "LA", split="dev", max_len=64000)
    if max_train:
        train_ds.trials = train_ds.trials[:max_train]
    if max_dev:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(dev_ds.trials), size=max_dev, replace=False)
        dev_ds.trials = [dev_ds.trials[i] for i in sorted(idx)]
    log.info(f"Dataset  train={len(train_ds)}  dev={len(dev_ds)}")

    nw = _num_workers()
    loader_kw = {
        "batch_size": batch_size,
        "num_workers": nw,
        "pin_memory": (device == "cuda"),
        "persistent_workers": (nw > 0),
        "prefetch_factor": (4 if nw > 0 else None),
    }
    log.info(f"DataLoader num_workers={nw}")
    return (
        DataLoader(train_ds, shuffle=True, **loader_kw),
        DataLoader(dev_ds, shuffle=False, **loader_kw),
    )


def _build_paired_dataloaders(
    data_root: Path,
    precomputed_root: Path,
    pipelines: list[str],
    depths: list[int],
    eval_pipeline: str,
    eval_depth: int,
    max_train: int | None,
    max_dev: int | None,
    batch_size: int,
    device: str,
    seed: int,
    log: logging.Logger,
) -> tuple[DataLoader, DataLoader]:
    """Construct paired (clean + laundered) train and dev DataLoaders.

    Each DataLoader wraps a PairedLaunderDataset so that __getitem__ always
    returns the clean and laundered waveform for the *same* utterance index.
    This guarantees alignment under shuffle=True without any external
    synchronisation between loaders.

    Alignment is verified inside PairedLaunderDataset.__init__; an
    AssertionError is raised immediately if utterance IDs diverge.

    Train DataLoader: shuffle=True, samples all (pipeline, depth) conditions.
    Dev DataLoader:   shuffle=False, single fixed (eval_pipeline, eval_depth).
    """
    log.info("Building PairedLaunderDataset (train) …")
    train_ds = PairedLaunderDataset(
        data_root=data_root,
        track="LA",
        precomputed_root=precomputed_root,
        pipelines=pipelines,
        depths=depths,
        split="train",
        max_len=64000,
        seed=seed,
    )
    if max_train:
        train_ds.trials = train_ds.trials[:max_train]
        train_ds._clean_ds.trials = train_ds.trials
        train_ds._assigned = train_ds._assigned[:max_train]
        for ds in train_ds._precomp_ds.values():
            ds.trials = ds.trials[:max_train]
    log.info(
        f"  train: {len(train_ds)} utts × {len(train_ds._conditions)} conditions "
        f"({', '.join(f'{p}/d{d}' for p,d in train_ds._conditions)})"
    )

    log.info("Building PairedLaunderDataset (dev) …")
    dev_ds = PairedLaunderDataset(
        data_root=data_root,
        track="LA",
        precomputed_root=precomputed_root,
        pipelines=[eval_pipeline],
        depths=[eval_depth],
        split="dev",
        max_len=64000,
        seed=seed,
    )
    if max_dev:
        rng = np.random.default_rng(42)
        idx = sorted(rng.choice(len(dev_ds.trials), size=max_dev, replace=False))
        dev_ds.trials = [dev_ds.trials[i] for i in idx]
        dev_ds._clean_ds.trials = dev_ds.trials
        dev_ds._assigned = [dev_ds._assigned[i] for i in idx]
        for ds in dev_ds._precomp_ds.values():
            ds.trials = [ds.trials[i] for i in idx]
    log.info(f"  dev:   {len(dev_ds)} utts  condition: ({eval_pipeline}, d={eval_depth})")

    nw = _num_workers()
    loader_kw = {
        "batch_size": batch_size,
        "num_workers": nw,
        "pin_memory": (device == "cuda"),
        "persistent_workers": (nw > 0),
        "prefetch_factor": (4 if nw > 0 else None),
    }
    log.info(f"DataLoader num_workers={nw}")
    return (
        DataLoader(train_ds, shuffle=True, **loader_kw),
        DataLoader(dev_ds, shuffle=False, **loader_kw),
    )


def _handle_epoch_checkpoint(
    epoch: int,
    do_eval: bool,
    dev_eer_laund: float,
    dev_eer_clean: float,
    dev_scores: dict | None,
    best_dev_eer_laund: float,
    best_dev_eer_clean: float,
    patience_ctr: int,
    backend: nn.Module,
    optimizer: torch.optim.Optimizer,
    batch_rng: np.random.Generator,
    paths: dict,
    ckpt_every: int,
    lw_every: int,
    log: logging.Logger,
) -> tuple[float, float, int, bool]:
    """
    Handle checkpoint saving and patience tracking for one epoch.

    Returns updated (best_dev_eer_laund, best_dev_eer_clean, patience_ctr,
    improved_laund). improved_laund is used by the caller to trigger an
    extra layer-weight snapshot.
    """
    improved_laund = False

    if do_eval and not np.isnan(dev_eer_laund):
        if dev_eer_laund < best_dev_eer_laund:
            best_dev_eer_laund = dev_eer_laund
            improved_laund = True
            patience_ctr = 0
            save_weights_only(backend, paths["best_laund"])
            save_dev_scores(dev_scores, "best_laund", paths["scores_dir"])
            log.info(f"  → best_laund.pth  dev_eer_laund={best_dev_eer_laund:.4f}%")
        else:
            patience_ctr += 1

    if do_eval and not np.isnan(dev_eer_clean):
        if dev_eer_clean < best_dev_eer_clean:
            best_dev_eer_clean = dev_eer_clean
            save_weights_only(backend, paths["best_clean"])
            save_dev_scores(dev_scores, "best_clean", paths["scores_dir"])

    if epoch % ckpt_every == 0:
        save_full_checkpoint(
            backend,
            optimizer,
            epoch,
            {
                "best_dev_eer_laund": best_dev_eer_laund,
                "best_dev_eer_clean": best_dev_eer_clean,
                "patience_ctr": patience_ctr,
                "batch_rng_state": batch_rng.bit_generator.state,
            },
            paths["ckpt_dir"] / f"epoch_{epoch:04d}.pth",
        )

    if epoch % lw_every == 0 or improved_laund:
        save_layer_weights(backend, epoch, paths["lw_dir"])

    return best_dev_eer_laund, best_dev_eer_clean, patience_ctr, improved_laund


def main() -> None:
    args = parse_args()

    # Resolve eval_depth before dry_run may modify args.depths.
    eval_depth = args.eval_depth if args.eval_depth is not None else max(args.depths)
    _validate_args(args, eval_depth)

    # Reproducibility.
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    batch_rng = np.random.default_rng(seed=args.seed)

    if args.dry_run:
        args.max_train = min(args.max_train or 200, 200)
        args.max_dev = min(args.max_dev or 200, 200)
        args.epochs = min(args.epochs, 2)
        args.ckpt_every = 1
        args.lw_every = 1
        args.eval_every = 1

    # ── Run directory ─────────────────────────────────────────────────────
    run_id = args.run_id or (
        f"lam{args.lambda_stab}"
        f"_lc{args.lambda_clean}"
        f"_pip{''.join(sorted(args.pipelines))}"
        f"_d{eval_depth}"
        f"_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    )
    run_dir = Path(args.output_dir) / f"{args.model}_{args.backend}" / run_id
    ckpt_dir = run_dir / "checkpoints"
    lw_dir = run_dir / "layer_weights"
    scores_dir = run_dir / "dev_scores"
    for d in (run_dir, ckpt_dir, lw_dir, scores_dir):
        d.mkdir(parents=True, exist_ok=True)

    log = setup_logging(run_dir)
    log.info(f"Run directory: {run_dir}")

    # Convenience dict of paths passed into _handle_epoch_checkpoint.
    paths = {
        "best_laund": run_dir / "best_laund.pth",
        "best_clean": run_dir / "best_clean.pth",
        "final": run_dir / "final.pth",
        "ckpt_dir": ckpt_dir,
        "lw_dir": lw_dir,
        "scores_dir": scores_dir,
    }

    # ── Config ────────────────────────────────────────────────────────────
    registry_key = _REGISTRY_KEY[(args.model, args.backend)]
    config = {
        "run_id": run_id,
        "model": args.model,
        "backend": args.backend,
        "registry_key": registry_key,
        "init_weights": args.init_weights,
        "pipelines": args.pipelines,
        "depths": args.depths,
        "strength": args.strength,
        "eval_depth": eval_depth,
        "eval_pipeline": args.pipelines[0],
        "eval_depth_in_train_depths": eval_depth in args.depths,
        "lambda_stab": args.lambda_stab,
        "lambda_clean": args.lambda_clean,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "epochs": args.epochs,
        "patience": args.patience,
        "eval_every": args.eval_every,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "dry_run": args.dry_run,
        "data_root": str(args.data_root),
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "loss_formulation": {
            "L_total": "L_bce_laund + lambda_clean*L_bce_clean + lambda_stab*L_stab",
            "L_stab": "MSE(mean_pool(h_agg_clean), mean_pool(h_agg_laund))",
            "h_agg": "weighted_sum(frozen_ssl_layers, learnable_alpha_j)",
            "gradient_path": "L_stab → h_agg → alpha_j (SSL encoder frozen throughout)",
            "dev_loss_excludes_L_stab": True,
        },
        "early_stopping": {
            "metric": "dev_eer_laund",
            "patience_unit": "eval_events",
            "effective_max_epoch_patience": args.patience * args.eval_every,
        },
        "noise_dir_note": (
            "Additive noise in pipeline stages N3/P3 uses whatever noise_dir "
            "is set in configs/N_params.yaml and configs/P_params.yaml. "
            "null (default) → white-noise fallback, reproducible, no external "
            "files required. Set noise_dir to SPIB/QUT path in those YAMLs "
            "before training if real environmental noise is desired."
        ),
        "eval_command": (
            f"python eval_suite.py --model {registry_key} "
            f"--weights {run_dir}/best_laund.pth"
        ),
    }
    config_path = run_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    log.info(f"Config saved → {config_path}")

    # ── Device ────────────────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    log.info(f"Device: {device}")

    # ── SSL frontend ──────────────────────────────────────────────────────
    import yaml

    with open(f"configs/{args.model}_probe.yaml") as f:
        ssl_config = yaml.safe_load(f)
    extract_layers = ssl_config.get("extract_layers", list(range(args.num_layers)))

    use_cache = not args.no_cache
    frontend_train, frontend_dev = _build_frontends(
        args.model, extract_layers, device, args.cache_dir, use_cache
    )
    log.info(f"SSL frontend: {args.model}  layers={extract_layers}  cache={use_cache}")

    # ── Backend ───────────────────────────────────────────────────────────
    backend = _build_backend(args.backend, args.num_layers, args.embed_dim, device)
    _load_init_weights(backend, args.init_weights, device, log)

    # torch.compile must be done before optimizer construction.
    if device == "cuda":
        backend = torch.compile(backend, mode="reduce-overhead")

    log.info(
        f"Backend trainable params: {sum(p.numel() for p in backend.parameters() if p.requires_grad):,}"
    )

    # ── Laundering engine + DataLoaders ───────────────────────────────────
    # Only instantiate LaunderingEngine when online laundering is needed.
    if args.precomputed_root is None:
        engine = LaunderingEngine(args.config_dir)
        log.info("Laundering: online (LaunderingEngine active)")
        train_loader, dev_loader = _build_dataloaders(
            Path(args.data_root),
            args.max_train,
            args.max_dev,
            args.batch_size,
            device,
            log,
        )
    else:
        engine = None
        log.info(f"Laundering: precomputed  root={args.precomputed_root}")
        train_loader, dev_loader = _build_paired_dataloaders(
            data_root=Path(args.data_root),
            precomputed_root=Path(args.precomputed_root),
            pipelines=args.pipelines,
            depths=args.depths,
            eval_pipeline=args.pipelines[0],
            eval_depth=eval_depth,
            max_train=args.max_train,
            max_dev=args.max_dev,
            batch_size=args.batch_size,
            device=device,
            seed=args.seed,
            log=log,
        )

    # ── Optimiser ─────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        backend.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    criterion = nn.CrossEntropyLoss()

    # Bundle per-run training constants into a single config object.
    train_cfg = TrainConfig(
        engine=engine,
        pipelines=args.pipelines,
        depths=args.depths,
        strength=args.strength,
        lambda_stab=args.lambda_stab,
        lambda_clean=args.lambda_clean,
        rng=batch_rng,
        use_cache=use_cache,
    )

    # ── Resume ────────────────────────────────────────────────────────────
    start_epoch = 1
    best_dev_eer_laund = float("inf")
    best_dev_eer_clean = float("inf")
    patience_ctr = 0

    if args.resume:
        ckpts = sorted(ckpt_dir.glob("epoch_*.pth"))
        if ckpts:
            resumed_epoch, saved_state = load_checkpoint(
                backend, optimizer, ckpts[-1], device
            )
            start_epoch = resumed_epoch + 1
            best_dev_eer_laund = saved_state.get("best_dev_eer_laund", float("inf"))
            best_dev_eer_clean = saved_state.get("best_dev_eer_clean", float("inf"))
            patience_ctr = saved_state.get("patience_ctr", 0)
            if "batch_rng_state" in saved_state:
                batch_rng = np.random.default_rng(args.seed)
                batch_rng.bit_generator.state = saved_state["batch_rng_state"]
                train_cfg.rng = batch_rng
            log.info(
                f"[RESUME] From epoch {start_epoch}  "
                f"best_eer_laund={best_dev_eer_laund:.4f}%  "
                f"patience={patience_ctr}/{args.patience}"
            )
        else:
            log.info("[RESUME] No checkpoints found — starting fresh.")

    save_layer_weights(backend, 0, lw_dir)

    # ── Training loop ─────────────────────────────────────────────────────
    log.info("=" * 65)
    log.info(
        f"  Model: {args.model}+{args.backend}  λ_stab={args.lambda_stab}  λ_clean={args.lambda_clean}"
    )
    log.info(
        f"  Train: pipelines={args.pipelines}  depths={args.depths}  strength={args.strength}"
    )
    log.info(
        f"  Dev EER: pipeline={args.pipelines[0]}  depth={eval_depth}  (every {args.eval_every} epochs)"
    )
    if eval_depth not in args.depths:
        log.info(
            f"  WARNING: eval_depth={eval_depth} not in training depths — checkpoint on unseen condition"
        )
    log.info(
        f"  Early stop: patience={args.patience} eval events  (≤{args.patience * args.eval_every} epochs)"
    )
    log.info("=" * 65)

    last_epoch = start_epoch
    log_path = run_dir / "training_log.csv"

    for epoch in range(start_epoch, args.epochs + 1):
        last_epoch = epoch
        t0 = time.time()

        # Rotate per-utterance laundering conditions before each epoch so that
        # every utterance sees a different (pipeline, depth) each pass, matching
        # the augmentation diversity of the original per-batch online sampling.
        # No-op in online mode (WavDataset has no set_epoch).
        if isinstance(train_loader.dataset, PairedLaunderDataset):
            train_loader.dataset.set_epoch(epoch)

        train_metrics = train_epoch(
            backend,
            frontend_train,
            train_loader,
            optimizer,
            criterion,
            train_cfg,
            device,
        )

        # Dev evaluation (every eval_every epochs and always on last epoch).
        dev_eer_laund = dev_eer_clean = dev_loss = dev_acc = float("nan")
        dev_scores = None
        do_eval = (epoch % args.eval_every == 0) or (epoch == args.epochs)

        if do_eval:
            dev_result = evaluate_dev(
                backend,
                frontend_dev,
                dev_loader,
                criterion,
                engine,
                device,
                pipeline=args.pipelines[0],
                depth=eval_depth,
                strength=args.strength,
                lambda_clean=args.lambda_clean,
                use_cache=use_cache,
            )
            dev_eer_laund = dev_result["dev_eer_laund"]
            dev_eer_clean = dev_result["dev_eer_clean"]
            dev_loss = dev_result["dev_loss"]
            dev_acc = dev_result["dev_acc_laund"]
            dev_scores = dev_result["scores"]

        elapsed = time.time() - t0

        best_dev_eer_laund, best_dev_eer_clean, patience_ctr, improved_laund = (
            _handle_epoch_checkpoint(
                epoch,
                do_eval,
                dev_eer_laund,
                dev_eer_clean,
                dev_scores,
                best_dev_eer_laund,
                best_dev_eer_clean,
                patience_ctr,
                backend,
                optimizer,
                batch_rng,
                paths,
                args.ckpt_every,
                args.lw_every,
                log,
            )
        )

        eer_l_str = f"{dev_eer_laund:.3f}%" if not np.isnan(dev_eer_laund) else "  -   "
        eer_c_str = f"{dev_eer_clean:.3f}%" if not np.isnan(dev_eer_clean) else "  -   "
        log.info(
            f"Epoch {epoch:3d}/{args.epochs}"
            f"  L={train_metrics['loss_total']:.4f}"
            f"  bce_l={train_metrics['loss_bce_laund']:.4f}"
            f"  bce_c={train_metrics['loss_bce_clean']:.4f}"
            f"  stab={train_metrics['loss_stab']:.6f}"
            f"  acc={train_metrics['acc_laund']:.4f}"
            f"  | eer_laund={eer_l_str}"
            f"  eer_clean={eer_c_str}"
            f"  pat={patience_ctr}/{args.patience}"
            f"  [{elapsed:.0f}s]"
        )
        if use_cache:
            ts = frontend_train.cache.stats()
            ds = frontend_dev.cache.stats()
            log.info(
                f"  [cache] train h={ts['hits']} m={ts['misses']}  dev h={ds['hits']} m={ds['misses']}"
            )

        append_training_log(
            log_path,
            {
                "epoch": epoch,
                "loss_total": round(train_metrics["loss_total"], 6),
                "loss_bce_laund": round(train_metrics["loss_bce_laund"], 6),
                "loss_bce_clean": round(train_metrics["loss_bce_clean"], 6),
                "loss_stab": round(train_metrics["loss_stab"], 8),
                "acc_laund": round(train_metrics["acc_laund"], 6),
                "dev_loss": round(dev_loss, 6) if not np.isnan(dev_loss) else "",
                "dev_acc_laund": round(dev_acc, 6) if not np.isnan(dev_acc) else "",
                "dev_eer_clean": round(dev_eer_clean, 4)
                if not np.isnan(dev_eer_clean)
                else "",
                "dev_eer_laund": round(dev_eer_laund, 4)
                if not np.isnan(dev_eer_laund)
                else "",
                "best_dev_eer_laund": round(best_dev_eer_laund, 4)
                if best_dev_eer_laund < float("inf")
                else "",
                "best_dev_eer_clean": round(best_dev_eer_clean, 4)
                if best_dev_eer_clean < float("inf")
                else "",
                "patience_ctr": patience_ctr,
                "elapsed_s": round(elapsed, 1),
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            },
        )

        if patience_ctr >= args.patience:
            log.info(
                f"[EARLY STOP] {patience_ctr} consecutive eval events without "
                f"improvement on dev_eer_laund. Stopping at epoch {epoch}."
            )
            break

    # ── End-of-training artifacts ─────────────────────────────────────────
    save_weights_only(backend, paths["final"])
    save_layer_weights(backend, last_epoch, lw_dir)

    config.update(
        {
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "epochs_trained": last_epoch,
            "best_dev_eer_laund": best_dev_eer_laund,
            "best_dev_eer_clean": best_dev_eer_clean,
            "best_laund_weights": str(paths["best_laund"]),
            "best_clean_weights": str(paths["best_clean"]),
            "final_weights": str(paths["final"]),
        }
    )
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    log.info("=" * 65)
    log.info("Training complete.")
    log.info(
        f"  Best laundered EER : {best_dev_eer_laund:.4f}%  → {paths['best_laund']}"
    )
    log.info(
        f"  Best clean EER     : {best_dev_eer_clean:.4f}%  → {paths['best_clean']}"
    )
    log.info("")
    log.info("Next step — evaluate on full laundering grid:")
    log.info(
        f"  python eval_suite.py --model {registry_key} --weights {paths['best_laund']}"
    )
    log.info("=" * 65)


if __name__ == "__main__":
    main()

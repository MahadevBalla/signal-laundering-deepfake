"""warm_ssl_cache.py

Pre-warm the SSL embedding cache for all precomputed laundered conditions so
that eval_suite.py (and eval_stab.sbatch) can run without any wav2vec2
transformer forward passes.

This script mirrors the _prewarm_cache() logic in layer_sweep.py, extended
to cover laundered (precomputed) audio across every (pipeline, depth,
strength) combination in the evaluation grid.

How it works
------------
For each (pipeline, depth, strength) variant:
  1. Instantiate a WavDataset pointing at
       <precomputed_root>/<pipeline>_d<depth>_<strength>/
  2. Run the frozen SSL frontend once over the eval split.
  3. Write each utterance's {layer_idx: Tensor[T, D]} output to
       <cache_dir>/<model_type>/eval_{pipeline}_d{depth}_{strength}/<utt_id>.pt

The cache key format (model_type + split string) is identical to the one used
by EmbeddingCache inside SSLEvalWrapper.evaluate() when precomputed_root is
active, so subsequent eval_suite.py calls will get 100% cache hits.

Cache key alignment
-------------------
The SSL frontend is initialised with extract_layers=[0..11] (all 12 layers),
matching the weighted backend configuration in configs/wav2vec2_probe.yaml.
If the backend uses a subset of layers, the cache will still be a hit because
EmbeddingCache checks that the requested layers are a *subset* of the cached
layers, not an exact match.

Usage
-----
# All 27 variants (3 pipelines × 3 strengths × 3 depths):
python warm_ssl_cache.py --model wav2vec2 --precomputed_root data/precomputed

# Specific subset:
python warm_ssl_cache.py --model wav2vec2 --precomputed_root data/precomputed \\
    --pipelines N --strengths M H --depths 1 2

# Dry run (first 200 utterances per variant):
python warm_ssl_cache.py --model wav2vec2 --precomputed_root data/precomputed \\
    --dry_run
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.ssl_frontend import SSLFrontend
from src.models.embedding_cache import EmbeddingCache
from src.models.dataset import WavDataset

log = logging.getLogger("warm_ssl_cache")

ALL_PIPELINES = ["N", "M", "P"]
ALL_STRENGTHS = ["L", "M", "H"]
ALL_DEPTHS    = [1, 2, 3]


def _setup_logging() -> None:
    fmt = logging.Formatter("%(asctime)s %(levelname)-8s %(message)s", datefmt="%H:%M:%S")
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    if not root.handlers:
        root.addHandler(ch)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the SSL cache warmer."""
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--model",
        required=True,
        choices=["wav2vec2", "hubert", "wavlm"],
        help="SSL model type to warm the cache for.",
    )
    p.add_argument(
        "--precomputed_root",
        required=True,
        help="Root directory produced by precompute_laundering.py "
             "(e.g. data/precomputed).",
    )
    p.add_argument(
        "--cache_dir",
        default="data/ssl_cache",
        help="Base directory for the SSL embedding cache "
             "(default: data/ssl_cache).",
    )
    p.add_argument(
        "--pipelines",
        nargs="+",
        default=ALL_PIPELINES,
        choices=ALL_PIPELINES,
        help="Laundering pipelines to warm (default: N M P).",
    )
    p.add_argument(
        "--strengths",
        nargs="+",
        default=ALL_STRENGTHS,
        choices=ALL_STRENGTHS,
        help="Laundering strengths to warm (default: L M H).",
    )
    p.add_argument(
        "--depths",
        nargs="+",
        type=int,
        default=ALL_DEPTHS,
        help="Laundering depths to warm (default: 1 2 3).",
    )
    p.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for SSL frontend forward passes (default: 8).",
    )
    p.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="DataLoader worker count (default: 8).",
    )
    p.add_argument(
        "--max_eval",
        type=int,
        default=None,
        help="Cap the number of utterances per variant (None = full eval set).",
    )
    p.add_argument(
        "--dry_run",
        action="store_true",
        help="Process only the first 200 utterances per variant for a quick test.",
    )
    return p.parse_args(argv)


def warm_variant(
    *,
    model_type: str,
    variant_root: Path,
    split_key: str,
    cache_dir: str,
    batch_size: int,
    num_workers: int,
    max_eval: int | None,
    device: str,
    frontend: SSLFrontend,
) -> dict[str, int]:
    """Run one forward pass over a precomputed variant and populate the cache.

    Args:
        model_type: SSL model identifier (e.g. ``"wav2vec2"``).
        variant_root: Directory of the precomputed laundered dataset
            (e.g. ``data/precomputed/N_d1_M``).
        split_key: Cache subdirectory key (e.g. ``"eval_N_d1_M"``).
        cache_dir: Base cache directory.
        batch_size: Batch size for DataLoader.
        num_workers: Worker count for DataLoader.
        max_eval: Utterance cap (None = full eval set).
        device: Torch device string.
        frontend: Pre-built SSLFrontend instance (re-used across variants to
            avoid reloading the 360 MB wav2vec2 checkpoint each time).

    Returns:
        Cache stats dict ``{"hits": int, "misses": int}``.
    """
    cache = EmbeddingCache(cache_dir, model_type, split_key)
    # Swap the frontend's active cache to the new condition-specific store.
    frontend.cache = cache

    dataset = WavDataset(variant_root, "LA", split="eval", max_len=64000)
    if max_eval is not None:
        dataset.trials = dataset.trials[:max_eval]

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
        persistent_workers=(num_workers > 0),
        prefetch_factor=4 if num_workers > 0 else None,
    )

    log.info(
        "[WARM] %s | split=%s | utterances=%d",
        model_type, split_key, len(dataset),
    )
    t0 = time.monotonic()
    with torch.no_grad():
        for batch_x, ids, _srcs, _keys in tqdm(
            loader, desc=split_key, unit="batch", dynamic_ncols=True
        ):
            batch_x = batch_x.to(device, non_blocking=True)
            frontend.forward_with_ids(batch_x, list(ids))

    elapsed = time.monotonic() - t0
    stats = cache.stats()
    log.info(
        "[WARM] %s | split=%s | hits=%d misses=%d | %.1fs",
        model_type, split_key, stats["hits"], stats["misses"], elapsed,
    )
    return stats


def main(argv: list[str] | None = None) -> None:
    """Warm the SSL embedding cache for all requested laundered variants."""
    _setup_logging()
    args = parse_args(argv)

    precomputed_root = Path(args.precomputed_root).resolve()
    if not precomputed_root.is_dir():
        log.error("precomputed_root does not exist: %s", precomputed_root)
        sys.exit(1)

    bad_depths = [d for d in args.depths if not (1 <= d <= 3)]
    if bad_depths:
        log.error("--depths must be in [1, 3]. Invalid: %s", bad_depths)
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Device: %s", device)

    # All SSL backends use all 12 layers (see configs/wav2vec2_probe.yaml).
    # Use list(range(12)) to guarantee the cache covers any requested subset.
    all_layers = list(range(12))

    log.info(
        "Initialising %s frontend (extract_layers=%s) ...", args.model, all_layers
    )
    frontend = SSLFrontend(
        model_type=args.model,
        extract_layers=all_layers,
        device=device,
        cache=None,  # will be swapped per-variant in warm_variant()
    )
    frontend.eval()
    for p in frontend.parameters():
        p.requires_grad_(False)

    max_eval = 200 if args.dry_run else args.max_eval

    variants = [
        (pipeline, depth, strength)
        for pipeline in args.pipelines
        for strength in args.strengths
        for depth in args.depths
    ]
    log.info(
        "Warming %d variant(s) × ~%s utterances each ...",
        len(variants),
        max_eval if max_eval else "full-eval",
    )

    total_hits = total_misses = 0
    t_start = time.monotonic()

    for pipeline, depth, strength in variants:
        variant_root = precomputed_root / f"{pipeline}_d{depth}_{strength}"
        if not variant_root.is_dir():
            log.warning(
                "[SKIP] Variant directory not found: %s — run precompute_laundering.py first.",
                variant_root,
            )
            continue

        split_key = f"eval_{pipeline}_d{depth}_{strength}"
        stats = warm_variant(
            model_type=args.model,
            variant_root=variant_root,
            split_key=split_key,
            cache_dir=args.cache_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            max_eval=max_eval,
            device=device,
            frontend=frontend,
        )
        total_hits   += stats["hits"]
        total_misses += stats["misses"]

    elapsed = time.monotonic() - t_start
    log.info("=" * 56)
    log.info(
        "Done in %.1fs | total hits=%d misses=%d",
        elapsed, total_hits, total_misses,
    )
    if total_hits > 0:
        log.info(
            "%.1f%% of utterances were already cached (incremental warm).",
            100.0 * total_hits / (total_hits + total_misses),
        )
    log.info(
        "Cache written to: %s/%s/eval_<pipeline>_d<depth>_<strength>/",
        args.cache_dir, args.model,
    )
    log.info(
        "eval_stab.sbatch will now get 100%% SSL cache hits for laundered conditions."
    )


if __name__ == "__main__":
    main()

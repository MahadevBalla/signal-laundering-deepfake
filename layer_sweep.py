"""
layer_sweep.py
==============
Trains an FFN backend for each transformer layer (0-11) of a frozen SSL model
and evaluates clean EER per layer. Replicates the layer-wise analysis in:
  El Kheir et al., NAACL 2025 Findings.

The paper's key finding: lower layers (roughly 1-6 for base models) are more
discriminative than upper layers. This sweep tells you which layer to use for
laundering experiments before committing to full eval.

SSL frontend caching:
  Each of the 12 per-layer training runs would otherwise recompute the same
  frozen SSL forward pass independently. To avoid that, this script first
  pre-warms the embedding cache (data/ssl_cache/<model>/{train,dev}) with
  ALL 12 transformer layers in one pass over the dataset, then every
  per-layer `train_ssl_backend.py --mode single --layer L` subprocess hits
  the cache instead of recomputing the SSL frontend. Subsequent full sweeps
  (or re-runs after --dry_run) reuse the same cache.

  Disable with --no_cache (every subprocess recomputes from scratch).

Outputs:
  models/<model>_ffn_layer{l}.pth   (one per layer)
  outputs/sweep/<model>_layer_sweep.json
  outputs/sweep/<model>_layer_eer.png

Usage:
  python layer_sweep.py --model wav2vec2 --dry_run
  python layer_sweep.py --model wav2vec2
  python layer_sweep.py --model hubert --layers 0 1 2 3 4 5
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.ssl_frontend import SSLFrontend
from src.models.embedding_cache import EmbeddingCache
from src.models.dataset import WavDataset

def parse_args() -> argparse.Namespace:
    """Parse arguments for single-model layer sweep."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model",     required=True, choices=["wav2vec2", "hubert", "wavlm"])
    p.add_argument("--data_root", default="data/ASVspoof2019/LA")
    p.add_argument("--layers",    nargs="+", type=int, default=list(range(12)))
    p.add_argument("--epochs",     type=int,   default=20)
    p.add_argument("--lr",         type=float, default=1e-4)
    p.add_argument("--batch_size", type=int,   default=32)
    p.add_argument("--max_train",  type=int,   default=None)
    p.add_argument("--max_eval",   type=int,   default=None,
                   help="Cap eval utterances per layer (None=full eval set).")
    p.add_argument("--dry_run",   action="store_true",
                   help="200 utterances, 2 epochs. Fast end-to-end test.")
    p.add_argument("--skip_trained", action="store_true", default=True,
                   help="Skip training if weights already exist (default: True).")
    p.add_argument("--no_skip_trained", dest="skip_trained", action="store_false")
    p.add_argument("--cache_dir",  default="data/ssl_cache",
                   help="Directory for cached frozen-SSL sequence embeddings.")
    p.add_argument("--no_cache",   action="store_true",
                   help="Disable embedding caching (passed through to train_ssl_backend.py, "
                        "and skips the pre-warm pass below).")
    return p.parse_args()


def _prewarm_cache(args: argparse.Namespace) -> None:
    """Run one forward pass over train+dev to populate the SSL embedding cache
    with all 12 transformer layers, so every per-layer subprocess hits the cache.
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"
    all_layers = list(range(12))
    max_train = min(args.max_train or 200, 200) if args.dry_run else args.max_train
    max_dev = 200 if args.dry_run else None

    for split, max_n in (("train", max_train), ("dev", max_dev)):
        cache = EmbeddingCache(args.cache_dir, args.model, split)
        frontend = SSLFrontend(model_type=args.model, extract_layers=all_layers, device=device, cache=cache)
        frontend.eval()
        for p in frontend.parameters():
            p.requires_grad_(False)

        ds = WavDataset(Path(args.data_root), "LA", split=split, max_len=64000)
        if max_n:
            ds.trials = ds.trials[:max_n]
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=8)

        print(f"[CACHE PREWARM] {args.model} {split}: {len(ds)} utterances, layers={all_layers}")
        with torch.no_grad():
            for batch_x, ids, _srcs, _keys in tqdm(loader, desc=f"prewarm-{split}"):
                batch_x = batch_x.to(device, non_blocking=True)
                frontend.forward_with_ids(batch_x, list(ids))
        stats = cache.stats()
        print(f"[CACHE PREWARM] {args.model} {split}: hits={stats['hits']} misses={stats['misses']}")
        frontend.remove_hooks()
        del frontend


def _train_layer(args: argparse.Namespace, layer: int) -> None:
    """Train one single-layer FFN backend checkpoint."""
    cmd = [
        sys.executable, "train_ssl_backend.py",
        "--model", args.model, "--mode", "single", "--layer", str(layer),
        "--data_root", args.data_root,
        "--epochs", str(args.epochs),
        "--lr", str(args.lr),
        "--batch_size", str(args.batch_size),
        "--cache_dir", args.cache_dir,
    ]
    if args.max_train:
        cmd += ["--max_train", str(args.max_train)]
    if args.dry_run:
        cmd += ["--dry_run"]
    if args.no_cache:
        cmd += ["--no_cache"]
    subprocess.run(cmd, check=True)


def _eval_layer(args: argparse.Namespace, layer: int) -> float:
    """Evaluate one trained layer checkpoint and return its clean EER."""
    from src.models.ssl_eval_wrapper import SSLEvalWrapper
    wrapper = SSLEvalWrapper(
        config_path=f"configs/{args.model}_probe.yaml",
        data_root=args.data_root,
        backend_mode="single",
        layer=layer,
    )
    wrapper.load_weights(f"models/{args.model}_ffn_layer{layer}.pth")
    max_eval = 200 if args.dry_run else args.max_eval
    eer, _   = wrapper.evaluate(
        output_dir=f"outputs/sweep/{args.model}/layer{layer}",
        launder_fn=None,
        max_eval=max_eval,
    )
    return eer


def _plot(results: dict[int, float], model: str, out_dir: Path) -> None:
    """Plot bar chart of EER by transformer layer for one SSL model."""
    layers = sorted(results)
    eers   = [results[layer] for layer in layers]
    best   = min(results, key=results.get)
    colors = ["#e05c5c" if layer == best else "#4b8bbe" for layer in layers]

    fig, ax = plt.subplots(figsize=(12, 4))
    bars = ax.bar(layers, eers, color=colors, edgecolor="white")
    for bar, v in zip(bars, eers):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15,
                f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xlabel("Transformer Layer Index", fontsize=12)
    ax.set_ylabel("EER (%) — clean eval", fontsize=12)
    ax.set_title(f"{model.upper()} Layer-wise EER  |  best layer {best} "
                 f"({results[best]:.2f}%, red)", fontsize=12)
    ax.set_xticks(layers)
    ax.grid(axis="y", alpha=0.3)
    out = out_dir / f"{model}_layer_eer.png"
    fig.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] {out}")


def main() -> None:
    """Run train/eval loop across layers and save sweep artifacts."""
    args    = parse_args()
    out_dir = Path(f"outputs/sweep/{args.model}")
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{args.model}_layer_sweep.json"

    if not args.no_cache:
        _prewarm_cache(args)

    # Resume: load existing results
    results: dict[int, float] = {}
    if json_path.exists():
        with open(json_path) as f:
            results = {int(k): v for k, v in json.load(f).items()}
        print(f"[RESUME] {len(results)} results already in {json_path}")

    for layer in args.layers:
        print(f"\n{'='*55}\n[SWEEP] {args.model}  Layer {layer}\n{'='*55}")

        weights_exist = Path(f"models/{args.model}_ffn_layer{layer}.pth").exists()
        if not (weights_exist and args.skip_trained):
            _train_layer(args, layer)

        if layer in results:
            print(f"[SKIP] EER already recorded: {results[layer]:.4f}%")
            continue

        try:
            eer = _eval_layer(args, layer)
            results[layer] = eer
            print(f"[RESULT] Layer {layer} -> EER={eer:.4f}%")
        except Exception as exc:
            print(f"[ERROR] Layer {layer}: {exc}")
            results[layer] = float("nan")

        with open(json_path, "w") as f:
            json.dump(results, f, indent=2)

    valid = {layer: v for layer, v in results.items() if v == v}  # drop NaN
    if not valid:
        print("[ERROR] No valid results.")
        return

    best = min(valid, key=valid.get)
    print(f"\n{'Layer':>6}  {'EER (%)':>10}")
    print("─" * 22)
    for layer in sorted(results):
        marker = "  <- best" if layer == best else ""
        print(f"{layer:>6}  {results[layer]:>10.4f}{marker}")

    print("\nNext step — train weighted model:")
    print(f"  python train_ssl_backend.py --model {args.model} --mode weighted")
    print("Then run laundering eval:")
    print(f"  python eval_suite.py --model {args.model} [--run_cka]")

    _plot(results, args.model, out_dir)
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[DONE] {json_path}")


if __name__ == "__main__":
    main()
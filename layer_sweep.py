"""
Sweeps all 12 layers for a given SSL model.
Trains a probe per layer, runs clean eval, reports EER per layer.
Usage:
    python layer_sweep.py --model wav2vec2 --data_root data/ASVspoof2019/LA
"""
import argparse
import subprocess
import json
from pathlib import Path

import numpy as np
import yaml


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=["wav2vec2", "hubert", "wavlm"])
    p.add_argument("--data_root", default="data/ASVspoof2019/LA")
    p.add_argument("--epochs", type=int, default=10,
                   help="Fewer epochs OK for sweep — full training later on best layer")
    p.add_argument("--max_train", type=int, default=5000,
                   help="Cap samples for speed during sweep")
    p.add_argument("--num_layers", type=int, default=12)
    return p.parse_args()


def main():
    args = parse_args()
    results = {}   # layer → EER

    for layer in range(args.num_layers):
        print(f"\n{'='*50}")
        print(f"[SWEEP] Model={args.model}  Layer={layer}")
        print(f"{'='*50}")

        # --- Train probe for this layer ---
        subprocess.run([
            "python", "train_probe.py",
            "--model", args.model,
            "--layer", str(layer),
            "--data_root", args.data_root,
            "--epochs", str(args.epochs),
            "--max_train", str(args.max_train),
        ], check=True)

        # --- Update config to point to this layer's probe ---
        config_path = Path(f"configs/{args.model}_probe.yaml")
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        cfg["probe_layer"] = layer
        with open(config_path, "w") as f:
            yaml.dump(cfg, f)

        # --- Run clean eval (depth=0) ---
        outdir = f"outputs/sweep/{args.model}/layer{layer}"
        subprocess.run([
            "python", "run.py",
            "--model", args.model,
            "--depth", "0",
            "--output", outdir,
            "--data_root", args.data_root,
        ], check=True)

        # --- Read EER from CSV ---
        csv_path = Path(outdir) / args.model / "clean" / "k0" / "" / "results.csv"
        # find results.csv anywhere under outdir
        csv_files = list(Path(outdir).rglob("results.csv"))
        if csv_files:
            import csv
            with open(csv_files[0]) as f:
                reader = csv.DictReader(f)
                row = next(reader)
                eer = float(row["eer"])
            results[layer] = eer
            print(f"[SWEEP] Layer {layer} → EER: {eer:.4f}%")
        else:
            print(f"[SWEEP] Layer {layer} → results.csv not found, skipping")
            results[layer] = None

    # --- Save sweep results ---
    out_path = Path(f"outputs/sweep/{args.model}_layer_sweep.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[SWEEP] Results saved → {out_path}")

    # --- Print summary table ---
    print(f"\n{'Layer':>6}  {'EER (%)':>10}")
    print("-" * 20)
    for layer, eer in sorted(results.items()):
        marker = " ← best" if eer == min(v for v in results.values() if v) else ""
        print(f"{layer:>6}  {eer:>10.4f}{marker}")

    # --- Auto-update config with best layer ---
    valid = {k: v for k, v in results.items() if v is not None}
    best_layer = min(valid, key=valid.get)
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    cfg["probe_layer"] = best_layer
    with open(config_path, "w") as f:
        yaml.dump(cfg, f)
    print(f"\n[SWEEP] Config updated → probe_layer: {best_layer}")


if __name__ == "__main__":
    main()

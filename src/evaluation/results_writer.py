"""Helpers for writing evaluation result tables to disk."""

import csv
from pathlib import Path


def write_csv(all_results: list[dict], output_dir: str):
    """
    all_results: list of dicts with keys:
      model, pipeline, depth, strength, eer, tdcf
    """
    out = Path(output_dir) / "results.csv"
    fieldnames = ["model", "pipeline", "depth", "strength", "eer", "tdcf"]

    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)

    print(f"[results] Saved → {out}")
    return out

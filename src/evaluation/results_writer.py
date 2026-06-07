"""Helpers for writing reusable evaluation artifacts to disk."""

import csv
import json
from pathlib import Path
from datetime import datetime, timezone

import numpy as np

from src.evaluation.metrics import EvalResult, compute_eer


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


def _json_safe(value):
    """Convert numpy scalars and non-finite floats into JSON-safe values."""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    return value


def _read_score_rows(score_path: str | Path) -> list[dict]:
    """Read `utt_id source key score` CM score rows."""
    rows = []
    with open(score_path) as f:
        for line_no, line in enumerate(f, start=1):
            parts = line.strip().split()
            if not parts:
                continue
            if len(parts) != 4:
                raise ValueError(f"Invalid score row at {score_path}:{line_no}: {line.rstrip()}")
            utt_id, source, key, score = parts
            rows.append({"utt_id": utt_id, "source": source, "key": key, "score": float(score)})
    return rows


def compute_score_stats(score_path: str | Path) -> dict:
    """Compute bonafide/spoof score distribution statistics for one condition."""
    rows = _read_score_rows(score_path)
    bona = np.array([r["score"] for r in rows if r["key"] == "bonafide"], dtype=float)
    spoof = np.array([r["score"] for r in rows if r["key"] == "spoof"], dtype=float)

    def _stats(prefix: str, values: np.ndarray) -> dict:
        if values.size == 0:
            return {
                f"{prefix}_mean": None,
                f"{prefix}_std": None,
                f"{prefix}_median": None,
                f"{prefix}_min": None,
                f"{prefix}_q25": None,
                f"{prefix}_q75": None,
                f"{prefix}_max": None,
            }
        return {
            f"{prefix}_mean": float(np.mean(values)),
            f"{prefix}_std": float(np.std(values)),
            f"{prefix}_median": float(np.median(values)),
            f"{prefix}_min": float(np.min(values)),
            f"{prefix}_q25": float(np.percentile(values, 25)),
            f"{prefix}_q75": float(np.percentile(values, 75)),
            f"{prefix}_max": float(np.max(values)),
        }

    stats = {
        "num_bonafide": int(bona.size),
        "num_spoof": int(spoof.size),
        **_stats("bonafide", bona),
        **_stats("spoof", spoof),
    }
    if bona.size > 0 and spoof.size > 0:
        bona_mean = float(np.mean(bona))
        spoof_mean = float(np.mean(spoof))
        bona_var = float(np.var(bona))
        spoof_var = float(np.var(spoof))
        score_gap = bona_mean - spoof_mean
        fisher = float((score_gap ** 2) / (bona_var + spoof_var + 1e-12))
        _, eer_threshold = compute_eer(bona, spoof)
        stats.update({
            "score_gap": score_gap,
            "fisher_separability": fisher,
            "SCI": fisher,
            "eer_threshold": eer_threshold,
        })
    else:
        stats.update({
            "score_gap": None,
            "fisher_separability": None,
            "SCI": None,
            "eer_threshold": None,
        })
    return stats


def write_condition_artifacts(
    output_dir: str | Path,
    score_path: str | Path,
    result: EvalResult,
    condition_meta: dict,
) -> dict:
    """Write machine-readable artifacts for one evaluated condition."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    score_path = Path(score_path)
    score_rows = _read_score_rows(score_path)
    stats = compute_score_stats(score_path)

    meta = {
        **condition_meta,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "score_file": score_path.name,
        "artifact_version": 1,
    }

    metrics = {
        "eer": float(result.eer),
        "tdcf": float(result.min_tdcf),
        "min_tdcf": float(result.min_tdcf),
        "eer_threshold": stats["eer_threshold"],
        "SCI": stats["SCI"],
        "score_gap": stats["score_gap"],
        "fisher_separability": stats["fisher_separability"],
        "num_bonafide": stats["num_bonafide"],
        "num_spoof": stats["num_spoof"],
        "condition": meta,
    }

    with open(out / "metrics.json", "w") as f:
        json.dump(_json_safe(metrics), f, indent=2)
    with open(out / "score_stats.json", "w") as f:
        json.dump(_json_safe(stats), f, indent=2)
    with open(out / "condition_meta.json", "w") as f:
        json.dump(_json_safe(meta), f, indent=2)

    with open(out / "det_curve.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["threshold", "frr", "far", "frr_pct", "far_pct"])
        writer.writeheader()
        for thr, frr, far in zip(result.det_thr, result.det_frr, result.det_far):
            writer.writerow({
                "threshold": float(thr),
                "frr": float(frr),
                "far": float(far),
                "frr_pct": float(frr * 100),
                "far_pct": float(far * 100),
            })
    np.savez_compressed(
        out / "det_curve.npz",
        threshold=np.asarray(result.det_thr),
        frr=np.asarray(result.det_frr),
        far=np.asarray(result.det_far),
        frr_pct=np.asarray(result.det_frr) * 100,
        far_pct=np.asarray(result.det_far) * 100,
    )

    with open(out / "per_attack_eer.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["attack", "eer"])
        writer.writeheader()
        for attack, eer in result.eer_per_attack.items():
            writer.writerow({"attack": attack, "eer": eer})

    with open(out / "score_distribution.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["utt_id", "source", "key", "score"])
        writer.writeheader()
        writer.writerows(score_rows)

    return {"metrics": metrics, "score_stats": stats, "condition_meta": meta}


def write_representation_drift_artifacts(
    output_dir: str | Path,
    model_name: str,
    pipeline: str,
    strength: str,
    cka_res: dict[int, dict[int, float]],
    cos_res: dict[int, dict[int, float]],
) -> dict[str, Path]:
    """Write CKA/cosine representation drift rows in reusable CSV form."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{pipeline}_{strength}"
    drift_csv = out / f"{tag}_drift.csv"
    fieldnames = [
        "model",
        "pipeline",
        "strength",
        "depth",
        "layer",
        "cosine_similarity",
        "cosine_drift",
        "cka",
    ]

    rows = []
    for depth in sorted(cos_res):
        for layer in sorted(cos_res[depth]):
            cosine = float(cos_res[depth][layer])
            rows.append({
                "model": model_name,
                "pipeline": pipeline,
                "strength": strength,
                "depth": int(depth),
                "layer": int(layer),
                "cosine_similarity": cosine,
                "cosine_drift": float(1.0 - cosine),
                "cka": cka_res.get(depth, {}).get(layer),
            })

    with open(drift_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return {"drift_csv": drift_csv}

def write_layer_weights_artifacts(
    output_dir: str | Path,
    model_name: str,
    weights: dict[int, float],
) -> dict[str, Path]:
    """
    Write learned SSL layer importance weights to disk.

    Weights are static post-training and model-level — not per condition.
    Writes:
      - layer_weights.json   : {layer_idx: weight, ..., meta: {...}}
      - layer_weights.csv    : one row per layer, for spreadsheet/report use

    Args:
        output_dir: directory to write into (e.g. outputs/{model}/layer_weights/)
        model_name: model identifier string
        weights:    {layer_idx: weight} — post-softmax, should sum to ~1.0

    Returns:
        {"json": Path, "csv": Path}
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    weight_sum = float(sum(weights.values()))
    top_layer = max(weights, key=weights.__getitem__)

    payload = {
        "model": model_name,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifact_version": 1,
        "num_layers": len(weights),
        "weight_sum": weight_sum,          # sanity check — should be ~1.0
        "top_layer": int(top_layer),
        "top_layer_weight": float(weights[top_layer]),
        "weights": {str(k): float(v) for k, v in sorted(weights.items())},
    }

    json_path = out / "layer_weights.json"
    with open(json_path, "w") as f:
        json.dump(_json_safe(payload), f, indent=2)

    csv_path = out / "layer_weights.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "layer", "weight", "rank"])
        writer.writeheader()
        ranked = sorted(weights, key=weights.__getitem__, reverse=True)
        rank_map = {layer: rank + 1 for rank, layer in enumerate(ranked)}
        for layer in sorted(weights):
            writer.writerow({
                "model": model_name,
                "layer": int(layer),
                "weight": float(weights[layer]),
                "rank": rank_map[layer],
            })

    return {"json": json_path, "csv": csv_path}

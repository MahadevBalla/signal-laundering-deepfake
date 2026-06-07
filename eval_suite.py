"""Run the full laundering evaluation grid for one model.

This script is the main evaluation entry point. It executes clean and
laundered conditions, supports resume via a master CSV, and optionally runs
CKA/cosine representation analysis for SSL-based models.

Usage:
    python eval_suite.py --model aasist --dry_run
    python eval_suite.py --model wav2vec2 --run_cka
    python eval_suite.py --model hubert_aasist --pipelines N M --strengths M H --depths 0 1 2 3
"""

from __future__ import annotations

import sys
import argparse
import csv
import json
import logging
import time
from pathlib import Path

import numpy as np

from src.laundering import LaunderingEngine
from src.models.registry import get_model
from src.models.model_config import CONFIGS, WEIGHTS, SSL_MODELS
from src.evaluation.plots import (
    plot_collapse_curves,
    plot_strength_heatmap,
    plot_aurc_comparison,
    plot_cka_heatmap,
    plot_det_curve,
    plot_layerwise_drift_vs_depth,
    plot_mean_cosine_drift_vs_depth,
    plot_per_attack_eer,
    plot_score_distribution_shift,
    plot_layer_weight_distribution,
    plot_layer_weight_vs_cka,
    plot_sci_degradation,
)
from src.evaluation.metrics import (
    compute_aurc,
    compute_aurc_trap,
    compute_collapse_depth,
    compute_collapse_strength,
    compute_degradation_rate,
    compute_eer_amplification_factor,
    compute_relative_collapse_depth,
)
from src.evaluation.results_writer import write_condition_artifacts, write_representation_drift_artifacts, write_layer_weights_artifacts

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", module="transformers")
warnings.filterwarnings("ignore", module="huggingface_hub")

AASIST_ROOT = Path(__file__).resolve().parent / "external" / "aasist"
if str(AASIST_ROOT) not in sys.path:
    sys.path.insert(0, str(AASIST_ROOT))

ALL_PIPELINES = ["N", "M", "P"]
ALL_STRENGTHS = ["L", "M", "H"]
ALL_DEPTHS    = [0, 1, 2, 3]
MASTER_CSV_FIELDS = ["model", "pipeline", "strength", "depth", "eer", "tdcf", "elapsed_s"]


def parse_args():
    """Parse CLI options for full-grid evaluation."""
    p = argparse.ArgumentParser()
    p.add_argument("--model",       required=True, choices=list(CONFIGS))
    p.add_argument("--pipelines",   nargs="+", default=ALL_PIPELINES, choices=ALL_PIPELINES)
    p.add_argument("--strengths",   nargs="+", default=ALL_STRENGTHS, choices=ALL_STRENGTHS)
    p.add_argument("--depths",      nargs="+", type=int, default=ALL_DEPTHS)
    p.add_argument("--data_root",   default="data/ASVspoof2019/LA")
    p.add_argument("--config_dir",  default="configs")
    p.add_argument("--output_dir",  default="outputs/eval_suite")
    p.add_argument("--weights",     default=None,
                   help="Optional checkpoint path to override registry default.")
    p.add_argument("--dry_run",     action="store_true")
    p.add_argument("--max_eval",    type=int, default=None)
    p.add_argument("--run_cka",     action="store_true")
    p.add_argument("--cka_max_eval", type=int, default=1000)
    p.add_argument("--no_resume",   action="store_true")
    return p.parse_args()


def setup_logging(output_dir):
    """Create a logger that writes to both console and eval log file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger("eval_suite")
    log.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S")
    fh = logging.FileHandler(output_dir / "eval_suite.log", mode="a")
    fh.setFormatter(fmt)
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    if not log.handlers:
        log.addHandler(fh)
        log.addHandler(ch)
    return log


def _condition_key(pipeline, strength, depth):
    """Build a stable key used for resume and deduplication."""
    return f"{pipeline}|{strength}|{depth}"


def append_result(csv_path, row):
    """Append one evaluated condition to the master CSV."""
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MASTER_CSV_FIELDS)
        if write_header:
            w.writeheader()
        w.writerow(row)


def read_model_results(csv_path, model):
    """Load all rows in the master CSV for a specific model key."""
    if not csv_path.exists():
        return []
    with open(csv_path, newline="") as f:
        return [r for r in csv.DictReader(f) if r["model"] == model]


def evaluate_condition(model, engine, pipeline, strength, depth, cond_outdir, max_eval, log):
    """Evaluate one (pipeline, strength, depth) condition and return a result row."""
    launder_fn = engine.get_batch_fn(pipeline, depth, strength) if depth > 0 else None
    label = "clean" if depth == 0 else f"{pipeline}_{strength}_k{depth}"
    t0 = time.time()
    eer, tdcf = model.evaluate(output_dir=str(cond_outdir), launder_fn=launder_fn, max_eval=max_eval)
    elapsed = time.time() - t0
    result_obj = getattr(model, "_last_eval_result", None)
    cfg = getattr(model, "config", None)
    model_tag = cfg.get("model_type", type(model).__name__) if isinstance(cfg, dict) else type(model).__name__
    if result_obj is not None:
        write_condition_artifacts(
            output_dir=cond_outdir,
            score_path=cond_outdir / "eval_scores.txt",
            result=result_obj,
            condition_meta={
                "model": model_tag,
                "pipeline": pipeline,
                "strength": strength,
                "depth": depth,
                "condition_label": label,
                "max_eval": max_eval,
                "elapsed_s": round(elapsed, 3),
            },
        )
        plot_det_curve({label: result_obj}, str(cond_outdir), model_tag, condition_label=label)
        plot_per_attack_eer(result_obj.eer_per_attack, str(cond_outdir), model_tag, condition_label=label)
    log.info(f"  {label:30s}  EER={eer:6.3f}%  tDCF={tdcf:.4f}  [{elapsed:.0f}s]")
    return {
        "pipeline": pipeline,
        "strength": strength,
        "depth": depth,
        "eer": round(eer, 6),
        "tdcf": round(tdcf, 6),
        "elapsed_s": round(elapsed, 1),
    }


def run_cka_analysis(
    model, engine, pipeline, strength, depths, max_eval, output_dir, log
):
    """Run layer-wise CKA and cosine stability analysis for one pipeline-strength pair."""
    from src.evaluation.cka import cka_layer_stability, cosine_stability

    non_zero = [d for d in depths if d > 0]
    if not non_zero:
        return
    cka_dir = output_dir / "cka"
    cka_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{pipeline}_{strength}"
    if (cka_dir / f"{tag}_cka.json").exists():
        log.info(f"[CKA] {tag} already done")
        return

    # Extract 2× utterances for within-condition baseline split
    log.info(f"[CKA] clean embeddings max={max_eval * 2} (2x for baseline split)")
    clean_full = model.extract_all_layers(
        output_dir=str(output_dir / "cka_tmp/clean"),
        launder_fn=None,
        max_eval=max_eval * 2,
        save_embeddings=False,
    )
    N = next(iter(clean_full.values())).shape[0]
    half = N // 2
    clean_a = {layer_idx: v[:half] for layer_idx, v in clean_full.items()}
    clean_b = {layer_idx: v[half:] for layer_idx, v in clean_full.items()}

    # Within-condition baseline CKA — the ceiling any laundered score is measured against
    baseline_cka = {
        int(layer_idx): float(v) for layer_idx, v in cka_layer_stability(clean_a, clean_b).items()
    }
    baseline_cos = {
        int(layer_idx): float(v) for layer_idx, v in cosine_stability(clean_a, clean_b).items()
    }
    with open(cka_dir / f"{tag}_baseline_cka.json", "w") as f:
        json.dump({"cka": baseline_cka, "cosine": baseline_cos}, f, indent=2)
    log.info(
        f"  Baseline CKA (clean_a vs clean_b): { {layer_idx: f'{v:.3f}' for layer_idx, v in baseline_cka.items()} }"
    )

    clean = clean_a
    del clean_full, clean_b

    cka_res = {}
    cos_res = {}
    for depth in non_zero:
        log.info(f"[CKA] {pipeline} {strength} k={depth}")
        lnd = model.extract_all_layers(
            output_dir=str(output_dir / f"cka_tmp/{tag}_k{depth}"),
            launder_fn=engine.get_batch_fn(pipeline, depth, strength),
            max_eval=max_eval,
            save_embeddings=False,
        )
        cka_res[depth] = {
            int(l): float(v) for l, v in cka_layer_stability(clean, lnd).items()
        }
        cos_res[depth] = {
            int(l): float(v) for l, v in cosine_stability(clean, lnd).items()
        }
        log.info(f"  CKA: { {l: f'{v:.3f}' for l,v in cka_res[depth].items()} }")
        del lnd
    del clean

    with open(cka_dir / f"{tag}_cka.json", "w") as f:
        json.dump(cka_res, f, indent=2)
    with open(cka_dir / f"{tag}_cosine.json", "w") as f:
        json.dump(cos_res, f, indent=2)

    cfg = getattr(model, "config", None)
    mname = cfg.get("model_type", "ssl") if isinstance(cfg, dict) else "ssl"
    write_representation_drift_artifacts(
        output_dir=cka_dir,
        model_name=mname,
        pipeline=pipeline,
        strength=strength,
        cka_res=cka_res,
        cos_res=cos_res,
    )

    # Inject baseline as depth-label "baseline" row in the heatmap
    cka_with_baseline = {"baseline": baseline_cka, **cka_res}
    if cka_res:
        plot_cka_heatmap(cka_with_baseline, str(cka_dir), mname, pipeline, strength)
    if cos_res:
        _plot_cosine_drift(cos_res, cka_dir, mname, pipeline, strength)
        plot_mean_cosine_drift_vs_depth(
            cos_res, str(cka_dir), mname, pipeline, strength
        )
        plot_layerwise_drift_vs_depth(cos_res, str(cka_dir), mname, pipeline, strength)


def run_layer_weight_drift_analysis(
    model,
    output_dir: Path,
    log: logging.Logger,
) -> dict[int, float] | None:
    """
    Extract and save learned SSL layer importance weights.

    Weights are static post-training, so this is called once per model run.
    Saves layer_weights.json and generates the weight distribution bar chart.

    Returns:
        {layer_idx: weight} dict, or None if model does not expose get_layer_weights().
    """
    if not hasattr(model, "get_layer_weights"):
        log.info("[LAYER_WEIGHTS] model does not expose get_layer_weights() — skipping")
        return None

    try:
        raw_weights = model.get_layer_weights()
    except Exception as e:
        log.warning(f"[LAYER_WEIGHTS] get_layer_weights() raised: {e} — skipping")
        return None

    if isinstance(raw_weights, dict):
        weights = {int(k): float(v) for k, v in raw_weights.items()}
    else:
        weights = {i: float(v) for i, v in enumerate(raw_weights)}

    cfg = getattr(model, "config", None)
    mname = cfg.get("model_type", "ssl") if isinstance(cfg, dict) else "ssl"

    weights_dir = output_dir / "layer_weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    artifacts = write_layer_weights_artifacts(
        output_dir=weights_dir,
        model_name=mname,
        weights=weights,
    )
    log.info(f"[LAYER_WEIGHTS] Saved → {artifacts['json']}")

    plot_layer_weight_distribution(weights, str(weights_dir), mname)

    cka_dir = output_dir / "cka"
    if cka_dir.exists():
        for cka_file in cka_dir.glob("*_cka.json"):
            stem = cka_file.stem.replace("_cka", "")
            parts = stem.split("_")
            if len(parts) != 2:
                continue
            pipeline, strength = parts[0], parts[1]
            with open(cka_file) as f:
                cka_data = json.load(f)
            for depth_str, layer_cka in cka_data.items():
                cka_by_layer = {int(layer): float(v) for layer, v in layer_cka.items()}
                plot_layer_weight_vs_cka(
                    weights=weights,
                    cka_by_layer=cka_by_layer,
                    output_dir=str(weights_dir),
                    model_name=mname,
                    pipeline=pipeline,
                    depth=int(depth_str),
                    strength=strength,
                )
    else:
        log.info("[LAYER_WEIGHTS] No CKA results found yet — weight-vs-CKA scatter deferred")
    return weights


def _plot_cosine_drift(cos_res, out_dir, model_name, pipeline, strength):
    """Plot cosine-based representation drift as a depth-by-layer heatmap."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    depths = sorted(cos_res)
    layers = sorted(cos_res[depths[0]])
    matrix = np.array([[1.0 - cos_res[d][layer] for layer in layers] for d in depths])
    fig, ax = plt.subplots(figsize=(14, 4))
    sns.heatmap(matrix, xticklabels=[f"L{layer}" for layer in layers], yticklabels=[f"k={d}" for d in depths],
                annot=True, fmt=".3f", cmap="Blues", vmin=0.0, vmax=1.0, ax=ax,
                cbar_kws={"label": "Drift (1 - cosine)"})
    ax.set_xlabel("Layer")
    ax.set_ylabel("Laundering Depth")
    ax.set_title(f"Cosine Drift — {model_name.upper()} | {pipeline} | {strength}")
    plt.tight_layout()
    plt.savefig(str(out_dir / f"cosine_drift_{model_name}_{pipeline}_{strength}.png"), dpi=150, bbox_inches="tight")
    plt.close()


def _condition_artifact_path(output_dir, row, filename):
    """Return the per-condition artifact path for a master CSV row."""
    depth = int(row["depth"])
    if depth == 0 or row["pipeline"] == "clean":
        return output_dir / "clean" / "k0" / filename
    return output_dir / row["pipeline"] / row["strength"] / f"k{depth}" / filename


def _read_score_stat(output_dir, row, key):
    """Read one value from a condition's score_stats.json if available."""
    path = _condition_artifact_path(output_dir, row, "score_stats.json")
    if not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)
    return data.get(key)


def _load_sci_and_gap(output_dir: Path, results: list[dict], pipeline: str, strength: str) -> tuple[dict[int, float], dict[int, float]]:
    """
    Load SCI and score_gap for all depths for one pipeline-strength pair.

    k=0 is always read from the clean condition row.
    k>0 rows are filtered by pipeline and strength.

    Returns:
        (sci_by_depth, gap_by_depth) — only depths where both values are present.
    """
    sci: dict[int, float] = {}
    gap: dict[int, float] = {}

    clean_rows = [r for r in results if int(r["depth"]) == 0]
    if clean_rows:
        v_sci = _read_score_stat(output_dir, clean_rows[0], "SCI")
        v_gap = _read_score_stat(output_dir, clean_rows[0], "score_gap")
        if v_sci is not None and v_gap is not None:
            sci[0] = float(v_sci)
            gap[0] = float(v_gap)

    for r in results:
        if r["pipeline"] != pipeline or r["strength"] != strength or int(r["depth"]) == 0:
            continue
        v_sci = _read_score_stat(output_dir, r, "SCI")
        v_gap = _read_score_stat(output_dir, r, "score_gap")
        if v_sci is not None and v_gap is not None:
            d = int(r["depth"])
            sci[d] = float(v_sci)
            gap[d] = float(v_gap)

    # Keep only depths present in both
    common = set(sci) & set(gap)
    return {d: sci[d] for d in common}, {d: gap[d] for d in common}


def _load_score_arrays(output_dir: Path, row: dict) -> dict | None:
    """Read bonafide/spoof score arrays from a condition's score_distribution.csv."""
    path = _condition_artifact_path(output_dir, row, "score_distribution.csv")
    if not path.exists():
        return None
    bona, spoof = [], []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if r["key"] == "bonafide":
                bona.append(float(r["score"]))
            elif r["key"] == "spoof":
                spoof.append(float(r["score"]))
    if not bona or not spoof:
        return None
    return {"bonafide": np.array(bona, dtype=float), "spoof": np.array(spoof, dtype=float)}


def _write_framework_metrics_csv(metrics, output_path):
    """Write flattened framework metrics for spreadsheet/report use."""
    fieldnames = [
        "pipeline",
        "aurc_trap",
        "aurc_mean",
        "collapse_depth_kc",
        "kc_2x",
        "kc_3x",
        "kc_5pct",
        "kc_10pct",
        "kc_25pct",
        "relative_collapse_depth_kc_star",
        "collapse_strength_lc",
        "degradation_rate",
        "eer_amplification_factor",
        "delta_sci_k3",
        "delta_gap_k3",
        "sci_k0",
        "sci_k1",
        "sci_k2",
        "sci_k3",
    ]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for pipeline, values in metrics.items():
            sci_by_depth = values.get("sci_by_depth_M", {}) or {}
            writer.writerow(
                {
                    "pipeline": pipeline,
                    "aurc_trap": values.get("aurc_trap"),
                    "aurc_mean": values.get("aurc_mean"),
                    "collapse_depth_kc": values.get("collapse_depth_kc"),
                    "kc_2x": values.get("kc_2x"),
                    "kc_3x": values.get("kc_3x"),
                    "kc_5pct": values.get("kc_5pct"),
                    "kc_10pct": values.get("kc_10pct"),
                    "kc_25pct": values.get("kc_25pct"),
                    "relative_collapse_depth_kc_star": values.get(
                        "relative_collapse_depth_kc_star"
                    ),
                    "collapse_strength_lc": values.get("collapse_strength_lc"),
                    "degradation_rate": values.get("degradation_rate"),
                    "eer_amplification_factor": values.get("eer_amplification_factor"),
                    "delta_sci_k3": values.get("delta_sci_k3"),
                    "delta_gap_k3": values.get("delta_gap_k3"),
                    "sci_k0": sci_by_depth.get(0),
                    "sci_k1": sci_by_depth.get(1),
                    "sci_k2": sci_by_depth.get(2),
                    "sci_k3": sci_by_depth.get(3),
                }
            )


def generate_summary_plots(results, model_name, output_dir, log):
    """Generate summary plots and framework metrics from master CSV rows."""
    plots_dir = output_dir / "summary"
    plots_dir.mkdir(parents=True, exist_ok=True)

    def _eer(r): return float(r["eer"])

    clean_rows = [r for r in results if int(r["depth"]) == 0]
    clean_eer = _eer(clean_rows[0]) if clean_rows else None

    collapse_data = {}
    for r in results:
        if r["pipeline"] == "clean" or r["strength"] != "M":
            continue
        collapse_data.setdefault(r["pipeline"], {})[int(r["depth"])] = _eer(r)
    if clean_eer is not None:
        for p in collapse_data:
            collapse_data[p][0] = clean_eer
    if collapse_data:
        plot_collapse_curves(collapse_data, str(plots_dir), model_name)

    for td in [1, 3]:
        hm = {}
        for r in results:
            if int(r["depth"]) != td or r["pipeline"] == "clean":
                continue
            hm.setdefault(r["pipeline"], {})[r["strength"]] = _eer(r)
        if len(hm) == 3 and all(len(v) == 3 for v in hm.values()):
            plot_strength_heatmap(hm, str(plots_dir), f"{model_name}_k{td}")

    aurc_data = {}  # trapezoidal (primary); used in plot and framework metrics
    aurc_mean_data = {}  # mean (supplementary); written to JSON only
    for pipeline in ALL_PIPELINES:
        depth_eer = {
            int(r["depth"]): _eer(r)
            for r in results
            if r["pipeline"] == pipeline and r["strength"] == "M"
        }
        if clean_eer is not None:
            depth_eer[0] = clean_eer
        if depth_eer:
            aurc_data[pipeline] = compute_aurc_trap(depth_eer)
            aurc_mean_data[pipeline] = compute_aurc(depth_eer)
    if aurc_data:
        plot_aurc_comparison({model_name: aurc_data}, str(plots_dir))

    # Score distribution shift plots — one figure per pipeline-strength pair, all strengths
    for pipeline in ALL_PIPELINES:
        for strength in ALL_STRENGTHS:
            score_data: dict[int, dict] = {}
            if clean_rows:
                arrays = _load_score_arrays(output_dir, clean_rows[0])
                if arrays is not None:
                    score_data[0] = arrays
            for r in results:
                if r["pipeline"] == pipeline and r["strength"] == strength and int(r["depth"]) > 0:
                    arrays = _load_score_arrays(output_dir, r)
                    if arrays is not None:
                        score_data[int(r["depth"])] = arrays
            if len(score_data) >= 2:
                plot_score_distribution_shift(score_data, str(plots_dir), model_name, pipeline, strength)

    # ΔSCI and Δscore_gap trend plots — one figure per pipeline-strength pair
    for pipeline in ALL_PIPELINES:
        for strength in ALL_STRENGTHS:
            sci_by_depth, gap_by_depth = _load_sci_and_gap(
                output_dir, results, pipeline, strength
            )
            if len(sci_by_depth) >= 2:
                plot_sci_degradation(
                    sci_by_depth=sci_by_depth,
                    score_gap_by_depth=gap_by_depth,
                    output_dir=str(plots_dir),
                    model_name=model_name,
                    pipeline=pipeline,
                    strength=strength,
                )

    metrics = {}
    clean_sci = None
    if clean_rows:
        clean_sci = _read_score_stat(output_dir, clean_rows[0], "SCI")
    for pipeline in ALL_PIPELINES:
        eer_by_depth_m = {int(r["depth"]): _eer(r) for r in results if r["pipeline"] == pipeline and r["strength"] == "M"}
        if clean_eer is not None:
            eer_by_depth_m[0] = clean_eer
        sk1 = {r["strength"]: _eer(r) for r in results if r["pipeline"] == pipeline and int(r["depth"]) == 1}
        if not eer_by_depth_m:
            continue

        sci_by_depth: dict[int, float] = {}
        gap_by_depth: dict[int, float] = {}
        if clean_sci is not None:
            sci_by_depth[0] = clean_sci
            clean_gap = _read_score_stat(output_dir, clean_rows[0], "score_gap") if clean_rows else None
            if clean_gap is not None:
                gap_by_depth[0] = float(clean_gap)
        for r in results:
            if r["pipeline"] == pipeline and r["strength"] == "M":
                v_sci = _read_score_stat(output_dir, r, "SCI")
                v_gap = _read_score_stat(output_dir, r, "score_gap")
                if v_sci is not None:
                    sci_by_depth[int(r["depth"])] = v_sci
                if v_gap is not None:
                    gap_by_depth[int(r["depth"])] = float(v_gap)

        sci_0 = sci_by_depth.get(0)
        gap_0 = gap_by_depth.get(0)
        max_depth_present = max(sci_by_depth) if sci_by_depth else None
        delta_sci_k3 = (
            sci_by_depth.get(3, sci_by_depth.get(max_depth_present)) - sci_0
            if sci_0 is not None
            and max_depth_present is not None
            and max_depth_present > 0
            else None
        )
        delta_gap_k3 = (
            gap_by_depth.get(3, gap_by_depth.get(max_depth_present)) - gap_0
            if gap_0 is not None
            and max_depth_present is not None
            and max_depth_present > 0
            else None
        )

        kc_results = {
            "kc_2x": compute_collapse_depth(eer_by_depth_m, relative_factor=2.0),
            "kc_3x": compute_collapse_depth(eer_by_depth_m, relative_factor=3.0),
            "kc_5pct": compute_collapse_depth(eer_by_depth_m, tau=5.0),
            "kc_10pct": compute_collapse_depth(eer_by_depth_m, tau=10.0),
            "kc_25pct": compute_collapse_depth(eer_by_depth_m, tau=25.0),
        }
        metrics[pipeline] = {
            "aurc_trap": aurc_data.get(pipeline),
            "aurc_mean": aurc_mean_data.get(pipeline),
            "aurc": aurc_data.get(pipeline),
            "collapse_depth_kc": kc_results[
                "kc_2x"
            ],  # primary (2× relative, backward compat)
            **kc_results,
            "relative_collapse_depth_kc_star": compute_relative_collapse_depth(eer_by_depth_m),
            "collapse_strength_lc": (
                compute_collapse_strength(sk1, baseline_eer=clean_eer)
                if sk1 and clean_eer
                else None
            ),
            "degradation_rate": compute_degradation_rate(eer_by_depth_m),
            "eer_amplification_factor": compute_eer_amplification_factor(eer_by_depth_m),
            "sci_by_depth_M": sci_by_depth,
            "gap_by_depth_M": gap_by_depth,
            "delta_sci_k3": delta_sci_k3,
            "delta_gap_k3": delta_gap_k3,
            "eer_by_depth_M": eer_by_depth_m,
            "eer_by_strength_k1": sk1,
        }
    with open(plots_dir / f"{model_name}_framework_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    _write_framework_metrics_csv(metrics, plots_dir / f"{model_name}_framework_metrics.csv")
    log.info(f"[SUMMARY] Done → {plots_dir}")


def main():
    """Orchestrate full evaluation, optional CKA, and summary generation."""
    args = parse_args()
    output_dir = Path(args.output_dir) / args.model
    log = setup_logging(output_dir)
    master_csv = output_dir / "master_results.csv"

    max_eval = args.max_eval
    if max_eval is None and args.dry_run:
        max_eval = 200
        log.info(f"[DRY RUN] cap={max_eval}")

    completed = set()
    if not args.no_resume and master_csv.exists():
        with open(master_csv) as f:
            for row in csv.DictReader(f):
                if row["model"] == args.model:
                    completed.add(_condition_key(row["pipeline"], row["strength"], int(row["depth"])))
        if completed:
            log.info(f"[RESUME] {len(completed)} done")

    log.info(f"Model: {args.model}")
    weights_path = args.weights or WEIGHTS.get(args.model)
    if weights_path and not Path(weights_path).exists():
        bk = "aasist" if "_aasist" in args.model else ("rawnet2" if "_rawnet2" in args.model else "ffn")
        sb = args.model.split("_")[0]
        hint = (f"python train_ssl_backend.py --model {sb} --mode weighted --backend {bk}"
                if args.model in SSL_MODELS else "Check pretrained weights were downloaded.")
        raise FileNotFoundError(f"Weights not found: {weights_path}\n{hint}")
    if weights_path:
        log.info(f"Weights: {weights_path}")

    model = get_model(args.model, config_path=CONFIGS[args.model], data_root=args.data_root)
    model.load_weights(weights_path)
    engine = LaunderingEngine(args.config_dir)

    clean_key = _condition_key("clean", "-", 0)
    if clean_key not in completed:
        log.info("[EVAL] Clean baseline")
        cond_dir = output_dir / "clean" / "k0"
        cond_dir.mkdir(parents=True, exist_ok=True)
        row = evaluate_condition(model, engine, "clean", "-", 0, cond_dir, max_eval, log)
        row["model"] = args.model
        append_result(master_csv, row)
    else:
        log.info("[RESUME] Skipping clean baseline.")

    for pipeline in args.pipelines:
        for strength in args.strengths:
            for depth in args.depths:
                if depth == 0:
                    continue
                key = _condition_key(pipeline, strength, depth)
                if key in completed:
                    log.info(f"[RESUME] {key}")
                    continue
                log.info(f"[EVAL] {pipeline} {strength} k={depth}")
                cond_dir = output_dir / pipeline / strength / f"k{depth}"
                cond_dir.mkdir(parents=True, exist_ok=True)
                row = evaluate_condition(model, engine, pipeline, strength, depth, cond_dir, max_eval, log)
                row["model"] = args.model
                append_result(master_csv, row)

    if args.run_cka:
        if args.model not in SSL_MODELS:
            log.warning("[CKA] only for SSL models")
        elif not hasattr(model, "extract_all_layers"):
            log.error("[CKA] model has no extract_all_layers()")
        else:
            cka_max = min(args.cka_max_eval, 200) if args.dry_run else args.cka_max_eval
            for pipeline in args.pipelines:
                for strength in args.strengths:
                    run_cka_analysis(model, engine, pipeline, strength, args.depths, cka_max, output_dir, log)

    all_results = read_model_results(master_csv, args.model)
    if all_results:
        if args.model in SSL_MODELS:
            run_layer_weight_drift_analysis(model, output_dir, log)
        generate_summary_plots(all_results, args.model, output_dir, log)
    log.info(f"[DONE] {master_csv}")


if __name__ == "__main__":
    main()

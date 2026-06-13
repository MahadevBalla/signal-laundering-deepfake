"""Plotting helpers for laundering robustness and representation analysis."""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import gaussian_kde
from pathlib import Path


def plot_collapse_curves(results: dict, output_dir: str, model_name: str):
    """
    Obj 1+2 - EER vs laundering depth k, one line per pipeline.
    results: {pipeline: {depth: eer}} e.g. {'N': {0:0.83, 1:4.2, 2:11.3, 3:28.1}}
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    markers = {"N": "o", "M": "s", "P": "^"}

    for pipeline, depth_eer in results.items():
        depths = sorted(depth_eer.keys())
        eers = [depth_eer[k] for k in depths]
        ax.plot(
            depths,
            eers,
            marker=markers[pipeline],
            label=f"Pipeline {pipeline}",
            linewidth=2,
        )

    ax.set_xlabel("Laundering Depth (k)", fontsize=12)
    ax.set_ylabel("EER (%)", fontsize=12)
    ax.set_title(f"{model_name} - EER vs Laundering Depth (ℓ=M)", fontsize=13)
    ax.set_xticks([0, 1, 2, 3])
    ax.legend()
    ax.grid(True, alpha=0.3)

    out = Path(output_dir) / f"{model_name}_collapse_curves.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plots] Saved → {out}")


def plot_strength_heatmap(results: dict, output_dir: str, model_name: str):
    """
    Obj 3 - EER heatmap: rows=pipeline, cols=strength, at fixed depth k.
    results: {pipeline: {strength: eer}} at k=1 and k=3 separately
    Call this twice: once for k=1, once for k=3.
    """
    pipelines = ["N", "M", "P"]
    strengths = ["L", "M", "H"]

    matrix = np.array([[results[p][s] for s in strengths] for p in pipelines])

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        matrix,
        xticklabels=strengths,
        yticklabels=pipelines,
        annot=True,
        fmt=".2f",
        cmap="YlOrRd",
        ax=ax,
        cbar_kws={"label": "EER (%)"},
    )
    ax.set_xlabel("Strength (ℓ)", fontsize=11)
    ax.set_ylabel("Pipeline", fontsize=11)
    ax.set_title(f"{model_name} - EER by Pipeline × Strength", fontsize=12)

    out = Path(output_dir) / f"{model_name}_strength_heatmap.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plots] Saved → {out}")


def plot_aurc_comparison(aurc_dict: dict, output_dir: str):
    """
    Cross-model AURC bar chart.
    aurc_dict: {'AASIST': {'N':x,'M':y,'P':z}, 'RawNet2': {...}, ...}
    """
    models = list(aurc_dict.keys())
    pipelines = ["N", "M", "P"]
    x = np.arange(len(pipelines))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, model in enumerate(models):
        vals = [aurc_dict[model][p] for p in pipelines]
        ax.bar(x + i * width, vals, width, label=model)

    ax.set_xticks(x + width)
    ax.set_xticklabels([f"Pipeline {p}" for p in pipelines])
    ax.set_ylabel("AURC (mean EER %)", fontsize=12)
    ax.set_title("AURC Comparison Across Models and Pipelines", fontsize=13)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    out = Path(output_dir) / "aurc_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plots] Saved → {out}")


def plot_det_curve(
    results: dict,
    output_dir: str,
    model_name: str,
    condition_label: str = "clean",
) -> None:
    """
    DET curve: FRR vs FAR (both in %).
    results: {'pipeline_depth_strength': EvalResult} - each has det_frr/far arrays.
    One curve per condition.
    """
    fig, ax = plt.subplots(figsize=(6, 6))

    for label, result in results.items():
        # Convert to percent, skip first sentinel point
        frr_pct = result.det_frr[1:] * 100
        far_pct = result.det_far[1:] * 100
        ax.plot(far_pct, frr_pct, linewidth=1.5, label=label)

    ax.plot([0, 100], [0, 100], "k--", linewidth=0.8, alpha=0.4)
    ax.set_xlabel("FAR (%)", fontsize=12)
    ax.set_ylabel("FRR (%)", fontsize=12)
    ax.set_title(f"{model_name} - DET Curve ({condition_label})", fontsize=13)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    out = Path(output_dir) / f"{model_name}_det_{condition_label}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plots] Saved → {out}")


def plot_per_attack_eer(
    results: dict,
    output_dir: str,
    model_name: str,
    condition_label: str = "clean",
) -> None:
    """
    Per-attack EER bar chart for attacks A07–A19.
    results: {attack: eer_pct} - from EvalResult.eer_per_attack.
    Call once per (model, condition).
    """
    from src.evaluation.metrics import ATTACK_TYPES

    attacks = ATTACK_TYPES
    eers = [results.get(a, float("nan")) for a in attacks]

    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(attacks))
    bars = ax.bar(x, eers, color="steelblue", edgecolor="white")

    # Annotate values
    for bar, v in zip(bars, eers):
        if not np.isnan(v):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3,
                f"{v:.1f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(attacks, rotation=45, ha="right")
    ax.set_ylabel("EER (%)", fontsize=12)
    ax.set_title(f"{model_name} - Per-Attack EER ({condition_label})", fontsize=13)
    ax.grid(True, axis="y", alpha=0.3)

    out = Path(output_dir) / f"{model_name}_per_attack_{condition_label}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plots] Saved → {out}")


def plot_cka_heatmap(
    cka_data: dict,  # keys: int depth OR str "baseline"
    output_dir: str,
    model_name: str,
    pipeline: str,
    strength: str,
):
    """
    Heatmap: X-axis = layer (0–11), Y-axis = depth rows, color = CKA score.
    First row is the within-condition baseline (clean_A vs clean_B) when present.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # Separate baseline row from laundering-depth rows
    baseline_row = cka_data.get("baseline", None)
    depth_keys = sorted(k for k in cka_data if k != "baseline")
    layers = sorted(cka_data[depth_keys[0]].keys())

    rows = []
    row_labels = []
    if baseline_row is not None:
        rows.append([baseline_row[layer] for layer in layers])
        row_labels.append("baseline\n(clean A vs B)")
    for d in depth_keys:
        rows.append([cka_data[d][layer] for layer in layers])
        row_labels.append(f"k={d}")

    matrix = np.array(rows)

    fig, ax = plt.subplots(figsize=(14, max(3, len(row_labels) * 0.9 + 1)))
    im = ax.imshow(
        matrix, aspect="auto", cmap="RdYlGn", vmin=0.0, vmax=1.0, origin="upper"
    )

    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels([f"L{layer}" for layer in layers])
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)

    # Annotate cells
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=7, color="black")

    # Draw a separator line between baseline row and depth rows
    if baseline_row is not None:
        ax.axhline(0.5, color="white", linewidth=2)

    ax.set_xlabel("Transformer Layer")
    ax.set_ylabel("Condition")
    ax.set_title(
        f"CKA Stability — {model_name.upper()} | Pipeline {pipeline} | Strength {strength}\n"
        f"Baseline row = within-condition ceiling (clean_A vs clean_B)"
    )
    plt.colorbar(im, ax=ax, label="CKA Score (1=identical, 0=collapsed)")
    plt.tight_layout()

    out_path = Path(output_dir) / f"cka_heatmap_{model_name}_{pipeline}_{strength}.png"
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] CKA heatmap → {out_path}")


def plot_mean_cosine_drift_vs_depth(
    cos_data: dict[int, dict[int, float]],
    output_dir: str,
    model_name: str,
    pipeline: str,
    strength: str,
) -> None:
    """Line plot of mean cosine drift across layers as laundering depth increases."""
    depths = sorted(cos_data)
    mean_drift = [np.mean([1.0 - v for v in cos_data[d].values()]) for d in depths]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(depths, mean_drift, marker="o", linewidth=2, color="darkslateblue")
    ax.set_xticks(depths)
    ax.set_xlabel("Laundering Depth (k)", fontsize=11)
    ax.set_ylabel("Mean Cosine Drift (1 - cosine)", fontsize=11)
    ax.set_title(
        f"Mean Representation Drift — {model_name.upper()} | {pipeline} | {strength}"
    )
    ax.grid(True, alpha=0.3)

    out = (
        Path(output_dir)
        / f"mean_cosine_drift_vs_depth_{model_name}_{pipeline}_{strength}.png"
    )
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plots] Saved → {out}")


def plot_layerwise_drift_vs_depth(
    cos_data: dict[int, dict[int, float]],
    output_dir: str,
    model_name: str,
    pipeline: str,
    strength: str,
) -> None:
    """Line plot of cosine drift vs depth for every SSL layer."""
    depths = sorted(cos_data)
    layers = sorted(cos_data[depths[0]])

    fig, ax = plt.subplots(figsize=(8, 5))
    for layer in layers:
        drift = [1.0 - cos_data[d][layer] for d in depths]
        ax.plot(depths, drift, marker="o", linewidth=1.0, alpha=0.55, label=f"L{layer}")

    ax.set_xticks(depths)
    ax.set_xlabel("Laundering Depth (k)", fontsize=11)
    ax.set_ylabel("Cosine Drift (1 - cosine)", fontsize=11)
    ax.set_title(
        f"Layerwise Representation Drift — {model_name.upper()} | {pipeline} | {strength}"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=4, fontsize=7, frameon=False)

    out = (
        Path(output_dir)
        / f"layerwise_drift_vs_depth_{model_name}_{pipeline}_{strength}.png"
    )
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plots] Saved → {out}")


def plot_score_distribution_shift(
    score_data: dict[int, dict],
    output_dir: str,
    model_name: str,
    pipeline: str,
    strength: str,
) -> None:
    """
    4-panel KDE plot: one panel per depth k=0..3.
    Each panel overlays bonafide and spoof score distributions as filled KDEs.
    As laundering deepens, distributions should converge toward overlap.

    Args:
        score_data: {depth: {"bonafide": np.ndarray, "spoof": np.ndarray}}
        output_dir: directory to save the figure
        model_name: used in title and filename
        pipeline:   pipeline label (N / M / P)
        strength:   strength label (L / M / H)
    """

    depths = sorted(score_data.keys())
    n_panels = len(depths)
    fig, axes = plt.subplots(1, n_panels, figsize=(4 * n_panels, 4), sharey=False)
    if n_panels == 1:
        axes = [axes]

    # Shared x-range across all panels for visual comparability
    all_scores = np.concatenate(
        [
            np.concatenate([score_data[d]["bonafide"], score_data[d]["spoof"]])
            for d in depths
        ]
    )
    x_min, x_max = float(np.percentile(all_scores, 0.5)), float(
        np.percentile(all_scores, 99.5)
    )
    x_grid = np.linspace(x_min, x_max, 500)

    for ax, depth in zip(axes, depths):
        bona = score_data[depth]["bonafide"]
        spoof = score_data[depth]["spoof"]

        for scores, label, color in [
            (bona, "Bonafide", "#2196F3"),
            (spoof, "Spoof", "#F44336"),
        ]:
            if scores.size < 2:
                continue
            kde = gaussian_kde(scores, bw_method="scott")
            density = kde(x_grid)
            ax.fill_between(x_grid, density, alpha=0.35, color=color, label=label)
            ax.plot(x_grid, density, color=color, linewidth=1.5)

        ax.set_title(f"k={depth}", fontsize=11)
        ax.set_xlabel("CM Score", fontsize=10)
        ax.set_ylabel("Density" if depth == depths[0] else "", fontsize=10)
        ax.grid(True, alpha=0.25)
        ax.set_xlim(x_min, x_max)
        if depth == depths[0]:
            ax.legend(fontsize=9, frameon=False)

    fig.suptitle(
        f"{model_name.upper()} — Score Distribution Shift | Pipeline {pipeline} | Strength {strength}",
        fontsize=12,
        y=1.02,
    )
    plt.tight_layout()

    out = Path(output_dir) / f"score_dist_shift_{model_name}_{pipeline}_{strength}.png"
    fig.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plots] Saved → {out}")

def plot_layer_weight_distribution(
    weights: dict[int, float],
    output_dir: str,
    model_name: str,
) -> None:
    """
    Bar chart of learned SSL layer importance weights after softmax.

    Args:
        weights:    {layer_idx: weight} — from model.get_layer_weights()
        output_dir: directory to save the figure
        model_name: used in title and filename
    """
    layers = sorted(weights)
    vals = [weights[l] for l in layers]

    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar(
        [f"L{l}" for l in layers],
        vals,
        color="steelblue",
        edgecolor="white",
        linewidth=0.5,
    )

    # Annotate each bar with its weight value
    for bar, v in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.002,
            f"{v:.3f}",
            ha="center",
            va="bottom",
            fontsize=7,
        )

    # Highlight the top-3 layers
    top3 = sorted(layers, key=lambda l: weights[l], reverse=True)[:3]
    for bar, l in zip(bars, layers):
        if l in top3:
            bar.set_color("darkorange")

    ax.set_xlabel("SSL Layer Index", fontsize=11)
    ax.set_ylabel("Learned Weight (post-softmax)", fontsize=11)
    ax.set_title(
        f"{model_name.upper()} — Learned Layer Importance Weights\n"
        f"(orange = top-3 highest-weight layers)"
    )
    ax.set_ylim(0, max(vals) * 1.18)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()

    out = Path(output_dir) / f"layer_weight_distribution_{model_name}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plots] Saved → {out}")


def plot_layer_weight_vs_cka(
    weights: dict[int, float],
    cka_by_layer: dict[int, float],
    output_dir: str,
    model_name: str,
    pipeline: str,
    depth: int,
    strength: str = "M",
) -> None:
    """
    Scatter plot: learned layer weight (x) vs CKA stability (y).

    Tests whether the model has learned to rely on robust layers (high weight
    AND high CKA) or fragile ones (high weight but low CKA — the dangerous case).

    Args:
        weights:      {layer_idx: weight} — from model.get_layer_weights()
        cka_by_layer: {layer_idx: cka_score} — from cka_layer_stability() at one depth
        output_dir:   directory to save the figure
        model_name:   used in title and filename
        pipeline:     pipeline label (N / M / P)
        depth:        laundering depth k at which CKA was measured
        strength:     strength label (L / M / H), default M
    """
    # Intersect layers present in both dicts
    common = sorted(set(weights) & set(cka_by_layer))
    if len(common) < 2:
        print(f"[plots] Skipping weight-vs-CKA scatter — fewer than 2 common layers")
        return

    x = np.array([weights[l] for l in common])       # layer weight
    y = np.array([cka_by_layer[l] for l in common])  # CKA stability
    labels = [f"L{l}" for l in common]

    # Pearson correlation
    if x.std() > 0 and y.std() > 0:
        r = float(np.corrcoef(x, y)[0, 1])
        corr_label = f"r = {r:+.3f}"
    else:
        r = None
        corr_label = "r = N/A"

    fig, ax = plt.subplots(figsize=(6, 5))

    # Color points by quadrant: high-weight+high-CKA (green), high-weight+low-CKA (red)
    x_med = float(np.median(x))
    y_med = float(np.median(y))
    colors = []
    for xi, yi in zip(x, y):
        if xi >= x_med and yi >= y_med:
            colors.append("#2e7d32")   # robust and important — good
        elif xi >= x_med and yi < y_med:
            colors.append("#c62828")   # important but fragile — dangerous
        elif xi < x_med and yi >= y_med:
            colors.append("#1565c0")   # stable but low weight — irrelevant
        else:
            colors.append("#78909c")   # low weight, fragile — irrelevant

    ax.scatter(x, y, c=colors, s=80, zorder=3, edgecolors="white", linewidths=0.5)

    # Label each point
    for xi, yi, lbl in zip(x, y, labels):
        ax.annotate(
            lbl,
            (xi, yi),
            textcoords="offset points",
            xytext=(5, 4),
            fontsize=8,
            color="dimgray",
        )

    # Median crosshair lines
    ax.axvline(x_med, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.axhline(y_med, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)

    # Quadrant annotations
    ax.text(ax.get_xlim()[0] if ax.get_xlim()[0] else x.min(),
            1.01, "Stable+Low-weight", fontsize=7, color="#1565c0", va="bottom")

    ax.set_xlabel("Learned Layer Weight (post-softmax)", fontsize=11)
    ax.set_ylabel(f"CKA Stability at k={depth}", fontsize=11)
    ax.set_title(
        f"{model_name.upper()} — Layer Weight vs CKA Stability\n"
        f"Pipeline {pipeline} | Strength {strength} | k={depth}   ({corr_label})",
        fontsize=11,
    )

    # Legend for quadrant colors
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#2e7d32",
               markersize=8, label="High weight + stable (ideal)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#c62828",
               markersize=8, label="High weight + fragile (dangerous)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#1565c0",
               markersize=8, label="Low weight + stable"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#78909c",
               markersize=8, label="Low weight + fragile"),
    ]
    ax.legend(handles=legend_elements, fontsize=7, frameon=True, loc="lower right")
    ax.grid(True, alpha=0.25)
    plt.tight_layout()

    out = (
        Path(output_dir)
        / f"layer_weight_vs_cka_{model_name}_{pipeline}_{strength}_k{depth}.png"
    )
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plots] Saved → {out}")

def plot_sci_degradation(
    sci_by_depth: dict[int, float],
    score_gap_by_depth: dict[int, float],
    output_dir: str,
    model_name: str,
    pipeline: str,
    strength: str,
) -> None:
    """
    Dual-axis line plot: SCI and score_gap vs laundering depth.

    Left axis  — SCI (Fisher separability): how well bonafide/spoof clusters
                 are separated. Drops toward 0 as laundering collapses classes.
    Right axis — score_gap (bonafide_mean - spoof_mean): raw margin between
                 class centroids. Can go negative when spoof scores exceed bonafide.

    ΔSCI and Δscore_gap are annotated as secondary series (dashed) relative to k=0.
    """
    depths = sorted(set(sci_by_depth) & set(score_gap_by_depth))
    if len(depths) < 2:
        print(f"[plots] Skipping SCI degradation plot — fewer than 2 common depths")
        return

    sci_vals = [sci_by_depth[d] for d in depths]
    gap_vals = [score_gap_by_depth[d] for d in depths]

    # Deltas relative to k=0 baseline
    sci_0 = sci_by_depth[depths[0]]
    gap_0 = score_gap_by_depth[depths[0]]
    delta_sci = [v - sci_0 for v in sci_vals]
    delta_gap = [v - gap_0 for v in gap_vals]

    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax2 = ax1.twinx()

    # Absolute values — solid lines
    l1, = ax1.plot(depths, sci_vals,  color="#1565c0", marker="o", linewidth=2,
                   label="SCI (abs)")
    l2, = ax2.plot(depths, gap_vals,  color="#c62828", marker="s", linewidth=2,
                   label="score_gap (abs)")

    # Delta values — dashed lines, same colours
    l3, = ax1.plot(depths, delta_sci, color="#1565c0", marker="o", linewidth=1.5,
                   linestyle="--", alpha=0.65, label="ΔSCI")
    l4, = ax2.plot(depths, delta_gap, color="#c62828", marker="s", linewidth=1.5,
                   linestyle="--", alpha=0.65, label="Δscore_gap")

    # Zero-reference line on both axes
    ax1.axhline(0, color="#1565c0", linewidth=0.5, alpha=0.3)
    ax2.axhline(0, color="#c62828", linewidth=0.5, alpha=0.3)

    ax1.set_xlabel("Laundering Depth (k)", fontsize=11)
    ax1.set_ylabel("SCI / ΔSCI", fontsize=11, color="#1565c0")
    ax2.set_ylabel("score_gap / Δscore_gap", fontsize=11, color="#c62828")
    ax1.tick_params(axis="y", labelcolor="#1565c0")
    ax2.tick_params(axis="y", labelcolor="#c62828")
    ax1.set_xticks(depths)

    ax1.set_title(
        f"{model_name.upper()} — SCI & Score Gap Degradation\n"
        f"Pipeline {pipeline} | Strength {strength} | dashed = Δ from k=0",
        fontsize=11,
    )

    lines = [l1, l2, l3, l4]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, fontsize=8, frameon=True, loc="upper right")
    ax1.grid(True, alpha=0.25)
    plt.tight_layout()

    out = (
        Path(output_dir)
        / f"sci_degradation_{model_name}_{pipeline}_{strength}.png"
    )
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plots] Saved → {out}")

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
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


def plot_layer_drift(
    drift_matrix: np.ndarray, output_dir: str, model_name: str, pipeline: str
):
    """
    Obj 2 (SSL model only) - layer drift heatmap.
    drift_matrix: shape (num_layers, 4) - rows=layer, cols=depth k=0..3
    Values = 1 - cosine_similarity(h_l_clean, h_l_laundered)
    """
    fig, ax = plt.subplots(figsize=(6, 8))
    sns.heatmap(
        drift_matrix,
        xticklabels=[f"k={k}" for k in range(drift_matrix.shape[1])],
        annot=True,
        fmt=".3f",
        cmap="Blues",
        ax=ax,
        cbar_kws={"label": "Representation Drift (1 - cos sim)"},
    )
    ax.set_xlabel("Laundering Depth", fontsize=11)
    ax.set_ylabel("SSL Layer Index", fontsize=11)
    ax.set_title(f"{model_name} - Layer Drift, Pipeline {pipeline}", fontsize=12)

    out = Path(output_dir) / f"{model_name}_layer_drift_{pipeline}.png"
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

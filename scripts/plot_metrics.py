#!/usr/bin/env python3
"""Generate plots from gamma metric CSV files comparing MEA methods against Viterbi."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import re

from .constants import (
    ALIGNER_COLORS,
    EVALUATION_METRICS_FOLDER,
    MEA_METHODS,
    METHOD_COLORS,
    METRICS_FIGURES_FOLDER,
    PLOT_DPI,
    PLOT_GRID_ALPHA,
)

VITERBI_COLOR = ALIGNER_COLORS["viterbi"]


def load_all_metrics(metrics_dir: Path) -> pd.DataFrame:
    """Load all {method}_gamma_{gamma}.csv files and combine into single dataframe."""
    dfs = []

    # Pattern: {method}_gamma_{gamma}.csv
    for csv_file in sorted(metrics_dir.glob("*_gamma_*.csv")):
        # Extract method and gamma from filename
        match = re.match(r"(\w+)_gamma_(\d+\.?\d*)", csv_file.stem)
        if not match:
            continue

        method = match.group(1)
        gamma = float(match.group(2))

        # Load CSV and add method/gamma columns
        df = pd.read_csv(csv_file)
        df["method"] = method
        df["gamma"] = gamma
        dfs.append(df)

    if not dfs:
        raise ValueError(f"No metric CSV files found in {metrics_dir}")

    return pd.concat(dfs, ignore_index=True)


def get_viterbi_baseline(df: pd.DataFrame, metric: str) -> float:
    """Get Viterbi's mean value for a metric (constant across gamma)."""
    vit_data = df[(df["Aligner"] == "viterbi") & (df["Metric"] == metric)]
    if vit_data.empty:
        return float("nan")
    return vit_data["Mean"].mean()


def _symmetric_limits(
    above_vals: list[float],
    all_vals: list[float],
    baseline: float,
    pad_frac: float = 0.05,
    default_span: float = 0.01,
) -> tuple[float, float]:
    vals = above_vals if above_vals else all_vals
    if not vals:
        return baseline - default_span, baseline + default_span

    max_dev = max(abs(v - baseline) for v in vals)
    pad = pad_frac * max_dev if max_dev > 0 else default_span
    return baseline - (max_dev + pad), baseline + (max_dev + pad)


def plot_column_identity_vs_gamma(df: pd.DataFrame, output_dir: Path):
    """Plot column identity vs gamma for all MEA methods with Viterbi baseline."""
    ci_data = df[df["Metric"] == "column_identity"]

    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

    # Get Viterbi baseline (constant across gamma)
    viterbi_baseline = get_viterbi_baseline(df, "column_identity")

    # Collect all data points above baseline for zoom calculation
    all_values_above_baseline = []

    # Plot each MEA method on both subplots
    for method in MEA_METHODS:
        method_data = ci_data[
            (ci_data["method"] == method) & (ci_data["Aligner"] == "mea")
        ].sort_values("gamma")

        if method_data.empty:
            continue

        # Filter values above baseline for zoom
        above_baseline = method_data[method_data["Mean"] > viterbi_baseline]
        if not above_baseline.empty:
            all_values_above_baseline.extend(above_baseline["Mean"].tolist())

        # Plot on both subplots
        for ax in [ax1, ax2]:
            ax.plot(
                method_data["gamma"],
                method_data["Mean"],
                label=f"MEA ({method})",
                color=METHOD_COLORS.get(method, "#666666"),
                linewidth=2.5,
                marker="o",
                markersize=6,
            )

    # Plot Viterbi baseline on both subplots
    for ax in [ax1, ax2]:
        ax.axhline(
            y=viterbi_baseline,
            color=VITERBI_COLOR,
            linestyle="--",
            linewidth=2,
            label="Viterbi",
        )

    # Left subplot: full view
    ax1.set_xlabel("γ (gamma)", fontsize=12)
    ax1.set_ylabel("Mean Column Identity", fontsize=12)
    ax1.set_title(
        "Column Identity vs Gamma (All MEA Methods)", fontsize=14, fontweight="bold"
    )
    ax1.legend(loc="best", framealpha=0.9)
    ax1.grid(True, alpha=PLOT_GRID_ALPHA)

    # Right subplot: symmetric zoom around Viterbi baseline
    ci_all_vals = ci_data["Mean"].tolist()
    y_min, y_max = _symmetric_limits(
        above_vals=all_values_above_baseline,
        all_vals=ci_all_vals,
        baseline=viterbi_baseline,
    )
    ax2.set_ylim(y_min, y_max)

    ax2.set_xlabel("γ (gamma)", fontsize=12)
    ax2.set_ylabel("Mean Column Identity", fontsize=12)
    ax2.set_title("Zoom around Viterbi baseline", fontsize=14, fontweight="bold")
    ax2.legend(loc="best", framealpha=0.9)
    ax2.grid(True, alpha=PLOT_GRID_ALPHA)

    plt.tight_layout()
    output_path = output_dir / "column_identity_vs_gamma.png"
    plt.savefig(output_path, dpi=PLOT_DPI, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def plot_delta_f1_vs_gamma(df: pd.DataFrame, output_dir: Path):
    """Plot (MEA F1 - Viterbi F1) vs gamma for all methods, with zoom above baseline."""
    f1_data = df[df["Metric"] == "f1"].copy()

    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

    # Collect deltas for zoom calculation
    all_positive_deltas: list[float] = []
    all_deltas: list[float] = []

    for method in MEA_METHODS:
        method_data = f1_data[f1_data["method"] == method]
        if method_data.empty:
            continue

        pivot = method_data.pivot_table(index="gamma", columns="Aligner", values="Mean")
        if "mea" not in pivot.columns or "viterbi" not in pivot.columns:
            continue

        pivot["delta"] = pivot["mea"] - pivot["viterbi"]
        pivot = pivot.sort_index()

        positive_deltas = pivot[pivot["delta"] > 0]["delta"]
        if not positive_deltas.empty:
            all_positive_deltas.extend(positive_deltas.tolist())
        all_deltas.extend(pivot["delta"].tolist())

        for ax in (ax1, ax2):
            ax.plot(
                pivot.index,
                pivot["delta"],
                label=f"MEA ({method})",
                color=METHOD_COLORS.get(method, "#666666"),
                linewidth=2.5,
                marker="o",
                markersize=6,
            )

    # Viterbi baseline (delta = 0) on both subplots
    for ax in (ax1, ax2):
        ax.axhline(
            y=0,
            color=VITERBI_COLOR,
            linestyle="--",
            linewidth=2,
            label="Viterbi (baseline)",
        )

    # Left subplot: full view
    ax1.set_xlabel("γ (gamma)", fontsize=12)
    ax1.set_ylabel("ΔF1 (MEA - Viterbi)", fontsize=12)
    ax1.set_title(
        "MEA vs Viterbi F1 Difference (All Methods)", fontsize=14, fontweight="bold"
    )
    ax1.legend(loc="best", framealpha=0.9)
    ax1.grid(True, alpha=PLOT_GRID_ALPHA)

    # Right subplot: symmetric zoom around baseline (0)
    y_min, y_max = _symmetric_limits(
        above_vals=all_positive_deltas,
        all_vals=all_deltas,
        baseline=0.0,
    )
    ax2.set_ylim(y_min, y_max)

    ax2.set_xlabel("γ (gamma)", fontsize=12)
    ax2.set_ylabel("ΔF1 (MEA - Viterbi)", fontsize=12)
    ax2.set_title("Zoom around Viterbi baseline", fontsize=14, fontweight="bold")
    ax2.legend(loc="best", framealpha=0.9)
    ax2.grid(True, alpha=PLOT_GRID_ALPHA)

    plt.tight_layout()
    output_path = output_dir / "delta_f1_vs_gamma.png"
    plt.savefig(output_path, dpi=PLOT_DPI, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def plot_f1_precision_recall_vs_gamma(df: pd.DataFrame, output_dir: Path):
    """Plot F1, precision, recall vs gamma - one subplot per MEA method."""
    metrics = ["precision", "recall", "f1"]
    line_styles = {"precision": "--", "recall": ":", "f1": "-"}

    # Filter to methods that have data
    available_methods = [m for m in MEA_METHODS if m in df["method"].unique()]
    n_methods = len(available_methods)

    if n_methods == 0:
        print("No method data available for F1/precision/recall plot")
        return

    # Create subplot grid
    n_cols = 2
    n_rows = (n_methods + 1) // 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 5 * n_rows))
    axes = axes.flatten() if n_methods > 1 else [axes]

    for idx, method in enumerate(available_methods):
        ax = axes[idx]
        method_data = df[df["method"] == method]

        # Plot MEA metrics
        for metric in metrics:
            mea_data = method_data[
                (method_data["Metric"] == metric) & (method_data["Aligner"] == "mea")
            ].sort_values("gamma")

            if not mea_data.empty:
                ax.plot(
                    mea_data["gamma"],
                    mea_data["Mean"],
                    label=f"MEA {metric}",
                    color=METHOD_COLORS.get(method, "#2E86AB"),
                    linestyle=line_styles[metric],
                    linewidth=2,
                    marker="o",
                    markersize=4,
                )

            # Plot Viterbi baseline for this metric
            vit_data = method_data[
                (method_data["Metric"] == metric)
                & (method_data["Aligner"] == "viterbi")
            ]
            if not vit_data.empty:
                vit_mean = vit_data["Mean"].mean()
                ax.axhline(
                    y=vit_mean,
                    color=VITERBI_COLOR,
                    linestyle=line_styles[metric],
                    linewidth=1.5,
                    alpha=0.7,
                    label=f"Viterbi {metric}",
                )

        ax.set_xlabel("γ (gamma)", fontsize=11)
        ax.set_ylabel("Metric Value", fontsize=11)
        ax.set_title(f"MEA Method: {method}", fontsize=12, fontweight="bold")
        ax.legend(loc="best", framealpha=0.9, fontsize=9)
        ax.grid(True, alpha=PLOT_GRID_ALPHA)

    # Hide unused subplots
    for idx in range(n_methods, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(
        "Precision, Recall, and F1 vs Gamma by MEA Method",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    output_path = output_dir / "f1_precision_recall_vs_gamma.png"
    plt.savefig(output_path, dpi=PLOT_DPI, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def main():
    """Main function to plot metrics."""
    # Create output directory
    METRICS_FIGURES_FOLDER.mkdir(parents=True, exist_ok=True)

    # Load all metrics
    print("Loading metrics from CSV files...")
    df = load_all_metrics(EVALUATION_METRICS_FOLDER)

    methods_found = df["method"].nunique()
    gammas_found = df["gamma"].nunique()
    print(
        f"Loaded {len(df)} rows from {methods_found} methods × {gammas_found} gamma values"
    )

    # Generate plots
    print("\nGenerating plots...")
    plot_column_identity_vs_gamma(df, METRICS_FIGURES_FOLDER)
    plot_delta_f1_vs_gamma(df, METRICS_FIGURES_FOLDER)
    plot_f1_precision_recall_vs_gamma(df, METRICS_FIGURES_FOLDER)

    print("\nDone! All plots saved to:", METRICS_FIGURES_FOLDER)


if __name__ == "__main__":
    main()

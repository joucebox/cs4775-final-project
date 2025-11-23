#!/usr/bin/env python3
"""Generate plots from gamma metric CSV files."""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import re


def load_all_metrics(metrics_dir: Path) -> pd.DataFrame:
    """Load all gamma_*.csv files and combine into single dataframe."""
    dfs = []
    for csv_file in sorted(metrics_dir.glob("gamma_*.csv")):
        # Extract gamma value from filename
        match = re.search(r"gamma_(\d+\.?\d*)", csv_file.name)
        if not match:
            continue
        gamma = float(match.group(1))

        # Load CSV and add gamma column
        df = pd.read_csv(csv_file)
        df["gamma"] = gamma
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


def plot_metrics_vs_gamma(df: pd.DataFrame, output_dir: Path):
    """Plot precision, recall, and F1 vs gamma for MEA and Viterbi."""
    metrics = ["precision", "recall", "f1"]

    _, ax = plt.subplots(figsize=(10, 6))

    colors = {"mea": "#2E86AB", "viterbi": "#A23B72"}
    line_styles = {"precision": "-", "recall": "--", "f1": "-."}

    for metric in metrics:
        for aligner in ["mea", "viterbi"]:
            data = df[(df["Metric"] == metric) & (df["Aligner"] == aligner)]
            data = data.sort_values("gamma")

            ax.plot(
                data["gamma"],
                data["Mean"],
                label=f"{aligner.upper()} {metric}",
                color=colors[aligner],
                linestyle=line_styles[metric],
                linewidth=2,
                marker="o",
                markersize=4,
            )

    ax.set_xlabel("γ (gamma)", fontsize=12)
    ax.set_ylabel("Metric Value", fontsize=12)
    ax.set_title("Precision, Recall, and F1 vs Gamma", fontsize=14, fontweight="bold")
    ax.legend(loc="best", framealpha=0.9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / "f1_precision_recall_vs_gamma.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def plot_delta_f1_vs_gamma(df: pd.DataFrame, output_dir: Path):
    """Plot MEA F1 - Viterbi F1 vs gamma."""
    f1_data = df[df["Metric"] == "f1"].copy()

    # Pivot to get MEA and Viterbi side by side
    pivot = f1_data.pivot_table(index="gamma", columns="Aligner", values="Mean")
    pivot["delta"] = pivot["mea"] - pivot["viterbi"]
    pivot = pivot.sort_index()

    _, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        pivot.index,
        pivot["delta"],
        color="#E63946",
        linewidth=2.5,
        marker="o",
        markersize=6,
    )
    ax.axhline(y=0, color="black", linestyle="--", linewidth=1, alpha=0.5)

    ax.set_xlabel("γ (gamma)", fontsize=12)
    ax.set_ylabel("ΔF1 (MEA - Viterbi)", fontsize=12)
    ax.set_title("MEA vs Viterbi F1 Difference", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / "delta_f1_vs_gamma.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def plot_column_identity_vs_gamma(df: pd.DataFrame, output_dir: Path):
    """Plot column identity vs gamma for MEA and Viterbi."""
    ci_data = df[df["Metric"] == "column_identity"]

    _, ax = plt.subplots(figsize=(10, 6))

    colors = {"mea": "#2E86AB", "viterbi": "#A23B72"}

    for aligner in ["mea", "viterbi"]:
        data = ci_data[ci_data["Aligner"] == aligner].sort_values("gamma")
        ax.plot(
            data["gamma"],
            data["Mean"],
            label=aligner.upper(),
            color=colors[aligner],
            linewidth=2.5,
            marker="o",
            markersize=6,
        )

    ax.set_xlabel("γ (gamma)", fontsize=12)
    ax.set_ylabel("Mean Column Identity", fontsize=12)
    ax.set_title("Column Identity vs Gamma", fontsize=14, fontweight="bold")
    ax.legend(loc="best", framealpha=0.9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / "column_identity_vs_gamma.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def main():
    # Setup paths
    project_root = Path(__file__).parent.parent
    metrics_dir = project_root / "results" / "metrics"
    output_dir = project_root / "results" / "figures"

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load all metrics
    print("Loading metrics from CSV files...")
    df = load_all_metrics(metrics_dir)
    print(f"Loaded {len(df)} rows from {df['gamma'].nunique()} gamma values")

    # Generate plots
    print("\nGenerating plots...")
    plot_metrics_vs_gamma(df, output_dir)
    plot_delta_f1_vs_gamma(df, output_dir)
    plot_column_identity_vs_gamma(df, output_dir)

    print("\nDone! All plots saved to:", output_dir)


if __name__ == "__main__":
    main()

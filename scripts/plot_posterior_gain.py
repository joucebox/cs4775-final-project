#!/usr/bin/env python3
"""Compute and visualize posterior mass differences between MEA and Viterbi alignments."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Ensure repo importability
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.algorithms.hmm import PairHMM
from src.types import Alignment
from src.utils import collect_alignments, load_pair_hmm, PosteriorCache
from src.utils.posterior_analysis import compute_posterior_metrics

from scripts.constants import (
    ALIGNMENTS_FOLDER,
    ALIGNER_COLORS,
    CACHE_FOLDER,
    GAMMA_VALUES,
    HMM_YAML,
    PLOT_DPI,
    PLOT_GRID_ALPHA,
    PLOT_TITLE_FONTSIZE,
    PLOT_XLABEL_FONTSIZE,
    PLOT_YLABEL_FONTSIZE,
    POSTERIOR_GAIN_FOLDER,
)


def compute_all_metrics(
    alignments: List[Alignment],
    hmm: PairHMM,
    gamma_values: List[float],
    cache: PosteriorCache,
) -> pd.DataFrame:
    """Compute per-pair posterior metrics for all gamma values.

    Args:
        alignments: List of reference alignments.
        hmm: The pair HMM model.
        gamma_values: List of gamma values for MEA.
        cache: Cache for posteriors and Viterbi pairs.

    Returns:
        DataFrame with columns: family, pair_id, gamma, seq_len_x, seq_len_y,
        and all metrics from compute_posterior_metrics.
    """
    rows: List[dict] = []

    print(f"Computing metrics for {len(alignments)} alignments...")

    for idx, ref in enumerate(alignments):
        if ref.num_sequences != 2:
            continue

        family = ref.name.split("_")[0] if ref.name else f"pair_{idx}"
        pair_id = ref.name if ref.name else f"pair_{idx}"

        seq_x, seq_y = ref.original_sequences

        # Get posteriors and Viterbi pairs (from cache or compute)
        post_M = cache.get_or_compute_posteriors(pair_id, hmm, seq_x, seq_y)
        vit_pairs = cache.get_or_compute_viterbi(pair_id, hmm, seq_x, seq_y)

        for gamma in gamma_values:
            # Get MEA pairs (from cache or compute)
            mea_pairs = cache.get_or_compute_mea(pair_id, gamma, hmm, seq_x, seq_y)

            # Compute metrics
            metrics = compute_posterior_metrics(post_M, mea_pairs, vit_pairs)

            row = {
                "family": family,
                "pair_id": pair_id,
                "gamma": gamma,
                "seq_len_x": len(seq_x),
                "seq_len_y": len(seq_y),
                **metrics,
            }
            rows.append(row)

        if (idx + 1) % 20 == 0:
            print(f"  Processed {idx + 1}/{len(alignments)} alignments")

    print(f"Computed metrics for {len(rows)} alignment-gamma combinations.")
    return pd.DataFrame(rows)


def plot_multi_panel(df: pd.DataFrame, output_dir: Path) -> None:
    """Create 2x2 multi-panel figure showing posterior gain analysis.

    Top-left: Mean posterior mass captured vs gamma (MEA and Viterbi lines with std)
    Top-right: Delta posterior mass vs gamma
    Bottom-left: Stacked bar chart at key gammas
    Bottom-right: Violin plot of delta distribution at gamma=1.0
    """
    _, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Aggregate by gamma
    agg = df.groupby("gamma").agg(
        {
            "mass_mea": ["mean", "std"],
            "mass_vit": ["mean", "std"],
            "delta_mass": ["mean", "std"],
            "mass_shared": "mean",
            "mass_mea_only": "mean",
            "mass_vit_only": "mean",
        }
    )
    agg.columns = ["_".join(col).strip("_") for col in agg.columns]
    agg = agg.reset_index().sort_values("gamma")

    # Top-left: Mean posterior mass captured vs gamma
    ax = axes[0, 0]
    gammas = agg["gamma"].values

    ax.errorbar(
        gammas,
        agg["mass_mea_mean"],
        yerr=agg["mass_mea_std"],
        label="MEA",
        color=ALIGNER_COLORS["mea"],
        linewidth=2,
        marker="o",
        markersize=5,
        capsize=3,
    )
    ax.errorbar(
        gammas,
        agg["mass_vit_mean"],
        yerr=agg["mass_vit_std"],
        label="Viterbi",
        color=ALIGNER_COLORS["viterbi"],
        linewidth=2,
        marker="s",
        markersize=5,
        capsize=3,
    )
    ax.set_xlabel("γ (gamma)", fontsize=PLOT_XLABEL_FONTSIZE)
    ax.set_ylabel("Mean Posterior Mass Captured", fontsize=PLOT_YLABEL_FONTSIZE)
    ax.set_title(
        "Posterior Mass Captured vs Gamma",
        fontsize=PLOT_TITLE_FONTSIZE,
        fontweight="bold",
    )
    ax.legend(loc="best", framealpha=0.9)
    ax.grid(True, alpha=PLOT_GRID_ALPHA)

    # Top-right: Delta posterior mass vs gamma
    ax = axes[0, 1]
    ax.errorbar(
        gammas,
        agg["delta_mass_mean"],
        yerr=agg["delta_mass_std"],
        color="#E63946",
        linewidth=2.5,
        marker="o",
        markersize=6,
        capsize=3,
    )
    ax.axhline(y=0, color="black", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_xlabel("γ (gamma)", fontsize=PLOT_XLABEL_FONTSIZE)
    ax.set_ylabel("Δ Posterior Mass (MEA - Viterbi)", fontsize=PLOT_YLABEL_FONTSIZE)
    ax.set_title(
        "Posterior Mass Gain: MEA vs Viterbi",
        fontsize=PLOT_TITLE_FONTSIZE,
        fontweight="bold",
    )
    ax.grid(True, alpha=PLOT_GRID_ALPHA)

    # Bottom-left: Stacked bar chart at key gammas
    ax = axes[1, 0]
    key_gammas = [0.1, 0.5, 1.0]
    key_data = agg[agg["gamma"].isin(key_gammas)].copy()
    key_data = key_data.sort_values("gamma")

    x_pos = np.arange(len(key_data))
    width = 0.6

    # Stack: shared (bottom), mea_only (middle), vit_only (top)
    ax.bar(
        x_pos,
        key_data["mass_shared_mean"],
        width,
        label="Shared",
        color=ALIGNER_COLORS["shared"],
    )
    ax.bar(
        x_pos,
        key_data["mass_mea_only_mean"],
        width,
        bottom=key_data["mass_shared_mean"],
        label="MEA-only",
        color=ALIGNER_COLORS["mea"],
    )
    ax.bar(
        x_pos,
        key_data["mass_vit_only_mean"],
        width,
        bottom=key_data["mass_shared_mean"] + key_data["mass_mea_only_mean"],
        label="Viterbi-only",
        color=ALIGNER_COLORS["viterbi"],
    )

    ax.set_xlabel("γ (gamma)", fontsize=PLOT_XLABEL_FONTSIZE)
    ax.set_ylabel("Mean Posterior Mass", fontsize=PLOT_YLABEL_FONTSIZE)
    ax.set_title(
        "Posterior Mass Partitioning", fontsize=PLOT_TITLE_FONTSIZE, fontweight="bold"
    )
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"γ={g}" for g in key_data["gamma"]])
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(True, alpha=PLOT_GRID_ALPHA, axis="y")

    # Bottom-right: Violin plot of delta distribution at gamma=1.0
    ax = axes[1, 1]
    gamma_1_data = df[df["gamma"] == 1.0]["delta_mass"].dropna()

    if len(gamma_1_data) > 0:
        parts = ax.violinplot(
            [gamma_1_data], positions=[0], showmeans=True, showmedians=True
        )
        for pc in parts["bodies"]:
            pc.set_facecolor("#E63946")
            pc.set_alpha(0.7)
        parts["cmeans"].set_color("black")
        parts["cmedians"].set_color("blue")

        # Add individual points with jitter
        jitter = np.random.normal(0, 0.04, len(gamma_1_data))
        ax.scatter(
            jitter,
            gamma_1_data,
            alpha=0.3,
            s=10,
            color="#E63946",
            zorder=1,
        )

    ax.axhline(y=0, color="black", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_xlim(-0.5, 0.5)
    ax.set_xticks([])
    ax.set_ylabel("Δ Posterior Mass (MEA - Viterbi)", fontsize=PLOT_YLABEL_FONTSIZE)
    ax.set_title(
        "Distribution of Δ Mass at γ=1.0",
        fontsize=PLOT_TITLE_FONTSIZE,
        fontweight="bold",
    )
    ax.grid(True, alpha=PLOT_GRID_ALPHA, axis="y")

    plt.tight_layout()
    output_path = output_dir / "multi_panel.png"
    plt.savefig(output_path, dpi=PLOT_DPI, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def plot_family_heatmap(df: pd.DataFrame, output_dir: Path) -> None:
    """Create heatmap: families (rows) x gamma (cols), color by delta_mass_pct."""
    # Aggregate by family and gamma
    pivot = df.pivot_table(
        index="family",
        columns="gamma",
        values="delta_mass_pct",
        aggfunc="mean",
    )

    # Sort families by mean delta across gammas
    pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]

    fig, ax = plt.subplots(figsize=(12, max(6, len(pivot) * 0.3)))

    # Use diverging colormap centered at 0
    vmax = max(abs(pivot.min().min()), abs(pivot.max().max()))
    im = ax.imshow(
        pivot.values,
        aspect="auto",
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
    )

    # Labels
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{g:.2f}" for g in pivot.columns], rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=8)

    ax.set_xlabel("γ (gamma)", fontsize=PLOT_XLABEL_FONTSIZE)
    ax.set_ylabel("Rfam Family", fontsize=PLOT_YLABEL_FONTSIZE)
    ax.set_title(
        "Posterior Mass Gain (%) by Family and Gamma",
        fontsize=PLOT_TITLE_FONTSIZE,
        fontweight="bold",
    )

    cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    cbar.set_label("Δ Mass % (MEA - Viterbi)", fontsize=10)

    plt.tight_layout()
    output_path = output_dir / "family_heatmap.png"
    plt.savefig(output_path, dpi=PLOT_DPI, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def plot_efficiency_scatter(df: pd.DataFrame, output_dir: Path) -> None:
    """Create scatter plot: x=num_pairs, y=efficiency, color by aligner.

    Facet by gamma (show 3-4 key gammas in subplots).
    """
    key_gammas = [0.1, 0.5, 0.75, 1.0]
    key_gammas = [g for g in key_gammas if g in df["gamma"].unique()]

    n_cols = len(key_gammas)
    if n_cols == 0:
        print("No key gamma values found for efficiency scatter plot.")
        return

    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 5), sharey=True)
    _ = fig  # silence unused warning
    if n_cols == 1:
        axes = [axes]

    for ax, gamma in zip(axes, key_gammas):
        gamma_df = df[df["gamma"] == gamma].copy()

        # Plot MEA efficiency
        ax.scatter(
            gamma_df["num_mea"],
            gamma_df["efficiency_mea"],
            alpha=0.5,
            s=20,
            c=ALIGNER_COLORS["mea"],
            label="MEA",
        )
        # Plot Viterbi efficiency
        ax.scatter(
            gamma_df["num_vit"],
            gamma_df["efficiency_vit"],
            alpha=0.5,
            s=20,
            c=ALIGNER_COLORS["viterbi"],
            label="Viterbi",
            marker="x",
        )

        ax.set_xlabel("Number of Aligned Pairs", fontsize=PLOT_XLABEL_FONTSIZE)
        ax.set_title(f"γ = {gamma}", fontsize=PLOT_TITLE_FONTSIZE, fontweight="bold")
        ax.grid(True, alpha=PLOT_GRID_ALPHA)

        if ax == axes[0]:
            ax.set_ylabel("Efficiency (Mass / Pairs)", fontsize=PLOT_YLABEL_FONTSIZE)
            ax.legend(loc="upper right", framealpha=0.9)

    plt.suptitle(
        "Posterior Efficiency: Mass per Aligned Pair",
        fontsize=PLOT_TITLE_FONTSIZE + 2,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    output_path = output_dir / "efficiency_scatter.png"
    plt.savefig(output_path, dpi=PLOT_DPI, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def main() -> None:
    """Main function to compute metrics and generate all posterior gain plots."""
    # Create output directory
    POSTERIOR_GAIN_FOLDER.mkdir(parents=True, exist_ok=True)

    # Validate inputs
    if not ALIGNMENTS_FOLDER.exists():
        raise FileNotFoundError(f"Alignment directory not found: {ALIGNMENTS_FOLDER}")
    if not HMM_YAML.exists():
        raise FileNotFoundError(f"HMM parameter file not found: {HMM_YAML}")

    # Load alignments and HMM
    print("Loading alignments...")
    alignments = list(collect_alignments(str(ALIGNMENTS_FOLDER)))
    alignments = sorted(alignments, key=lambda a: a.name or "")
    print(f"Loaded {len(alignments)} alignments.")

    print("Loading HMM...")
    hmm = load_pair_hmm(HMM_YAML)

    # Initialize cache
    cache = PosteriorCache(CACHE_FOLDER)
    print(f"Using cache at: {CACHE_FOLDER}")

    # Compute all metrics
    df = compute_all_metrics(alignments, hmm, GAMMA_VALUES, cache)

    # Save metrics CSV for debugging/analysis
    csv_path = POSTERIOR_GAIN_FOLDER / "metrics_per_pair.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved metrics to: {csv_path}")

    # Generate all plots
    print("\nGenerating plots...")
    plot_multi_panel(df, POSTERIOR_GAIN_FOLDER)
    plot_family_heatmap(df, POSTERIOR_GAIN_FOLDER)
    plot_efficiency_scatter(df, POSTERIOR_GAIN_FOLDER)

    print(f"\nDone! All plots saved to: {POSTERIOR_GAIN_FOLDER}")


if __name__ == "__main__":
    main()

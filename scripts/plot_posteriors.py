#!/usr/bin/env python3
"""Generate posterior match-probability heatmaps for pairwise alignments."""

from __future__ import annotations

import argparse
import io
import math
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
from PIL import Image

# Ensure repo importability
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils import load_pair_hmm, PosteriorCache
from src.utils.posterior_analysis import extract_alignment_pairs
from src.utils.stockholm import read_rna_stockholm

from scripts.constants import (
    ALIGNMENTS_FOLDER,
    CACHE_FOLDER,
    HMM_YAML,
    POSTERIORS_FOLDER,
)


def plot_posterior_heatmap(
    post_M: np.ndarray,
    ref_pairs: set,
    out_path: Path,
    gamma: float | None = None,
    dpi: int = 96,
    fmt: str = "png",
):
    arr = post_M.copy()
    arr = arr[1:, 1:]
    if gamma is not None:
        arr = np.power(arr, float(gamma))

    plt.figure(figsize=(6, 5))
    im = plt.imshow(arr.T, origin="lower", aspect="auto", cmap="viridis")
    plt.colorbar(im, label="Posterior(M)" + (f" ^{gamma}" if gamma is not None else ""))
    plt.xlabel("i (position in seq X)")
    plt.ylabel("j (position in seq Y)")
    plt.title(out_path.stem)

    if ref_pairs:
        xs = [i for i, j in ref_pairs]
        ys = [j for i, j in ref_pairs]
        plt.scatter(xs, ys, c="red", s=6, marker="o", label="reference")
        plt.legend(loc="upper right")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path = out_path.with_suffix(f".{fmt}")
    if fmt in ("jpg", "jpeg"):
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=dpi)
        buf.seek(0)
        if Image is not None:
            img = Image.open(buf).convert("RGB")
            img.save(out_path, format="JPEG", quality=85, optimize=True)
        else:
            png_path = out_path.with_suffix(".png")
            buf.seek(0)
            with open(png_path, "wb") as fh:
                fh.write(buf.read())
    else:
        plt.savefig(out_path, dpi=dpi)
    plt.close()


def plot_overlay(
    post_M: np.ndarray,
    mea_pairs: set,
    viterbi_pairs: set,
    out_path: Path,
    gamma: float | None = None,
    dpi: int = 96,
    fmt: str = "png",
):
    """Overlay MEA posterior heatmap with MEA and Viterbi alignment markers on the same axes."""
    arr = post_M.copy()
    arr = arr[1:, 1:]
    mea_grid = arr.copy()
    if gamma is not None:
        mea_grid = np.power(mea_grid, float(gamma))

    fig, ax = plt.subplots(1, 1, figsize=(7, 6))
    im = ax.imshow(mea_grid.T, origin="lower", aspect="auto", cmap="viridis")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(out_path.stem + (f" (gamma={gamma})" if gamma is not None else ""))
    ax.set_xlabel("i (position in seq X)")
    ax.set_ylabel("j (position in seq Y)")

    # Plot MEA alignment pairs (white circles) and Viterbi pairs (black x)
    if mea_pairs:
        xs = [i for i, j in mea_pairs]
        ys = [j for i, j in mea_pairs]
        ax.scatter(xs, ys, c="white", edgecolors="black", s=18, marker="o", label="MEA")
    if viterbi_pairs:
        xs = [i for i, j in viterbi_pairs]
        ys = [j for i, j in viterbi_pairs]
        ax.scatter(xs, ys, c="black", s=18, marker="x", label="Viterbi")

    ax.legend(loc="upper right")
    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path = out_path.with_suffix(f".{fmt}")
    if fmt in ("jpg", "jpeg"):
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=dpi)
        buf.seek(0)
        if Image is not None:
            img = Image.open(buf).convert("RGB")
            img.save(out_path, format="JPEG", quality=85, optimize=True)
        else:
            png_path = out_path.with_suffix(".png")
            buf.seek(0)
            with open(png_path, "wb") as fh:
                fh.write(buf.read())
    else:
        plt.savefig(out_path, dpi=dpi)
    plt.close()


def compute_pair_metrics(
    post_M: np.ndarray, mea_pairs: set, viterbi_pairs: set, ref_pairs: set | None = None
):
    """Compute overlap, Jaccard, mean-posteriors, and posterior-mass metrics for a single pair.

    Returns dict with values useful for reporting.
    """
    import math
    from statistics import mean

    def _vals(pairs):
        vals = []
        for i, j in pairs:
            if 0 <= i < post_M.shape[0] - 1 and 0 <= j < post_M.shape[1] - 1:
                vals.append(float(post_M[i + 1, j + 1]))
        return vals

    mea_vals = _vals(mea_pairs)
    vit_vals = _vals(viterbi_pairs)

    n_mea = len(mea_pairs)
    n_vit = len(viterbi_pairs)
    inter = mea_pairs & viterbi_pairs
    union = mea_pairs | viterbi_pairs
    n_inter = len(inter)
    n_union = len(union)

    overlap_mea = n_inter / n_mea if n_mea else float("nan")
    overlap_vit = n_inter / n_vit if n_vit else float("nan")
    jaccard = n_inter / n_union if n_union else float("nan")

    mean_post_mea = mean(mea_vals) if mea_vals else float("nan")
    mean_post_vit = mean(vit_vals) if vit_vals else float("nan")
    delta_mean = (
        mean_post_mea - mean_post_vit
        if (not math.isnan(mean_post_mea) and not math.isnan(mean_post_vit))
        else float("nan")
    )

    total_mass = float(np.sum(post_M[1:, 1:]))
    mass_mea = float(
        np.sum(
            [
                post_M[i + 1, j + 1]
                for i, j in mea_pairs
                if 0 <= i < post_M.shape[0] - 1 and 0 <= j < post_M.shape[1] - 1
            ]
        )
    )
    mass_vit = float(
        np.sum(
            [
                post_M[i + 1, j + 1]
                for i, j in viterbi_pairs
                if 0 <= i < post_M.shape[0] - 1 and 0 <= j < post_M.shape[1] - 1
            ]
        )
    )
    frac_mass_mea = mass_mea / total_mass if total_mass > 0 else float("nan")
    frac_mass_vit = mass_vit / total_mass if total_mass > 0 else float("nan")

    out = {
        "n_mea": n_mea,
        "n_vit": n_vit,
        "n_inter": n_inter,
        "overlap_mea": overlap_mea,
        "overlap_vit": overlap_vit,
        "jaccard": jaccard,
        "mean_post_mea": mean_post_mea,
        "mean_post_vit": mean_post_vit,
        "delta_mean": delta_mean,
        "mass_mea": mass_mea,
        "mass_vit": mass_vit,
        "frac_mass_mea": frac_mass_mea,
        "frac_mass_vit": frac_mass_vit,
    }

    if ref_pairs is not None:

        def _score(pred):
            tp = len(pred & ref_pairs)
            prec = tp / len(pred) if len(pred) > 0 else float("nan")
            rec = tp / len(ref_pairs) if len(ref_pairs) > 0 else float("nan")
            f1 = (
                2 * prec * rec / (prec + rec)
                if (not math.isnan(prec) and not math.isnan(rec)) and (prec + rec) > 0
                else float("nan")
            )
            return {"tp": tp, "precision": prec, "recall": rec, "f1": f1}

        out["mea_vs_ref"] = _score(mea_pairs)
        out["vit_vs_ref"] = _score(viterbi_pairs)

    return out


def plot_diff_and_zoom(
    post_M: np.ndarray,
    mea_pairs: set,
    viterbi_pairs: set,
    out_path: Path,
    gamma: float | None = None,
    dpi: int = 96,
    fmt: str = "png",
    zoom_windows: int = 3,
    window_size: int = 20,
):
    """Produce a figure with an overlay and zoomed panels centered on top MEA-only posterior locations.

    This function finds positions where MEA selected a match but Viterbi did not, ranks them by posterior
    probability, and displays up to `zoom_windows` zoom panels to the right of a large overlay. It does
    not display a separate difference heatmap — the difference is only used to pick zoom centers.
    """
    arr = post_M.copy()[1:, 1:]
    if gamma is not None:
        arr = np.power(arr, float(gamma))

    n, m = arr.shape

    # Find MEA-only positions (MEA picked, Viterbi did not) and score by posterior
    mea_only = mea_pairs - viterbi_pairs
    scored_candidates = []
    for i, j in mea_only:
        if 0 <= i < n and 0 <= j < m:
            scored_candidates.append((arr[i, j], i, j))
    # Sort descending by posterior value (favor MEA the most)
    scored_candidates.sort(reverse=True)

    hs = window_size // 2
    max_overlap = (window_size * window_size) // 2  # allow up to half region overlap

    def compute_overlap_area(ci1, cj1, ci2, cj2):
        """Compute overlap area between two windows centered at given coordinates."""
        # Window 1 bounds (clipped to array)
        i0_1, i1_1 = max(0, ci1 - hs), min(n, ci1 + hs)
        j0_1, j1_1 = max(0, cj1 - hs), min(m, cj1 + hs)
        # Window 2 bounds (clipped to array)
        i0_2, i1_2 = max(0, ci2 - hs), min(n, ci2 + hs)
        j0_2, j1_2 = max(0, cj2 - hs), min(m, cj2 + hs)
        # Intersection
        overlap_i = max(0, min(i1_1, i1_2) - max(i0_1, i0_2))
        overlap_j = max(0, min(j1_1, j1_2) - max(j0_1, j0_2))
        return overlap_i * overlap_j

    def has_acceptable_overlap(ci, cj, centers):
        """Check if candidate has acceptable overlap with all existing centers."""
        for cx, cy in centers:
            if compute_overlap_area(ci, cj, cx, cy) >= max_overlap:
                return False
        return True

    # Greedily select centers: highest posterior first, allow if acceptable overlap
    centers = []
    for _, ci, cj in scored_candidates:
        if len(centers) >= zoom_windows:
            break
        if has_acceptable_overlap(ci, cj, centers):
            centers.append((ci, cj))

    # If no MEA-only centers found, skip zoom panels entirely
    num_zooms = len(centers)

    # Layout: if no zooms, just show overlay; otherwise overlay + zoom panels
    if num_zooms == 0:
        fig, ax_overlay = plt.subplots(1, 1, figsize=(8, 6))
        zoom_axes = []
    else:
        fig = plt.figure(figsize=(12, 3 * num_zooms))
        gs = GridSpec(num_zooms, 2, width_ratios=[3, 1], figure=fig)
        ax_overlay = fig.add_subplot(gs[:, 0])
        zoom_axes = [fig.add_subplot(gs[i, 1]) for i in range(num_zooms)]

    # Overlay: posterior heatmap with MEA and Viterbi markers
    im0 = ax_overlay.imshow(arr.T, origin="lower", aspect="auto", cmap="viridis")
    ax_overlay.set_title("Posterior (MEA background)")
    if mea_pairs:
        xs = [i for i, j in mea_pairs]
        ys = [j for i, j in mea_pairs]
        ax_overlay.scatter(
            xs, ys, c="white", edgecolors="black", s=10, marker="o", label="MEA"
        )
    if viterbi_pairs:
        xs = [i for i, j in viterbi_pairs]
        ys = [j for i, j in viterbi_pairs]
        ax_overlay.scatter(xs, ys, c="black", s=10, marker="x", label="Viterbi")
    ax_overlay.legend(loc="upper right")
    fig.colorbar(im0, ax=ax_overlay, fraction=0.046, pad=0.04)

    # Draw zoom windows and small subplots to the right
    for k, (ci, cj) in enumerate(centers):
        axz = zoom_axes[k]
        hs = window_size // 2
        i0 = max(0, ci - hs)
        i1 = min(n, ci + hs)
        j0 = max(0, cj - hs)
        j1 = min(m, cj + hs)

        sub = arr[i0:i1, j0:j1]
        axz.imshow(sub.T, origin="lower", aspect="auto", cmap="viridis")
        axz.set_title(f"Zoom center ({ci},{cj})")

        # Overlay markers within zoom (shift coordinates)
        mea_sub_x = [i - i0 for i, j in mea_pairs if i0 <= i < i1 and j0 <= j < j1]
        mea_sub_y = [j - j0 for i, j in mea_pairs if i0 <= i < i1 and j0 <= j < j1]
        vit_sub_x = [i - i0 for i, j in viterbi_pairs if i0 <= i < i1 and j0 <= j < j1]
        vit_sub_y = [j - j0 for i, j in viterbi_pairs if i0 <= i < i1 and j0 <= j < j1]
        if mea_sub_x:
            axz.scatter(
                mea_sub_x,
                mea_sub_y,
                c="white",
                edgecolors="black",
                s=30,
                marker="o",
                label="MEA",
            )
        if vit_sub_x:
            axz.scatter(
                vit_sub_x, vit_sub_y, c="black", s=30, marker="x", label="Viterbi"
            )
        # Only show legend if any handles exist (prevents UserWarning if none)
        handles, _ = axz.get_legend_handles_labels()
        if handles:
            axz.legend(loc="upper right")

        # Draw rectangle on overlay showing zoom region
        rect = Rectangle(
            (i0, j0),
            i1 - i0,
            j1 - j0,
            linewidth=1.0,
            edgecolor="yellow",
            facecolor="none",
        )
        ax_overlay.add_patch(rect)

    plt.suptitle(out_path.stem)
    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path = out_path.with_suffix(f".{fmt}")
    if fmt in ("jpg", "jpeg"):
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=dpi)
        buf.seek(0)
        if Image is not None:
            img = Image.open(buf).convert("RGB")
            img.save(out_path, format="JPEG", quality=85, optimize=True)
        else:
            png_path = out_path.with_suffix(".png")
            buf.seek(0)
            with open(png_path, "wb") as fh:
                fh.write(buf.read())
    else:
        plt.savefig(out_path, dpi=dpi)
    plt.close()


def plot_side_by_side(
    post_M: np.ndarray,
    mea_pairs: set,
    viterbi_pairs: set,
    out_path: Path,
    gamma: float | None = None,
    dpi: int = 96,
    fmt: str = "png",
):
    """Plot MEA-weighted posterior and Viterbi binary alignment side-by-side."""
    arr = post_M.copy()
    arr = arr[1:, 1:]
    mea_grid = arr.copy()
    if gamma is not None:
        mea_grid = np.power(mea_grid, float(gamma))

    n, m = mea_grid.shape
    viterbi_grid = np.zeros_like(mea_grid)
    for i, j in viterbi_pairs:
        if 0 <= i < n and 0 <= j < m:
            viterbi_grid[i, j] = 1.0

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    im0 = axes[0].imshow(mea_grid.T, origin="lower", aspect="auto", cmap="viridis")
    axes[0].set_title(
        "MEA posterior (post^gamma)"
        + (f" (gamma={gamma})" if gamma is not None else "")
    )
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    if mea_pairs:
        xs = [i for i, j in mea_pairs]
        ys = [j for i, j in mea_pairs]
        axes[0].scatter(xs, ys, c="white", s=6, marker="o", label="MEA alignment")
        axes[0].legend(loc="upper right")

    im1 = axes[1].imshow(viterbi_grid.T, origin="lower", aspect="auto", cmap="Reds")
    axes[1].set_title("Viterbi alignment (binary)")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    if viterbi_pairs:
        xs = [i for i, j in viterbi_pairs]
        ys = [j for i, j in viterbi_pairs]
        axes[1].scatter(xs, ys, c="black", s=6, marker="x", label="Viterbi")
        axes[1].legend(loc="upper right")

    plt.suptitle(out_path.stem)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path = out_path.with_suffix(f".{fmt}")
    if fmt in ("jpg", "jpeg"):
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=dpi)
        buf.seek(0)
        if Image is not None:
            img = Image.open(buf).convert("RGB")
            img.save(out_path, format="JPEG", quality=85, optimize=True)
        else:
            png_path = out_path.with_suffix(".png")
            buf.seek(0)
            with open(png_path, "wb") as fh:
                fh.write(buf.read())
    else:
        plt.savefig(out_path, dpi=dpi)
    plt.close()


def resample_to_grid(arr: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    """Resample 2D array `arr` (H x W) to `target_shape` using interpolation."""
    H, W = arr.shape
    tH, tW = target_shape
    if H == 0 or W == 0:
        return np.zeros((tH, tW), dtype=float)
    if H == tH and W == tW:
        return arr.copy()

    src_x = np.linspace(0, H - 1, H)
    src_y = np.linspace(0, W - 1, W)

    tgt_x = np.linspace(0, H - 1, tH)
    tgt_y = np.linspace(0, W - 1, tW)

    out = np.zeros((tH, tW), dtype=float)
    for i, xi in enumerate(tgt_x):
        x0 = int(math.floor(xi))
        x1 = min(x0 + 1, H - 1)
        wx = xi - x0
        for j, yj in enumerate(tgt_y):
            y0 = int(math.floor(yj))
            y1 = min(y0 + 1, W - 1)
            wy = yj - y0

            v00 = arr[x0, y0]
            v01 = arr[x0, y1]
            v10 = arr[x1, y0]
            v11 = arr[x1, y1]

            val = (
                v00 * (1 - wx) * (1 - wy)
                + v10 * wx * (1 - wy)
                + v01 * (1 - wx) * wy
                + v11 * wx * wy
            )
            out[i, j] = val

    return out


def plot_aggregated_heatmaps(
    grids: list[np.ndarray],
    ref_counts: np.ndarray,
    out_path: Path,
    gamma: float | None = None,
    dpi: int = 96,
    fmt: str = "png",
):
    """Plot mean and max aggregated grids side-by-side. `grids` is list of resampled arrays (same shape)."""
    if not grids:
        return
    stack = np.stack(grids, axis=0)
    mean_grid = np.mean(stack, axis=0)
    max_grid = np.max(stack, axis=0)

    if gamma is not None:
        mean_grid = np.power(mean_grid, float(gamma))
        max_grid = np.power(max_grid, float(gamma))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    im0 = axes[0].imshow(mean_grid.T, origin="lower", aspect="auto", cmap="viridis")
    axes[0].set_title("Mean posterior (resampled)")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(max_grid.T, origin="lower", aspect="auto", cmap="viridis")
    axes[1].set_title("Max posterior (resampled)")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    if ref_counts is not None:
        axes[0].contour(ref_counts.T, colors="red", linewidths=0.8, levels=3, alpha=0.7)

    plt.suptitle(out_path.stem)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path = out_path.with_suffix(f".{fmt}")
    if fmt in ("jpg", "jpeg"):
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=dpi)
        buf.seek(0)
        if Image is not None:
            img = Image.open(buf).convert("RGB")
            img.save(out_path, format="JPEG", quality=85, optimize=True)
        else:
            png_path = out_path.with_suffix(".png")
            buf.seek(0)
            with open(png_path, "wb") as fh:
                fh.write(buf.read())
    else:
        plt.savefig(out_path, dpi=dpi)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot posterior match probabilities and compare MEA vs Viterbi."
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=None,
        help="Gamma value for MEA weighting (post^gamma)",
    )
    parser.add_argument(
        "--max-count", type=int, default=0, help="Max number of outputs (0 = unlimited)"
    )
    parser.add_argument(
        "--per-pair",
        action="store_true",
        help="Produce per-pair heatmaps instead of aggregated",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Produce side-by-side MEA vs Viterbi comparison for each pair (implies --per-pair)",
    )
    parser.add_argument(
        "--overlay",
        action="store_true",
        help="Overlay MEA and Viterbi on the same chart when using --compare",
    )
    parser.add_argument(
        "--grid-size", type=int, default=48, help="Grid size for aggregated resampling"
    )
    parser.add_argument(
        "--zoom-diff",
        action="store_true",
        help="Generate difference heatmap and zoomed panels highlighting MEA vs Viterbi disagreements",
    )
    parser.add_argument("--dpi", type=int, default=192, help="Output image DPI")
    parser.add_argument("--fmt", default="png", help="Image format (png, jpg)")

    args = parser.parse_args()

    gamma = args.gamma
    max_count = int(args.max_count)
    per_pair = bool(args.per_pair) or bool(args.compare)
    compare = bool(args.compare)
    grid_size = int(args.grid_size)
    dpi = int(args.dpi)
    fmt = str(args.fmt)

    if not HMM_YAML.exists():
        raise FileNotFoundError(f"HMM yaml not found: {HMM_YAML}")

    print("Loading HMM parameters...")
    pair_hmm = load_pair_hmm(HMM_YAML)

    # Initialize cache
    cache = PosteriorCache(CACHE_FOLDER)
    print(f"Using cache at: {CACHE_FOLDER}")

    sto_files = sorted(ALIGNMENTS_FOLDER.glob("*.sto"))
    if not sto_files:
        raise FileNotFoundError(f"No Stockholm files found in {ALIGNMENTS_FOLDER}")

    POSTERIORS_FOLDER.mkdir(parents=True, exist_ok=True)

    mode = "per-pair" if per_pair else "aggregated"
    print(f"Processing {len(sto_files)} alignment files ({mode} mode)...")

    count = 0
    for file_idx, sto_path in enumerate(sto_files):
        if (file_idx + 1) % 10 == 0 or file_idx == 0:
            print(f"  [{file_idx + 1}/{len(sto_files)}] {sto_path.stem}")

        orig_bytes = sto_path.read_bytes()
        try:
            pairwise = read_rna_stockholm(str(sto_path))
        finally:
            try:
                sto_path.write_bytes(orig_bytes)
            except Exception:
                print(f"Warning: failed to restore original Stockholm file: {sto_path}")

        if not per_pair:
            grids = []
            ref_count_grid = None
            tshape = (grid_size, grid_size)
            for idx, ref in enumerate(pairwise):
                seq_x, seq_y = ref.original_sequences
                pair_id = ref.name if ref.name else f"{sto_path.stem}_{idx}"
                post = cache.get_or_compute_posteriors(pair_id, pair_hmm, seq_x, seq_y)
                arr = post[1:, 1:]
                grid = resample_to_grid(arr, tshape)
                grids.append(grid)

                if ref_count_grid is None:
                    ref_count_grid = np.zeros(tshape, dtype=float)
                ref_pairs = extract_alignment_pairs(
                    ref.aligned_sequences[0], ref.aligned_sequences[1]
                )
                n = arr.shape[0]
                m = arr.shape[1]
                if n > 0 and m > 0:
                    for i, j in ref_pairs:
                        gx = int(round(i / max(1, n - 1) * (tshape[0] - 1)))
                        gy = int(round(j / max(1, m - 1) * (tshape[1] - 1)))
                        ref_count_grid[gx, gy] += 1

            if grids:
                out_name = f"{sto_path.stem}_aggregated.png"
                out_path = POSTERIORS_FOLDER / out_name
                plot_aggregated_heatmaps(
                    grids, ref_count_grid, out_path, gamma=gamma, dpi=dpi, fmt=fmt
                )
                count += 1
                if max_count > 0 and count >= max_count:
                    print(f"Produced {count} aggregated heatmaps (limit reached).")
                    return
        else:
            for idx, ref in enumerate(pairwise):
                seq_x, seq_y = ref.original_sequences
                pair_id = ref.name if ref.name else f"{sto_path.stem}_{idx}"

                post = cache.get_or_compute_posteriors(pair_id, pair_hmm, seq_x, seq_y)
                ref_pairs = extract_alignment_pairs(
                    ref.aligned_sequences[0], ref.aligned_sequences[1]
                )

                out_name = f"{sto_path.stem}_{idx}.png"
                out_path = POSTERIORS_FOLDER / out_name

                if compare:
                    # Get MEA and Viterbi pairs from cache
                    mea_gamma = float(gamma) if gamma is not None else 1.0
                    mea_pairs = cache.get_or_compute_mea(
                        pair_id, mea_gamma, pair_hmm, seq_x, seq_y
                    )
                    viterbi_pairs = cache.get_or_compute_viterbi(
                        pair_id, pair_hmm, seq_x, seq_y
                    )

                    # If zoom-diff requested, generate difference heatmap + zooms
                    if getattr(args, "zoom_diff", False):
                        out_zoom = out_path.with_name(out_path.stem + "_diff_zoom")
                        plot_diff_and_zoom(
                            post,
                            mea_pairs,
                            viterbi_pairs,
                            out_zoom,
                            gamma=gamma,
                            dpi=dpi,
                            fmt=fmt,
                        )
                    # If overlay requested, draw both on the same chart
                    elif getattr(args, "overlay", False):
                        plot_overlay(
                            post,
                            mea_pairs,
                            viterbi_pairs,
                            out_path,
                            gamma=gamma,
                            dpi=dpi,
                            fmt=fmt,
                        )
                    else:
                        plot_side_by_side(
                            post,
                            mea_pairs,
                            viterbi_pairs,
                            out_path,
                            gamma=gamma,
                            dpi=dpi,
                            fmt=fmt,
                        )
                else:
                    plot_posterior_heatmap(
                        post, ref_pairs, out_path, gamma=gamma, dpi=dpi, fmt=fmt
                    )

                count += 1
                if max_count > 0 and count >= max_count:
                    print(f"Produced {count} heatmaps (limit reached).")
                    return

    print(f"\nDone! Produced {count} posterior heatmaps at {POSTERIORS_FOLDER}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Generate posterior match-probability heatmaps for pairwise alignments."""

from __future__ import annotations

from pathlib import Path
import sys
import math
import numpy as np
import matplotlib.pyplot as plt

# Ensure repo importability
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.stockholm import read_rna_stockholm
from src.algorithms.hmm import PairHMM
from src.algorithms.forward_backward import compute_forward, compute_backward
from src.types.parameters import (
    EmissionParameters,
    GapParameters,
    HMMParameters,
    TransitionParameters,
)
import yaml
import io
from PIL import Image
from src.algorithms.viterbi import ViterbiAligner
from src.algorithms.mea import MEAAligner
import argparse

from scripts.constants import ALIGNMENTS_FOLDER, HMM_YAML


def load_pair_hmm(yaml_path: Path) -> PairHMM:
    with yaml_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)

    params_dict = payload.get("parameters", payload)
    emissions_dict = params_dict["log_emissions"]
    transitions_dict = params_dict["log_transitions"]["matrix"]
    gaps_dict = params_dict["gaps"]

    params = HMMParameters(
        log_emissions=EmissionParameters(
            match=emissions_dict["match"],
            insert_x=emissions_dict["insert_x"],
            insert_y=emissions_dict["insert_y"],
        ),
        log_transitions=TransitionParameters(matrix=transitions_dict),
        gaps=GapParameters(**gaps_dict),
    )
    return PairHMM(params)


def build_posteriors(hmm: PairHMM, x_seq, y_seq) -> np.ndarray:
    """Compute posterior match matrix as in MEA._compute_match_posteriors."""
    if getattr(x_seq, "normalized", False) is False:
        x_seq = x_seq.normalize()
    if getattr(y_seq, "normalized", False) is False:
        y_seq = y_seq.normalize()

    F_M, _, _, logZ_f = compute_forward(hmm, x_seq, y_seq)
    B_M, _, _, logZ_b = compute_backward(hmm, x_seq, y_seq)

    if math.isfinite(logZ_f) and math.isfinite(logZ_b):
        if abs(logZ_f - logZ_b) > 1e-5:
            raise ValueError(
                f"Forward and backward log-normalizers disagree: {logZ_f} vs {logZ_b}"
            )
        logZ = 0.5 * (logZ_f + logZ_b)
    else:
        raise ValueError("Forward/backward log-normalizers are not finite")

    n = len(x_seq)
    m = len(y_seq)
    post_M = np.zeros((n + 1, m + 1), dtype=float)

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            joint_log = F_M[i][j] + B_M[i][j]
            if joint_log == float("-inf"):
                post_M[i, j] = 0.0
            else:
                post_M[i, j] = math.exp(joint_log - logZ)

    return post_M


def aligned_pairs_from_alignment(a1, a2):
    """Return set of 0-based aligned residue index pairs from two aligned sequences."""
    pairs = set()
    cur_i = 0
    cur_j = 0
    for c1, c2 in zip(a1.residues, a2.residues):
        is_res1 = c1 != "-"
        is_res2 = c2 != "-"
        if is_res1 and is_res2:
            pairs.add((cur_i, cur_j))
        if is_res1:
            cur_i += 1
        if is_res2:
            cur_j += 1
    return pairs


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


def compute_pair_metrics(post_M: np.ndarray, mea_pairs: set, viterbi_pairs: set, ref_pairs: set | None = None):
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
    delta_mean = mean_post_mea - mean_post_vit if (not math.isnan(mean_post_mea) and not math.isnan(mean_post_vit)) else float("nan")

    total_mass = float(np.sum(post_M[1:, 1:]))
    mass_mea = float(np.sum([post_M[i + 1, j + 1] for i, j in mea_pairs if 0 <= i < post_M.shape[0] - 1 and 0 <= j < post_M.shape[1] - 1]))
    mass_vit = float(np.sum([post_M[i + 1, j + 1] for i, j in viterbi_pairs if 0 <= i < post_M.shape[0] - 1 and 0 <= j < post_M.shape[1] - 1]))
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
            f1 = 2 * prec * rec / (prec + rec) if (not math.isnan(prec) and not math.isnan(rec)) and (prec + rec) > 0 else float("nan")
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

    # Build indicator masks
    mea_mask = np.zeros_like(arr, dtype=float)
    vit_mask = np.zeros_like(arr, dtype=float)
    for i, j in mea_pairs:
        if 0 <= i < n and 0 <= j < m:
            mea_mask[i, j] = 1.0
    for i, j in viterbi_pairs:
        if 0 <= i < n and 0 <= j < m:
            vit_mask[i, j] = 1.0

    # Score MEA-only locations by posterior (prefer locations MEA picked and Viterbi did not)
    score_map = arr * (mea_mask - vit_mask)
    flat = score_map.flatten()

    # Select positive-scoring centers first (MEA > VIT), fallback to top absolute scores
    idx_pos = np.argsort(flat)[::-1]
    centers = []
    for idx in idx_pos:
        if len(centers) >= zoom_windows:
            break
        val = flat[idx]
        if val <= 0:
            break
        ci = int(idx // m)
        cj = int(idx % m)
        centers.append((ci, cj))

    if not centers:
        # fallback: top absolute disagreements (could favor Viterbi in some cases)
        idx_abs = np.argsort(np.abs(flat))[::-1]
        for idx in idx_abs[:zoom_windows]:
            ci = int(idx // m)
            cj = int(idx % m)
            centers.append((ci, cj))

    # Limit number of zooms to available space
    max_zooms = max(1, min(len(centers), zoom_windows))
    centers = centers[:max_zooms]

    # If no centers identified (very short sequences), center on middle
    if not centers:
        centers = [(n // 2, m // 2)]
        max_zooms = 1

    # Layout with GridSpec: overlay on left spanning all rows, zoom panels stacked on right
    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=(12, 3 * max_zooms))
    gs = GridSpec(max_zooms, 2, width_ratios=[3, 1], figure=fig)

    ax_overlay = fig.add_subplot(gs[:, 0])
    zoom_axes = [fig.add_subplot(gs[i, 1]) for i in range(max_zooms)]

    # Overlay: MEA posterior (arr) with markers
    im0 = ax_overlay.imshow(arr.T, origin="lower", aspect="auto", cmap="viridis")
    ax_overlay.set_title("Posterior (MEA background)")
    if mea_pairs:
        xs = [i for i, j in mea_pairs]
        ys = [j for i, j in mea_pairs]
        ax_overlay.scatter(xs, ys, c="white", edgecolors="black", s=10, marker="o", label="MEA")
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

        # overlay markers within zoom (shift coordinates)
        mea_sub_x = [i - i0 for i, j in mea_pairs if i0 <= i < i1 and j0 <= j < j1]
        mea_sub_y = [j - j0 for i, j in mea_pairs if i0 <= i < i1 and j0 <= j < j1]
        vit_sub_x = [i - i0 for i, j in viterbi_pairs if i0 <= i < i1 and j0 <= j < j1]
        vit_sub_y = [j - j0 for i, j in viterbi_pairs if i0 <= i < i1 and j0 <= j < j1]
        if mea_sub_x:
            axz.scatter(mea_sub_x, mea_sub_y, c="white", edgecolors="black", s=30, marker="o", label="MEA")
        if vit_sub_x:
            axz.scatter(vit_sub_x, vit_sub_y, c="black", s=30, marker="x", label="Viterbi")
        axz.legend(loc="upper right")

        # Draw rectangle on overlay showing zoom region
        from matplotlib.patches import Rectangle

        rect = Rectangle((i0, j0), i1 - i0, j1 - j0, linewidth=1.0, edgecolor="yellow", facecolor="none")
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
    axes[0].set_title("MEA posterior (post^gamma)" + (f" (gamma={gamma})" if gamma is not None else ""))
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
    parser.add_argument("--align-dir", default=ALIGNMENTS_FOLDER, help="Directory with .sto files")
    parser.add_argument("--hmm-yaml", default=HMM_YAML, help="HMM YAML file")
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "results" / "posteriors"), help="Output directory for plots")
    parser.add_argument("--gamma", type=float, default=None, help="Gamma value for MEA weighting (post^gamma)")
    parser.add_argument("--max-count", type=int, default=0, help="Max number of outputs (0 = unlimited)")
    parser.add_argument("--per-pair", action="store_true", help="Produce per-pair heatmaps instead of aggregated")
    parser.add_argument("--compare", action="store_true", help="Produce side-by-side MEA vs Viterbi comparison for each pair (implies --per-pair)")
    parser.add_argument("--overlay", action="store_true", help="Overlay MEA and Viterbi on the same chart when using --compare")
    parser.add_argument("--grid-size", type=int, default=48, help="Grid size for aggregated resampling")
    parser.add_argument("--zoom-diff", action="store_true", help="Generate difference heatmap and zoomed panels highlighting MEA vs Viterbi disagreements")
    parser.add_argument("--dpi", type=int, default=192, help="Output image DPI")
    parser.add_argument("--fmt", default="png", help="Image format (png, jpg)")

    args = parser.parse_args()

    align_dir = Path(args.align_dir)
    hmm_yaml = Path(args.hmm_yaml)
    out_dir = Path(args.out_dir)
    gamma = args.gamma
    max_count = int(args.max_count)
    per_pair = bool(args.per_pair) or bool(args.compare)
    compare = bool(args.compare)
    grid_size = int(args.grid_size)
    dpi = int(args.dpi)
    fmt = str(args.fmt)

    if not hmm_yaml.exists():
        raise FileNotFoundError(f"HMM yaml not found: {hmm_yaml}")
    pair_hmm = load_pair_hmm(hmm_yaml)

    sto_files = sorted(align_dir.glob("*.sto"))
    if not sto_files:
        raise FileNotFoundError(f"No Stockholm files found in {align_dir}")

    count = 0
    for sto_path in sto_files:
        orig_bytes = sto_path.read_bytes()
        try:
            pairwise = read_rna_stockholm(str(sto_path))
        finally:
            try:
                sto_path.write_bytes(orig_bytes)
            except Exception:
                # If restoration fails, continue but warn
                print(f"Warning: failed to restore original Stockholm file: {sto_path}")

        if not per_pair:
            grids = []
            ref_count_grid = None
            tshape = (grid_size, grid_size)
            for idx, ref in enumerate(pairwise):
                seq_x, seq_y = ref.original_sequences
                post = build_posteriors(pair_hmm, seq_x, seq_y)
                arr = post[1:, 1:]
                grid = resample_to_grid(arr, tshape)
                grids.append(grid)

                if ref_count_grid is None:
                    ref_count_grid = np.zeros(tshape, dtype=float)
                ref_pairs = aligned_pairs_from_alignment(
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
                out_path = out_dir / out_name
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

                post = build_posteriors(pair_hmm, seq_x, seq_y)
                ref_pairs = aligned_pairs_from_alignment(
                    ref.aligned_sequences[0], ref.aligned_sequences[1]
                )

                out_name = f"{sto_path.stem}_{idx}.png"
                out_path = out_dir / out_name

                if compare:
                    # Run MEA and Viterbi aligners to get their alignments
                    mea_gamma = float(gamma) if gamma is not None else 1.0
                    mea_aligner = MEAAligner(gamma=mea_gamma)
                    viterbi_aligner = ViterbiAligner()

                    mea_result = mea_aligner.align(pair_hmm, seq_x, seq_y)
                    viterbi_result = viterbi_aligner.align(pair_hmm, seq_x, seq_y)

                    mea_pairs = aligned_pairs_from_alignment(
                        mea_result.alignment.aligned_sequences[0],
                        mea_result.alignment.aligned_sequences[1],
                    )
                    viterbi_pairs = aligned_pairs_from_alignment(
                        viterbi_result.alignment.aligned_sequences[0],
                        viterbi_result.alignment.aligned_sequences[1],
                    )

                    # If zoom-diff requested, generate difference heatmap + zooms
                    if getattr(args, "zoom_diff", False):
                        out_zoom = out_path.with_name(out_path.stem + "_diff_zoom")
                        plot_diff_and_zoom(post, mea_pairs, viterbi_pairs, out_zoom, gamma=gamma, dpi=dpi, fmt=fmt)
                    # If overlay requested, draw both on the same chart
                    elif getattr(args, "overlay", False):
                        plot_overlay(post, mea_pairs, viterbi_pairs, out_path, gamma=gamma, dpi=dpi, fmt=fmt)
                    else:
                        plot_side_by_side(post, mea_pairs, viterbi_pairs, out_path, gamma=gamma, dpi=dpi, fmt=fmt)
                else:
                    plot_posterior_heatmap(
                        post, ref_pairs, out_path, gamma=gamma, dpi=dpi, fmt=fmt
                    )

                count += 1
                if max_count > 0 and count >= max_count:
                    print(f"Produced {count} heatmaps (limit reached).")
                    return

    print(f"Produced {count} posterior heatmaps at {out_dir}")


if __name__ == "__main__":
    main()

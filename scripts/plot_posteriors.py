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
    # Use defaults instead of argparse arguments
    align_dir = ALIGNMENTS_FOLDER
    hmm_yaml = HMM_YAML
    out_dir = REPO_ROOT / "results" / "posteriors"
    gamma = None
    max_count = 0
    per_pair = False
    grid_size = 48
    dpi = 192
    fmt = "png"

    align_dir = Path(align_dir)
    hmm_yaml = Path(hmm_yaml)
    out_dir = Path(out_dir)

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

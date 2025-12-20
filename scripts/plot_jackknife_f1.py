#!/usr/bin/env python3
"""
Jackknife analysis of F1 scores by leaving one RNA family out for parameter estimation.
"""

import sys
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.constants import (
    ALIGNMENTS_FOLDER,
    RESULTS_FOLDER,
    PSEUDOCOUNT,
)
from src.utils import collect_alignments
from src.algorithms.mle_parameters import estimate_params_from_alignments
from src.algorithms.hmm import PairHMM
from src.algorithms.viterbi import ViterbiAligner
from src.algorithms.mea import MEAAligner
from src.evaluation import evaluate_all_metrics


GAMMA = 0.5
MEA_METHOD = "power"
OUTPUT_PATH = RESULTS_FOLDER / "figures" / "jackknife_f1.png"


def extract_family_from_name(alignment_name: str) -> str:
    """Extract RF family ID from alignment name (e.g., 'RF00058_0.sto' -> 'RF00058')."""
    name = alignment_name.replace(".sto", "")
    return name.split("_")[0]


def group_alignments_by_family(alignments):
    grouped = defaultdict(list)
    for aln in alignments:
        fam = extract_family_from_name(aln.name)
        grouped[fam].append(aln)
    return grouped


def evaluate_family(hmm, alignments):
    """Compute mean F1 for Viterbi and MEA on a list of reference alignments."""
    vit_align = ViterbiAligner()
    mea_align = MEAAligner(gamma=GAMMA, method=MEA_METHOD)

    vit_f1s = []
    mea_f1s = []

    for ref in alignments:
        seq_x, seq_y = ref.original_sequences

        vit_pred = vit_align.align(hmm, seq_x, seq_y).alignment
        vit_eval = evaluate_all_metrics([vit_pred], [ref])["f1"].mean
        vit_f1s.append(vit_eval)

        mea_pred = mea_align.align(hmm, seq_x, seq_y).alignment
        mea_eval = evaluate_all_metrics([mea_pred], [ref])["f1"].mean
        mea_f1s.append(mea_eval)

    return float(np.mean(vit_f1s)), float(np.mean(mea_f1s))


def main():
    print("Loading reference alignments...")
    all_alignments = collect_alignments(str(ALIGNMENTS_FOLDER))
    fam_groups = group_alignments_by_family(all_alignments)
    families = sorted(fam_groups.keys())
    print(f"Found {len(families)} families, {len(all_alignments)} total alignments")

    vit_fold_means = []
    mea_fold_means = []

    for idx, holdout_fam in enumerate(families, start=1):
        holdout_alignments = fam_groups[holdout_fam]
        train_alignments = [aln for fam, alns in fam_groups.items() if fam != holdout_fam for aln in alns]

        print(f"Fold {idx}/{len(families)}: holding out {holdout_fam} ({len(holdout_alignments)} alignments), training on {len(train_alignments)}")

        params = estimate_params_from_alignments(train_alignments, pseudocount=PSEUDOCOUNT)
        hmm = PairHMM(params)

        vit_mean, mea_mean = evaluate_family(hmm, holdout_alignments)
        vit_fold_means.append(vit_mean)
        mea_fold_means.append(mea_mean)

    _, ax = plt.subplots(figsize=(8, 6))

    positions = [1, 2]
    ax.boxplot(
        [vit_fold_means, mea_fold_means],
        positions=positions,
        tick_labels=["Viterbi", f"MEA ({MEA_METHOD}, γ={GAMMA})"],
        patch_artist=True,
        boxprops=dict(facecolor="#9ecae1"),
        medianprops=dict(color="black"),
        whiskerprops=dict(color="gray"),
        capprops=dict(color="gray"),
    )

    rng = np.random.default_rng(42)
    jitter_vit = positions[0] + (rng.random(len(vit_fold_means)) - 0.5) * 0.1
    jitter_mea = positions[1] + (rng.random(len(mea_fold_means)) - 0.5) * 0.1
    ax.scatter(jitter_vit, vit_fold_means, color="#3182bd", alpha=0.5, s=12, label="Viterbi folds")
    ax.scatter(jitter_mea, mea_fold_means, color="#e6550d", alpha=0.5, s=12, label="MEA folds")

    ax.set_ylabel("F1 Score", fontsize=12)
    ax.set_title("Jackknife F1 across families", fontsize=14, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.3)
    ax.legend(loc="best", frameon=True, fontsize=9)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
    print(f"Saved jackknife F1 plot to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

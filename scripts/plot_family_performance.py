#!/usr/bin/env python3
"""
Generate per-family F1 improvement (MEA - Viterbi) bar plot at gamma=0.5.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import os
import csv
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

from src.evaluation import evaluate_all_metrics
from src.types import Alignment
from src.utils import load_pair_hmm, PosteriorCache
from src.utils.stockholm import collect_alignments
from scripts.constants import ALIGNMENTS_FOLDER, CACHE_FOLDER, HMM_YAML


def _sorted_alignments(alignments):
    """Return alignments sorted by their name for stable ordering."""
    return sorted(alignments, key=lambda alignment: alignment.name or "")


def extract_family_from_name(alignment_name: str) -> str:
    """Extract RF family ID, stripping .sto suffixes."""
    cleaned = alignment_name.replace(".sto", "")
    if "_" in cleaned:
        return cleaned.split("_")[0]
    return cleaned


def _reconstruct_alignment_from_pairs(seq_x, seq_y, pairs, name=None):
    """Reconstruct Alignment from match pairs."""
    orig_x = seq_x
    orig_y = seq_y
    if not seq_x.aligned:
        seq_x = seq_x.normalize()
    if not seq_y.aligned:
        seq_y = seq_y.normalize()

    sorted_pairs = sorted(pairs)
    aligned_x_residues = []
    aligned_y_residues = []

    x_idx = 0
    y_idx = 0
    for i, j in sorted_pairs:
        while x_idx < i:
            aligned_x_residues.append(seq_x.residues[x_idx])
            aligned_y_residues.append("-")
            x_idx += 1
        while y_idx < j:
            aligned_x_residues.append("-")
            aligned_y_residues.append(seq_y.residues[y_idx])
            y_idx += 1
        aligned_x_residues.append(seq_x.residues[i])
        aligned_y_residues.append(seq_y.residues[j])
        x_idx = i + 1
        y_idx = j + 1

    while x_idx < len(seq_x):
        aligned_x_residues.append(seq_x.residues[x_idx])
        aligned_y_residues.append("-")
        x_idx += 1

    while y_idx < len(seq_y):
        aligned_x_residues.append("-")
        aligned_y_residues.append(seq_y.residues[y_idx])
        y_idx += 1

    aligned_seq_x = type(seq_x)(
        identifier=seq_x.identifier,
        residues=aligned_x_residues,
        aligned=True,
    )
    aligned_seq_y = type(seq_y)(
        identifier=seq_y.identifier,
        residues=aligned_y_residues,
        aligned=True,
    )

    return Alignment(
        name=name,
        aligned_sequences=[aligned_seq_x, aligned_seq_y],
        original_sequences=[orig_x, orig_y],
    )


def load_per_pair_results_from_cache():
    """
    Load per-pair F1 scores by computing Viterbi and MEA alignments via cache.
    """
    family_viterbi_f1 = defaultdict(list)
    family_mea_f1 = defaultdict(list)
    
    references = _sorted_alignments(collect_alignments(str(ALIGNMENTS_FOLDER)))
    hmm = load_pair_hmm(HMM_YAML)
    cache = PosteriorCache(CACHE_FOLDER)
    
    print(f"Computing alignments for {len(references)} pairs...")
    
    viterbi_aligns = []
    mea_aligns = []
    gamma = 0.5
    method = "power"
    
    for i, ref_alignment in enumerate(references):
        if i % 50 == 0:
            print(f"  Progress: {i}/{len(references)}")
        
        try:
            seq_x, seq_y = ref_alignment.original_sequences
            pair_id = ref_alignment.name or f"pair_{id(ref_alignment)}"
            vit_pairs = cache.get_or_compute_viterbi(pair_id, hmm, seq_x, seq_y)
            vit_align = _reconstruct_alignment_from_pairs(seq_x, seq_y, vit_pairs, pair_id)
            viterbi_aligns.append(vit_align)
            mea_pairs = cache.get_or_compute_mea(
                pair_id, gamma, hmm, seq_x, seq_y, method=method
            )
            mea_align = _reconstruct_alignment_from_pairs(seq_x, seq_y, mea_pairs, pair_id)
            mea_aligns.append(mea_align)
        except Exception as e:
            print(f"  Warning: Failed for {ref_alignment.name}: {e}")
            continue
    
    print("Evaluating F1 scores...")
    for ref_alignment, vit_align, mea_align in zip(references, viterbi_aligns, mea_aligns):
        family = extract_family_from_name(ref_alignment.name)
        
        try:
            vit_eval = evaluate_all_metrics([vit_align], [ref_alignment])
            mea_eval = evaluate_all_metrics([mea_align], [ref_alignment])
            
            vit_f1 = vit_eval.get("f1")
            mea_f1 = mea_eval.get("f1")
            
            if vit_f1 and mea_f1:
                family_viterbi_f1[family].append(vit_f1.mean)
                family_mea_f1[family].append(mea_f1.mean)
        except Exception as e:
            print(f"  Warning: Evaluation failed for {ref_alignment.name}: {e}")
    
    return dict(family_viterbi_f1), dict(family_mea_f1)


def compute_family_stats(viterbi_f1_dict, mea_f1_dict):
    """
    Compute mean F1 delta per family.
    """
    family_deltas = {}
    families = set(viterbi_f1_dict.keys()) | set(mea_f1_dict.keys())
    
    for family in families:
        vit_scores = viterbi_f1_dict.get(family, [])
        mea_scores = mea_f1_dict.get(family, [])
        
        if vit_scores and mea_scores:
            mean_vit = np.mean(vit_scores)
            mean_mea = np.mean(mea_scores)
            family_deltas[family] = mean_mea - mean_vit
    
    return family_deltas


def plot_family_performance(family_deltas, output_path="results/figures/family_performance.png"):
    """
    Create bar plot of F1 delta by family, showing top and bottom slices.
    """
    if not family_deltas:
        print("No per-family data available. Cannot generate figure.")
        return
    
    sorted_families = sorted(family_deltas.items(), key=lambda x: x[1], reverse=True)
    top_slice = sorted_families[:10]
    bottom_slice = sorted_families[-5:] if len(sorted_families) > 10 else []

    selected = top_slice + bottom_slice
    
    families = [f for f, _ in selected]
    deltas = [d for _, d in selected]
    
    _, ax = plt.subplots(figsize=(12, 6))
    
    colors = ['green' if d > 0 else 'red' for d in deltas]
    ax.bar(range(len(families)), deltas, color=colors, alpha=0.7, edgecolor='black')
    ax.set_xlabel("RNA Family", fontsize=12)
    ax.set_ylabel("F1 Improvement (MEA - Viterbi)", fontsize=12)
    ax.set_title("Family-Specific F1 Performance at γ=0.5", fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(families)))
    ax.set_xticklabels(families, rotation=45, ha='right')
    ax.axhline(y=0, color='black', linestyle='--', linewidth=0.8)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()




if __name__ == "__main__":
    print("Loading per-pair results from alignments (this may take ~1-2 minutes)...")
    vit_dict, mea_dict = load_per_pair_results_from_cache()
    
    if vit_dict and mea_dict:
        print(f"Loaded data for {len(vit_dict)} families")
        deltas = compute_family_stats(vit_dict, mea_dict)
        plot_family_performance(deltas)
    else:
        print("\nFailed to load alignment data.")
        sys.exit(1)


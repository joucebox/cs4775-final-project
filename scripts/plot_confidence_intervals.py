#!/usr/bin/env python3
"""
Generate confidence interval plot for ΔF1 (MEA - Viterbi) across γ values.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

from scripts.constants import GAMMA_VALUES, MEA_METHODS, CACHE_FOLDER, HMM_YAML, ALIGNMENTS_FOLDER
from src.utils import load_pair_hmm, PosteriorCache, collect_alignments
from src.evaluation import evaluate_all_metrics
from src.types import Alignment


def extract_family_from_name(alignment_name: str) -> str:
    """Extract RNA family ID from alignment name (e.g., 'RF00058_0.sto' -> 'RF00058')."""
    name = alignment_name.replace(".sto", "")
    return name.split("_")[0]


def _reconstruct_alignment_from_pairs(
    seq_x, seq_y, pairs: set[tuple[int, int]]
) -> Alignment:
    """Reconstruct Alignment object from set of (i,j) match pairs."""
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
        name=None,
        aligned_sequences=[aligned_seq_x, aligned_seq_y],
        original_sequences=[orig_x, orig_y],
    )


def compute_delta_f1_by_family(cache: PosteriorCache, hmm, references: list) -> dict:
    """
    Compute ΔF1 (MEA - Viterbi) for each family, gamma, and method.
    
    Returns:
        dict mapping (family, gamma, method) -> list of ΔF1 values
    """
    results = defaultdict(list)
    methods = [m for m in MEA_METHODS if m != "threshold"]
    
    for ref_alignment in references:
        family = extract_family_from_name(ref_alignment.name)
        seq_x, seq_y = ref_alignment.original_sequences
        pair_id = ref_alignment.name or f"pair_{id(ref_alignment)}"
        
        viterbi_pairs = cache.get_or_compute_viterbi(pair_id, hmm, seq_x, seq_y)
        viterbi_alignment = _reconstruct_alignment_from_pairs(seq_x, seq_y, viterbi_pairs)
        viterbi_metrics = evaluate_all_metrics([viterbi_alignment], [ref_alignment])
        viterbi_f1 = viterbi_metrics["f1"].mean
        
        for gamma in GAMMA_VALUES:
            for method in methods:
                if method == "probcons" and gamma <= 0.5:
                    continue
                if method == "log_odds" and (gamma <= 0 or gamma >= 1):
                    continue                
                mea_pairs = cache.get_or_compute_mea(pair_id, gamma, hmm, seq_x, seq_y, method=method)
                mea_alignment = _reconstruct_alignment_from_pairs(seq_x, seq_y, mea_pairs)
                mea_metrics = evaluate_all_metrics([mea_alignment], [ref_alignment])
                mea_f1 = mea_metrics["f1"].mean
                delta_f1 = mea_f1 - viterbi_f1
                results[(family, gamma, method)].append(delta_f1)
    
    return results


def stratified_bootstrap_ci(
    data_by_family: dict, 
    families: list[str], 
    n_bootstrap: int = 200, 
    confidence: float = 0.95
) -> tuple[float, float, float]:
    """
    Compute mean and confidence interval using stratified bootstrap.
    
    Args:
        data_by_family: dict mapping family -> list of values
        families: list of family IDs
        n_bootstrap: number of bootstrap iterations
        confidence: confidence level (e.g., 0.95 for 95% CI)
    
    Returns:
        (mean, lower_bound, upper_bound)
    """
    all_values = []
    for family in families:
        all_values.extend(data_by_family.get(family, []))
    
    if not all_values:
        return 0.0, 0.0, 0.0
    
    observed_mean = np.mean(all_values)
    
    bootstrap_means = []
    for _ in range(n_bootstrap):
        resampled_values = []
        for family in families:
            family_values = data_by_family.get(family, [])
            if family_values:
                resampled = np.random.choice(family_values, size=len(family_values), replace=True)
                resampled_values.extend(resampled)
        
        if resampled_values:
            bootstrap_means.append(np.mean(resampled_values))
    
    alpha = 1 - confidence
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    lower_bound = np.percentile(bootstrap_means, lower_percentile)
    upper_bound = np.percentile(bootstrap_means, upper_percentile)
    
    return observed_mean, lower_bound, upper_bound


def plot_confidence_intervals(results: dict, families: list[str], output_path: Path):
    """
    Plot ΔF1 with 95% confidence intervals for each weighting scheme.
    
    Args:
        results: dict mapping (family, gamma, method) -> list of ΔF1 values
        families: list of unique family IDs
        output_path: path to save figure
    """
    methods = [m for m in MEA_METHODS if m != "threshold"]
    print(f"Computing confidence intervals for {len(methods)} methods across {len(GAMMA_VALUES)} gamma values...")
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = {
        "power": "#2E86AB",
        "probcons": "#F18F01",
        "log_odds": "#C73E1D",
    }
    
    labels = {
        "power": "Power",
        "probcons": "ProbCons",
        "log_odds": "Log-Odds",
    }
    
    for method_idx, method in enumerate(methods):
        print(f"  Processing method {method_idx+1}/{len(methods)}: {method}")
        means = []
        lower_bounds = []
        upper_bounds = []
        
        for gamma in GAMMA_VALUES:
            data_by_family = defaultdict(list)
            for family in families:
                key = (family, gamma, method)
                if key in results:
                    data_by_family[family].extend(results[key])
            
            mean, lower, upper = stratified_bootstrap_ci(data_by_family, families)
            means.append(mean)
            lower_bounds.append(lower)
            upper_bounds.append(upper)
        
        ax.plot(GAMMA_VALUES, means, 
                color=colors[method], 
                linewidth=2, 
                label=labels[method],
                marker='o',
                markersize=4)
        
        ax.fill_between(GAMMA_VALUES, 
                        lower_bounds, 
                        upper_bounds,
                        color=colors[method],
                        alpha=0.2)
    
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5, label='Viterbi baseline')
    
    ax.set_xlabel('γ (Weighting Parameter)', fontsize=12)
    ax.set_ylabel('ΔF1 (MEA - Viterbi)', fontsize=12)
    ax.set_title('95% Confidence Intervals for F1 Improvement', fontsize=14, fontweight='bold')
    ax.legend(loc='best', frameon=True, fontsize=10)
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    
    ax.axvspan(0.3, 0.7, alpha=0.05, color='green', label='Intermediate γ (0.3-0.7)')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved confidence interval plot to {output_path}")


def main():
    print("Loading HMM parameters and reference alignments...")
    hmm = load_pair_hmm(HMM_YAML)
    references = collect_alignments(str(ALIGNMENTS_FOLDER))
    print(f"Loaded {len(references)} reference alignments")
    
    families = sorted(set(extract_family_from_name(ref.name) for ref in references))
    print(f"Found {len(families)} unique RNA families")
    
    cache = PosteriorCache(CACHE_FOLDER)
    
    print("Computing ΔF1 for all families, gammas, and methods...")
    results = compute_delta_f1_by_family(cache, hmm, references)
    
    print("Generating confidence interval plot...")
    output_path = project_root / "results" / "figures" / "ci_f1.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_confidence_intervals(results, families, output_path)
    
    print("Done!")


if __name__ == "__main__":
    main()

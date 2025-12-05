#!/usr/bin/env python3
"""Evaluate MEA and Viterbi aligners against gold standard alignments."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import pandas as pd

from .constants import (
    ALIGNMENTS_FOLDER,
    CACHE_FOLDER,
    EVALUATION_METRICS_FOLDER,
    GAMMA_VALUES,
    HMM_YAML,
    MEA_METHODS,
)

# Ensure repository modules are importable when invoked as a script
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation import evaluate_all_metrics
from src.algorithms.hmm import PairHMM
from src.types import Alignment, EvaluationResult, SequenceType
from src.utils import load_pair_hmm, PosteriorCache, MEAMethod
from src.utils.stockholm import collect_alignments


def _sorted_alignments(alignments: Iterable[Alignment]) -> List[Alignment]:
    """Return alignments sorted by their name for stable evaluation ordering."""
    return sorted(alignments, key=lambda alignment: alignment.name or "")


def _reconstruct_alignment_from_pairs(
    seq_x: SequenceType,
    seq_y: SequenceType,
    pairs: set[Tuple[int, int]],
    name: str | None = None,
) -> Alignment:
    """Reconstruct an Alignment object from match pairs and original sequences.

    Given the match pairs (positions that are aligned), reconstruct the full
    alignment by inserting gaps where positions are unmatched.

    Args:
        seq_x: First original sequence.
        seq_y: Second original sequence.
        pairs: Set of (i, j) tuples representing aligned positions (0-indexed).
        name: Optional name for the alignment.

    Returns:
        Alignment object with aligned sequences containing gaps.
    """
    # Normalize sequences first (converts T to U) - matching aligner behavior
    # aligned=True only allows A, C, G, U, and gaps
    orig_x = seq_x
    orig_y = seq_y
    if not seq_x.normalized:
        seq_x = seq_x.normalize()
    if not seq_y.normalized:
        seq_y = seq_y.normalize()

    # Sort pairs by x index, then y index for consistent processing
    sorted_pairs = sorted(pairs)

    aligned_x_residues: List[str] = []
    aligned_y_residues: List[str] = []

    x_idx = 0
    y_idx = 0

    for i, j in sorted_pairs:
        # Add any unaligned x positions before this match (gaps in y)
        while x_idx < i:
            aligned_x_residues.append(seq_x.residues[x_idx])
            aligned_y_residues.append("-")
            x_idx += 1

        # Add any unaligned y positions before this match (gaps in x)
        while y_idx < j:
            aligned_x_residues.append("-")
            aligned_y_residues.append(seq_y.residues[y_idx])
            y_idx += 1

        # Add the matched pair
        aligned_x_residues.append(seq_x.residues[i])
        aligned_y_residues.append(seq_y.residues[j])
        x_idx = i + 1
        y_idx = j + 1

    # Add remaining unaligned positions from x
    while x_idx < len(seq_x):
        aligned_x_residues.append(seq_x.residues[x_idx])
        aligned_y_residues.append("-")
        x_idx += 1

    # Add remaining unaligned positions from y
    while y_idx < len(seq_y):
        aligned_x_residues.append("-")
        aligned_y_residues.append(seq_y.residues[y_idx])
        y_idx += 1

    # Create aligned sequence objects (with aligned=True to allow gap characters)
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


def _run_aligner_cached(
    aligner_name: str,
    hmm: PairHMM,
    references: Sequence[Alignment],
    cache: PosteriorCache,
    gamma: float | None = None,
    method: MEAMethod = "power",
) -> List[Alignment]:
    """Run an aligner using cache for Viterbi and MEA pairs.

    Args:
        aligner_name: Either "viterbi" or "mea".
        hmm: The pair HMM model.
        references: List of reference alignments (for original sequences).
        cache: PosteriorCache instance for caching pairs.
        gamma: Gamma value for MEA (required if aligner_name is "mea").
        method: MEA weight function method (for MEA aligner).

    Returns:
        List of predicted Alignment objects.
    """
    predictions: List[Alignment] = []

    for reference in references:
        if reference.num_sequences != 2:
            raise ValueError(
                f"Alignment '{reference.name}' is not pairwise (contains "
                f"{reference.num_sequences} sequences)."
            )

        seq_x, seq_y = reference.original_sequences
        if not isinstance(seq_x, SequenceType) or not isinstance(seq_y, SequenceType):
            raise TypeError("Expected SequenceType instances in reference alignment.")

        pair_id = reference.name or f"pair_{id(reference)}"

        # Get pairs from cache or compute them
        if aligner_name == "viterbi":
            pairs = cache.get_or_compute_viterbi(pair_id, hmm, seq_x, seq_y)
        elif aligner_name == "mea":
            if gamma is None:
                raise ValueError("gamma is required for MEA aligner")
            pairs = cache.get_or_compute_mea(
                pair_id, gamma, hmm, seq_x, seq_y, method=method
            )
        else:
            raise ValueError(f"Unknown aligner: {aligner_name}")

        # Reconstruct alignment from cached pairs
        alignment = _reconstruct_alignment_from_pairs(seq_x, seq_y, pairs, pair_id)
        predictions.append(alignment)

    return predictions


def evaluate_method_gamma(
    method: MEAMethod,
    gamma: float,
    references: List[Alignment],
    viterbi_predictions: List[Alignment],
    hmm: PairHMM,
    cache: PosteriorCache,
) -> None:
    """Evaluate MEA (with given method) and Viterbi against gold alignments."""

    mea_predictions = _run_aligner_cached(
        "mea", hmm, references, cache, gamma=gamma, method=method
    )
    aligner_predictions = {
        "viterbi": viterbi_predictions,
        "mea": mea_predictions,
    }

    all_results: Dict[str, Dict[str, EvaluationResult]] = {}
    aggregate_rows: List[Dict[str, float | str]] = []

    for aligner_label, predictions in aligner_predictions.items():
        results = evaluate_all_metrics(predictions, references)
        all_results[aligner_label] = results

        for metric_name, evaluation in results.items():
            aggregate_rows.append(
                {
                    "Aligner": aligner_label,
                    "Metric": metric_name,
                    "Mean": evaluation.mean,
                    "Std": (
                        evaluation.std if evaluation.std is not None else float("nan")
                    ),
                    "Min": evaluation.minimum,
                    "Max": evaluation.maximum,
                    "Count": evaluation.count,
                }
            )

    aggregate_df = pd.DataFrame(aggregate_rows).drop_duplicates()
    if not aggregate_df.empty:
        aggregate_df = aggregate_df.sort_values(["Metric", "Aligner"]).reset_index(
            drop=True
        )
        print("\nAggregate metrics:")
        print(aggregate_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    csv_path = EVALUATION_METRICS_FOLDER / f"{method}_gamma_{gamma}.csv"
    aggregate_df.to_csv(csv_path, index=False)
    print(f"\nWrote {method}/gamma={gamma} metrics to {csv_path}")


def main() -> None:
    """Main function to evaluate MEA and Viterbi aligners against gold alignments."""

    EVALUATION_METRICS_FOLDER.mkdir(parents=True, exist_ok=True)

    if not ALIGNMENTS_FOLDER.exists():
        raise FileNotFoundError(f"Alignment directory not found: {ALIGNMENTS_FOLDER}")
    if not HMM_YAML.exists():
        raise FileNotFoundError(f"HMM parameter file not found: {HMM_YAML}")

    references = _sorted_alignments(collect_alignments(str(ALIGNMENTS_FOLDER)))
    hmm = load_pair_hmm(HMM_YAML)

    # Initialize cache for posteriors, Viterbi, and MEA pairs
    cache = PosteriorCache(CACHE_FOLDER)
    print(f"Using cache at: {CACHE_FOLDER}")

    # Get Viterbi predictions using cache (computed once, reused for all methods)
    print("Computing Viterbi alignments...")
    viterbi_predictions = _run_aligner_cached("viterbi", hmm, references, cache)

    # Evaluate all methods across all gamma values
    for method in MEA_METHODS:
        print(f"\n{'='*60}")
        print(f"MEA Method: {method}")
        print(f"{'='*60}")

        for gamma in GAMMA_VALUES:
            # Skip invalid gamma values for certain methods
            if method == "threshold" and gamma > 1:
                print(f"Skipping gamma={gamma} for '{method}' (requires gamma <= 1)")
                continue
            if method == "log_odds" and gamma >= 1:
                print(f"Skipping gamma={gamma} for '{method}' (requires gamma < 1)")
                continue

            print(f"\nEvaluating {method}/gamma={gamma}...")
            evaluate_method_gamma(
                method, gamma, references, viterbi_predictions, hmm, cache
            )


if __name__ == "__main__":
    main()

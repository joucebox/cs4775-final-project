#!/usr/bin/env python3
"""Evaluate MEA and Viterbi aligners against gold standard alignments."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import pandas as pd

from .constants import ALIGNMENTS_FOLDER, CSV_FOLDER, HMM_YAML, GAMMA_VALUES

# Ensure repository modules are importable when invoked as a script
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation import evaluate_all_metrics
from src.algorithms.hmm import PairHMM
from src.algorithms.mea import MEAAligner
from src.algorithms.viterbi import ViterbiAligner
from src.types import Alignment, AlignmentResult, EvaluationResult, SequenceType
from src.utils import load_pair_hmm
from src.utils.stockholm import collect_alignments


def _sorted_alignments(alignments: Iterable[Alignment]) -> List[Alignment]:
    """Return alignments sorted by their name for stable evaluation ordering."""
    return sorted(alignments, key=lambda alignment: alignment.name or "")


def _run_aligner(
    aligner_instance,
    hmm: PairHMM,
    references: Sequence[Alignment],
) -> List[Alignment]:
    """Run an aligner across the reference alignments and return predicted alignments."""
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

        result: AlignmentResult = aligner_instance.align(hmm, seq_x, seq_y)
        predictions.append(result.alignment)

    return predictions


def evaluate_gamma(
    gamma: float,
    references: List[Alignment],
    viterbi_predictions: List[Alignment],
    hmm: PairHMM,
) -> None:
    """Evaluate the MEA and Viterbi aligners against the gold alignments for a given gamma value."""

    mea_predictions = _run_aligner(MEAAligner(gamma=gamma), hmm, references)
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

    csv_path = CSV_FOLDER / f"gamma_{gamma}.csv"
    aggregate_df.to_csv(csv_path, index=False)
    print(f"\nWrote gamma={gamma} aggregate metrics to {csv_path}")


def main() -> None:
    """Main function to evaluate MEA and Viterbi aligners against gold alignments."""

    CSV_FOLDER.mkdir(parents=True, exist_ok=True)

    if not ALIGNMENTS_FOLDER.exists():
        raise FileNotFoundError(f"Alignment directory not found: {ALIGNMENTS_FOLDER}")
    if not HMM_YAML.exists():
        raise FileNotFoundError(f"HMM parameter file not found: {HMM_YAML}")

    references = _sorted_alignments(collect_alignments(str(ALIGNMENTS_FOLDER)))
    hmm = load_pair_hmm(HMM_YAML)

    viterbi_predictions = _run_aligner(ViterbiAligner(), hmm, references)

    for gamma in GAMMA_VALUES:
        evaluate_gamma(gamma, references, viterbi_predictions, hmm)


if __name__ == "__main__":
    main()

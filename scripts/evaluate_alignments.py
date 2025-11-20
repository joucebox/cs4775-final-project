#!/usr/bin/env python3
"""Evaluate MEA and Viterbi aligners against gold standard alignments."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import pandas as pd
import yaml

from .constants import ALIGNMENTS_FOLDER, CSV_PATH, HMM_YAML

# Ensure repository modules are importable when invoked as a script
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation import evaluate_all_metrics
from src.algorithms.hmm import PairHMM
from src.algorithms.mea import MEAAligner
from src.algorithms.viterbi import ViterbiAligner
from src.types import Alignment, AlignmentResult, EvaluationResult, SequenceType
from src.types.parameters import (
    EmissionParameters,
    GapParameters,
    HMMParameters,
    TransitionParameters,
)
from src.utils.stockholm import collect_alignments


def load_pair_hmm(yaml_path: Path) -> PairHMM:
    """Load PairHMM parameters from a YAML file."""
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


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate MEA and Viterbi aligners against gold alignments."
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=1.0,
        help="Gamma value for the MEA aligner (ignored if MEA not selected).",
    )
    return parser.parse_args()


def main() -> None:
    """Main function to evaluate MEA and Viterbi aligners against gold alignments."""
    args = parse_args()

    alignments_dir = ALIGNMENTS_FOLDER

    if not alignments_dir.exists():
        raise FileNotFoundError(f"Alignment directory not found: {alignments_dir}")
    if not HMM_YAML.exists():
        raise FileNotFoundError(f"HMM parameter file not found: {HMM_YAML}")

    references = _sorted_alignments(collect_alignments(str(alignments_dir)))
    hmm = load_pair_hmm(HMM_YAML)

    aligner_instances = {
        "viterbi": ViterbiAligner(),
        "mea": MEAAligner(gamma=args.gamma),
    }
    aligner_predictions: Dict[str, List[Alignment]] = {
        name: _run_aligner(instance, hmm, references)
        for name, instance in aligner_instances.items()
    }

    all_results: Dict[str, Dict[str, EvaluationResult]] = {}
    aggregate_rows: List[Dict[str, float | str]] = []

    for aligner_label, predictions in aligner_predictions.items():
        results = evaluate_all_metrics(predictions, references)
        all_results[aligner_label] = results

        per_alignment_rows: Dict[str, Dict[str, float]] = {}
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
            for metric_result in evaluation.per_alignment:
                alignment_identifier = (
                    metric_result.alignment_name
                    or f"{metric_result.sequence_ids[0]}_{metric_result.sequence_ids[1]}"
                )
                row = per_alignment_rows.setdefault(
                    alignment_identifier,
                    {
                        "sequences": "/".join(metric_result.sequence_ids),
                    },
                )
                row[metric_name] = metric_result.value

        per_alignment_df = (
            pd.DataFrame.from_dict(per_alignment_rows, orient="index")
            .rename_axis("Alignment")
            .reset_index()
            .rename(columns={"sequences": "Sequences"})
        )
        ordered_columns = ["Alignment", "Sequences", *results.keys()]
        per_alignment_df = per_alignment_df.reindex(columns=ordered_columns)

        print(f"\nPer-alignment metrics for {aligner_label}:")
        if per_alignment_df.empty:
            print("No alignments evaluated.")
        else:
            per_alignment_df.drop(columns=["Sequences"], inplace=True)
            print(
                per_alignment_df.sort_values("Alignment").to_string(
                    index=False, float_format=lambda x: f"{x:.4f}"
                )
            )

    aggregate_df = pd.DataFrame(aggregate_rows).drop_duplicates()
    if not aggregate_df.empty:
        aggregate_df = aggregate_df.sort_values(["Metric", "Aligner"]).reset_index(
            drop=True
        )
        print("\nAggregate metrics:")
        print(aggregate_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    aggregate_df.to_csv(CSV_PATH, index=False)
    print(f"\nWrote aggregate metrics to {CSV_PATH}")


if __name__ == "__main__":
    main()

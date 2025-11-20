"""Framework for evaluating predicted alignments against references."""

from __future__ import annotations

import math
from statistics import fmean, pstdev
from typing import Dict, List, Tuple

from src.evaluation.metrics import (
    MetricFunction,
    column_identity,
    f1_score,
    precision,
    recall,
    sensitivity,
    specificity,
)
from src.types import Alignment
from src.types.evaluation import EvaluationResult, MetricResult


def _resolve_metric_name(metric_fn: MetricFunction, metric_name: str | None) -> str:
    """Return a human-friendly metric name."""
    if metric_name:
        return metric_name
    func_name = getattr(metric_fn, "__name__", None)
    return func_name if func_name else "metric"


def _sequence_pair_ids(alignment: Alignment) -> Tuple[str, str]:
    """Return the identifiers of the sequences participating in the alignment."""
    if alignment.num_sequences != 2:
        raise ValueError(
            f"Expected pairwise alignment with 2 sequences, found {alignment.num_sequences}."
        )

    if alignment.original_sequences and len(alignment.original_sequences) == 2:
        return (
            alignment.original_sequences[0].identifier,
            alignment.original_sequences[1].identifier,
        )

    seq_x, seq_y = alignment.aligned_sequences
    return seq_x.identifier, seq_y.identifier


def evaluate_single(
    predicted: Alignment,
    gold: Alignment,
    metric_fn: MetricFunction,
    metric_name: str | None = None,
) -> MetricResult:
    """Evaluate a single alignment pair with the provided metric function."""
    resolved_name = _resolve_metric_name(metric_fn, metric_name)
    value = metric_fn(predicted, gold)

    sequence_ids = _sequence_pair_ids(predicted)
    alignment_name = predicted.name or gold.name

    return MetricResult(
        metric=resolved_name,
        value=value,
        alignment_name=alignment_name,
        sequence_ids=sequence_ids,
    )


def evaluate_multiple(
    predicted_alignments: List[Alignment],
    gold_alignments: List[Alignment],
    metric_fn: MetricFunction,
    metric_name: str | None = None,
) -> EvaluationResult:
    """Evaluate multiple alignments with the provided metric function."""
    if len(predicted_alignments) != len(gold_alignments):
        raise ValueError(
            "Predicted and gold alignment lists must have the same length: "
            f"{len(predicted_alignments)} vs {len(gold_alignments)}"
        )

    resolved_name = _resolve_metric_name(metric_fn, metric_name)
    per_alignment: List[MetricResult] = []

    for predicted, gold in zip(predicted_alignments, gold_alignments):
        result = evaluate_single(predicted, gold, metric_fn, resolved_name)
        per_alignment.append(result)

    values = [result.value for result in per_alignment]
    count = len(values)

    if count == 0:
        mean = math.nan
        std = None
        minimum = math.nan
        maximum = math.nan
    else:
        mean = fmean(values)
        std = pstdev(values) if count > 1 else None
        minimum = min(values)
        maximum = max(values)

    return EvaluationResult(
        metric=resolved_name,
        per_alignment=per_alignment,
        mean=mean,
        std=std,
        minimum=minimum,
        maximum=maximum,
        count=count,
    )


def evaluate_all_metrics(
    predicted_alignments: List[Alignment],
    gold_alignments: List[Alignment],
    metric_fns: Dict[str, MetricFunction] | None = None,
) -> Dict[str, EvaluationResult]:
    """Evaluate a suite of metrics and return results keyed by metric name."""
    if metric_fns is None:
        metric_fns = {
            "precision": precision,
            "recall": recall,
            "f1": f1_score,
            "sensitivity": sensitivity,
            "specificity": specificity,
            "column_identity": column_identity,
        }

    results: Dict[str, EvaluationResult] = {}
    for name, metric_fn in metric_fns.items():
        results[name] = evaluate_multiple(
            predicted_alignments, gold_alignments, metric_fn, name
        )

    return results


__all__ = ["evaluate_single", "evaluate_multiple", "evaluate_all_metrics"]

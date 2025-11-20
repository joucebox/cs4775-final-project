"""Evaluation result data structures."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class MetricResult:
    """Metric outcome for a single alignment instance."""

    metric: str
    value: float
    alignment_name: Optional[str]
    sequence_ids: Tuple[str, str]


@dataclass(frozen=True)
class EvaluationResult:
    """Aggregate evaluation results for a given metric."""

    metric: str
    per_alignment: List[MetricResult]
    mean: float
    std: Optional[float]
    minimum: float
    maximum: float
    count: int


__all__ = ["MetricResult", "EvaluationResult"]

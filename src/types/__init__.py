"""Types for the project."""

from .sequence import SequenceType, RNASequence
from .alignment import Alignment, AlignmentResult
from .evaluation import MetricResult, EvaluationResult


__all__ = [
    "SequenceType",
    "RNASequence",
    "Alignment",
    "AlignmentResult",
    "MetricResult",
    "EvaluationResult",
    "parameters",
]

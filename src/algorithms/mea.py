"""Maximum Expected Accuracy (MEA) algorithm for pairwise sequence alignment."""

from __future__ import annotations

from src.algorithms.base import PairwiseAligner
from src.algorithms.hmm import PairHMM
from src.types import AlignmentResult, SequenceType


class MEAAligner(PairwiseAligner):
    """Maximum Expected Accuracy (MEA) alignment algorithm."""

    def __init__(self, gamma: float = 1.0) -> None:
        """Initialize the MEA aligner with a configurable gamma parameter."""
        self.gamma = gamma

    def align(
        self,
        hmm: PairHMM,
        x_seq: SequenceType,
        y_seq: SequenceType,
    ) -> AlignmentResult:
        """Align two sequences by maximizing expected accuracy."""
        raise NotImplementedError


__all__ = ["MEAAligner"]

"""Viterbi alignment interfaces."""

from __future__ import annotations

from src.algorithms.base import PairwiseAligner
from src.algorithms.hmm import PairHMM
from src.types import AlignmentResult, SequenceType


class ViterbiAligner(PairwiseAligner):
    """Maximum a posteriori alignment using the Viterbi algorithm."""

    def align(
        self,
        hmm: PairHMM,
        x_seq: SequenceType,
        y_seq: SequenceType,
    ) -> AlignmentResult:
        """Compute the Viterbi alignment for the provided sequences."""
        raise NotImplementedError


__all__ = ["ViterbiAligner"]

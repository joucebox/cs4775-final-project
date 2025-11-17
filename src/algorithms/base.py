"""Shared interfaces for pairwise alignment algorithms."""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.algorithms.hmm import PairHMM
from src.types import AlignmentResult, SequenceType


class PairwiseAligner(ABC):
    """Abstract base class for pairwise alignment algorithms."""

    @abstractmethod
    def align(
        self,
        hmm: PairHMM,
        x_seq: SequenceType,
        y_seq: SequenceType,
    ) -> AlignmentResult:
        """Align two sequences using the provided HMM."""
        raise NotImplementedError


__all__ = ["PairwiseAligner"]

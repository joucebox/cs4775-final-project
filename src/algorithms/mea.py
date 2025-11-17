"""Maximum Expected Accuracy (MEA) algorithm for pairwise sequence alignment."""

from __future__ import annotations

from src.algorithms.hmm import PairHMM
from src.types.sequence import SequenceType


def compute_mea(hmm: PairHMM, x_seq: SequenceType, y_seq: SequenceType) -> float:
    """Compute the Maximum Expected Accuracy (MEA) for a pairwise sequence alignment."""
    pass


def mea_alignment(hmm: PairHMM, x_seq: SequenceType, y_seq: SequenceType) -> Alignment:
    """Compute the Maximum Expected Accuracy (MEA) alignment for a pairwise sequence alignment."""
    pass

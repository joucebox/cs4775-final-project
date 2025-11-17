"""Alignment types."""

from dataclasses import dataclass
from typing import List, Optional
import numpy as np

from .sequence import SequenceType


@dataclass(frozen=True)
class Alignment:
    """Multiple sequence alignment of sequences."""

    name: Optional[str]
    aligned_sequences: List[SequenceType]
    original_sequences: List[SequenceType]

    def __post_init__(self):
        # Validate that there are at least 2 sequences
        if self.num_sequences < 2:
            raise ValueError("At least 2 sequences are required.")

        # Validate that the number of aligned_sequences and original_sequences is the same
        if len(self.aligned_sequences) != len(self.original_sequences):
            raise ValueError(
                "aligned_sequences and original_sequences must have the same length."
            )

        # Validate that all aligned_sequences have aligned=True
        if any(s.aligned is False for s in self.aligned_sequences):
            raise ValueError("All aligned_sequences must have aligned=True.")

        # Validate that all original_sequences have aligned=False
        if any(s.aligned is True for s in self.original_sequences):
            raise ValueError("All original_sequences must have aligned=False.")

        # Validate that all aligned_sequences have the same length
        if any(len(s) != self.columns for s in self.aligned_sequences):
            raise ValueError("All aligned_sequences must have the same length.")

    @property
    def num_sequences(self) -> int:
        """Number of sequences in the alignment."""
        return len(self.aligned_sequences)

    @property
    def columns(self) -> Optional[int]:
        """Number of columns in the alignment."""
        return len(self.aligned_sequences[0])

    def __str__(self) -> str:
        class_name = self.__class__.__name__

        def _indent_seq_string(seq):
            s = str(seq)
            return "      " + s.replace("\n", "\n     ").replace("U", "T")

        aligned_str = "\n".join(
            _indent_seq_string(seq) for seq in self.aligned_sequences
        )
        original_str = "\n".join(
            _indent_seq_string(seq) for seq in self.original_sequences
        )
        return (
            f"{class_name} (\n"
            f"   name: {self.name}\n"
            f"   aligned_sequences (columns: {self.columns}):\n{aligned_str}\n"
            f"   original_sequences:\n{original_str}\n"
            f")"
        )


@dataclass(frozen=True)
class AlignmentResult:
    """Result of a pairwise alignment algorithm.

    Attributes:
        alignment: The pairwise alignment of two sequences
        score: The alignment score (log-likelihood for Viterbi, expected accuracy for MEA)
        posteriors: Optional (n+1) x (m+1) matrix of posterior match probabilities
    """

    alignment: Alignment
    score: float
    posteriors: Optional[np.ndarray] = None


__all__ = ["Alignment", "AlignmentResult"]

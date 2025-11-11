"""Alignment types."""

from dataclasses import dataclass
from typing import List, Optional

from .sequence import SequenceType


@dataclass(frozen=True)
class Alignment:
    """Multiple sequence alignment of sequences."""

    name: Optional[str]
    aligned_sequences: List[SequenceType]
    original_sequences: List[SequenceType]

    @property
    def num_sequences(self) -> int:
        """Number of sequences in the alignment."""
        return len(self.aligned_sequences)

    @property
    def columns(self) -> Optional[int]:
        """Number of columns in the alignment."""
        if not self.aligned_sequences:
            return None
        return len(self.aligned_sequences[0])


__all__ = ["Alignment"]

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
            f"   aligned_sequences:\n{aligned_str}\n"
            f"   original_sequences:\n{original_str}\n"
            f")"
        )


__all__ = ["Alignment"]

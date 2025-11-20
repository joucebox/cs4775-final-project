"""Sequence types."""

from dataclasses import dataclass
from typing import List, Optional
from abc import ABC, abstractmethod


@dataclass(frozen=True)
class SequenceType(ABC):
    """Generic biological sequence with an identifier and optional description."""

    identifier: str
    residues: List[str]
    description: Optional[str] = None
    aligned: bool = False
    normalized: bool = False

    def __post_init__(self) -> None:
        self._validate()

    def __len__(self) -> int:
        return len(self.residues)

    def __str__(self) -> str:
        class_name = self.__class__.__name__
        joined_residues = "".join(self.residues)
        return (
            f"{class_name} (\n"
            f"   id: {self.identifier} ({'aligned' if self.aligned else 'unaligned'})\n"
            f"   description: {self.description}\n"
            f"   residues: {joined_residues}\n"
            f")"
        )

    @abstractmethod
    def normalize(self) -> "SequenceType":
        """Normalize the residues for this sequence type. Return a new SequenceType with normalized residues."""
        raise NotImplementedError

    @abstractmethod
    def denormalize(self) -> List[str]:
        """Denormalize the residues for this sequence type. Return a list of denormalized residues."""
        raise NotImplementedError

    @abstractmethod
    def _validate(self) -> None:
        """Validate the residues for this sequence type. Raise ValueError if invalid."""
        raise NotImplementedError


@dataclass(frozen=True)
class RNASequence(SequenceType):
    """RNA sequence; extends SequenceType. Reserved for A,C,G,U (and common ambiguous N)."""

    def __post_init__(self) -> None:
        # Uppercase all residues on initialization, then validate
        uppercased: List[str] = [r.upper() for r in self.residues]
        object.__setattr__(self, "residues", uppercased)
        super().__post_init__()

    def normalize(self) -> "RNASequence":
        """Normalize the residues for this sequence type. Return a new SequenceType with normalized residues."""
        if self.normalized is True:
            raise ValueError("Sequence is already normalized.")
        if self.aligned is True:
            raise ValueError("Cannot normalize aligned sequence.")

        return RNASequence(
            identifier=self.identifier,
            residues=[res.upper().replace("T", "U") for res in self.residues],
            description=self.description,
            aligned=self.aligned,
            normalized=True,
        )

    def denormalize(self) -> "RNASequence":
        """Denormalize the residues for this sequence type. Return a list of denormalized residues."""
        if self.normalized is False:
            raise ValueError("Sequence is not normalized.")
        if self.aligned is True:
            raise ValueError("Cannot denormalize aligned sequence.")

        return RNASequence(
            identifier=self.identifier,
            residues=[res.upper().replace("U", "T") for res in self.residues],
            description=self.description,
            aligned=self.aligned,
            normalized=False,
        )

    # Additional RNA-specific helpers can be added here in the future.
    def _validate(self) -> None:
        if self.aligned is True:
            allowed = {"A", "C", "G", "U", "-"}
        elif self.normalized is False:
            allowed = {"A", "C", "G", "T"}
        else:
            allowed = {"A", "C", "G", "U"}

        invalid = {ch for ch in self.residues if ch not in allowed}
        if invalid:
            raise ValueError(
                f"Invalid RNA residues: {sorted(invalid)}; allowed: {sorted(allowed)}"
            )


__all__ = ["SequenceType", "RNASequence"]

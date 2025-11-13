"""
This module defines key data types for the parameters of a Hidden Markov Model (HMM)
designed to model affine-gap RNA sequence alignments. It includes the specification
of emission and transition probability tables, their validation mechanism, and constants
for the canonical RNA bases and HMM states.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Sequence

RNA_BASES: Tuple[str, str, str, str] = ("A", "C", "G", "U")
HMM_STATES: Tuple[str, str, str] = ("M", "X", "Y")


def _validate_keys(
    data: Dict[str, object], expected: Sequence[str], context: str
) -> None:
    missing = [k for k in expected if k not in data]
    if missing:
        raise ValueError(f"{context} missing keys: {missing}")
    unexpected = [k for k in data if k not in expected]
    if unexpected:
        raise ValueError(f"{context} has unexpected keys: {unexpected}")


@dataclass(frozen=True)
class EmissionParameters:
    """Emission probability tables for the affine-gap HMM."""

    match: Dict[str, Dict[str, float]]
    insert_x: Dict[str, float]
    insert_y: Dict[str, float]

    def __post_init__(self) -> None:
        _validate_keys(self.match, RNA_BASES, "match emissions")
        _validate_keys(self.insert_x, RNA_BASES, "insert_x emissions")
        _validate_keys(self.insert_y, RNA_BASES, "insert_y emissions")

        for base, row in self.match.items():
            _validate_keys(row, RNA_BASES, f"match[{base}]")


@dataclass(frozen=True)
class TransitionParameters:
    """Transition probabilities between HMM states."""

    matrix: Dict[str, Dict[str, float]]

    def __post_init__(self) -> None:
        _validate_keys(self.matrix, HMM_STATES, "transition rows")
        for state, row in self.matrix.items():
            _validate_keys(row, HMM_STATES, f"transition[{state}]")


@dataclass(frozen=True)
class GapParameters:
    """Summarized gap open/extend probabilities."""

    delta: float
    epsilon: float
    epsilon_x: float
    epsilon_y: float


@dataclass(frozen=True)
class HMMParameters:
    """Aggregate container for HMM emission/transition parameters."""

    log_emissions: EmissionParameters
    log_transitions: TransitionParameters
    gaps: GapParameters


__all__ = [
    "EmissionParameters",
    "TransitionParameters",
    "GapParameters",
    "HMMParameters",
    "RNA_BASES",
    "HMM_STATES",
]

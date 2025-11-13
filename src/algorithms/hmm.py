"""Pair Hidden Markov Model utilities."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Mapping, MutableMapping, Optional

from src.types.parameters import (
    HMMParameters,
    HMM_STATES,
    RNA_BASES,
)


def _validate_prob_table(
    table: Mapping[str, Mapping[str, float]],
    expected_outer: tuple[str, ...],
    expected_inner: tuple[str, ...],
    context: str,
) -> None:
    """Ensure a nested mapping contains the required keys."""
    missing_outer = [state for state in expected_outer if state not in table]
    if missing_outer:
        raise ValueError(f"{context} missing outer keys: {missing_outer}")

    unexpected_outer = [state for state in table if state not in expected_outer]
    if unexpected_outer:
        raise ValueError(f"{context} has unexpected outer keys: {unexpected_outer}")

    for outer, inner_map in table.items():
        missing_inner = [key for key in expected_inner if key not in inner_map]
        if missing_inner:
            raise ValueError(f"{context}[{outer}] missing inner keys: {missing_inner}")
        unexpected_inner = [key for key in inner_map if key not in expected_inner]
        if unexpected_inner:
            raise ValueError(
                f"{context}[{outer}] has unexpected inner keys: {unexpected_inner}"
            )


def _validate_emission_tables(
    params: HMMParameters,
) -> None:
    """Sanity-check emission tables adhere to expected alphabet."""
    match = params.log_emissions.match
    insert_x = params.log_emissions.insert_x
    insert_y = params.log_emissions.insert_y

    _validate_prob_table(match, RNA_BASES, RNA_BASES, "match emissions")

    missing_insert_x = [base for base in RNA_BASES if base not in insert_x]
    if missing_insert_x:
        raise ValueError(f"insert_x emissions missing bases: {missing_insert_x}")
    unexpected_insert_x = [base for base in insert_x if base not in RNA_BASES]
    if unexpected_insert_x:
        raise ValueError(
            f"insert_x emissions has unexpected bases: {unexpected_insert_x}"
        )

    missing_insert_y = [base for base in RNA_BASES if base not in insert_y]
    if missing_insert_y:
        raise ValueError(f"insert_y emissions missing bases: {missing_insert_y}")
    unexpected_insert_y = [base for base in insert_y if base not in RNA_BASES]
    if unexpected_insert_y:
        raise ValueError(
            f"insert_y emissions has unexpected bases: {unexpected_insert_y}"
        )


def _validate_transition_table(params: HMMParameters) -> None:
    """Verify transition matrix contains all states."""
    matrix = params.log_transitions.matrix
    _validate_prob_table(matrix, HMM_STATES, HMM_STATES, "transition matrix")


def _default_uniform_log_probs(states: tuple[str, ...]) -> Dict[str, float]:
    """Create uniform log-probability distribution."""
    log_prob = -math.log(len(states))
    return {state: log_prob for state in states}


@dataclass(frozen=True)
class PairHMM:
    """Pairwise HMM wrapper exposing log-space parameters and helpers."""

    params: HMMParameters
    start_log_probs: Mapping[str, float]
    end_log_probs: Mapping[str, float]

    def __init__(
        self,
        params: HMMParameters,
        start_log_probs: Optional[Mapping[str, float]] = None,
        end_log_probs: Optional[Mapping[str, float]] = None,
    ) -> None:
        _validate_emission_tables(params)
        _validate_transition_table(params)

        object.__setattr__(self, "params", params)

        start_defaults = (
            dict(start_log_probs)
            if start_log_probs is not None
            else _default_uniform_log_probs(HMM_STATES)
        )
        _ensure_state_keys(start_defaults, "start_log_probs")
        object.__setattr__(self, "start_log_probs", start_defaults)

        end_defaults = (
            dict(end_log_probs)
            if end_log_probs is not None
            else {state: 0.0 for state in HMM_STATES}
        )
        _ensure_state_keys(end_defaults, "end_log_probs")
        object.__setattr__(self, "end_log_probs", end_defaults)

    @property
    def log_match(self) -> Mapping[str, Mapping[str, float]]:
        """Return the match emission matrix in log-space."""
        return self.params.log_emissions.match

    @property
    def log_insert_x(self) -> Mapping[str, float]:
        """Return the insert_x emission matrix in log-space."""
        return self.params.log_emissions.insert_x

    @property
    def log_insert_y(self) -> Mapping[str, float]:
        """Return the insert_y emission matrix in log-space."""
        return self.params.log_emissions.insert_y

    @property
    def log_transitions(self) -> Mapping[str, Mapping[str, float]]:
        """Return the transition matrix in log-space."""
        return self.params.log_transitions.matrix

    @property
    def gap_params(self):
        """Return the gap parameters."""
        return self.params.gaps

    def log_emit_M(self, x_base: str, y_base: str) -> float:
        """Return the log emission probability for the match state."""
        self._assert_base(x_base, "x_base")
        self._assert_base(y_base, "y_base")
        return self.log_match[x_base][y_base]

    def log_emit_X(self, x_base: str) -> float:
        """Return the log emission probability for the insert_x state."""
        self._assert_base(x_base, "x_base")
        return self.log_insert_x[x_base]

    def log_emit_Y(self, y_base: str) -> float:
        """Return the log emission probability for the insert_y state."""
        self._assert_base(y_base, "y_base")
        return self.log_insert_y[y_base]

    def log_trans(self, state_from: str, state_to: str) -> float:
        """Return the log transition probability for the given states."""
        self._assert_state(state_from, "state_from")
        self._assert_state(state_to, "state_to")
        return self.log_transitions[state_from][state_to]

    def _assert_base(self, base: str, arg_name: str) -> None:
        if base not in RNA_BASES:
            raise ValueError(f"{arg_name} must be one of {RNA_BASES}, got '{base}'")

    def _assert_state(self, state: str, arg_name: str) -> None:
        if state not in HMM_STATES:
            raise ValueError(f"{arg_name} must be one of {HMM_STATES}, got '{state}'")


def _ensure_state_keys(mapping: MutableMapping[str, float], context: str) -> None:
    missing = [state for state in HMM_STATES if state not in mapping]
    if missing:
        raise ValueError(f"{context} missing state keys: {missing}")
    unexpected = [state for state in mapping if state not in HMM_STATES]
    if unexpected:
        raise ValueError(f"{context} has unexpected state keys: {unexpected}")


__all__ = ["PairHMM"]

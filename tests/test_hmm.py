"""Unit tests for PairHMM convenience wrapper."""

from __future__ import annotations

import math

import pytest

from src.algorithms.hmm import PairHMM
from src.types.parameters import (
    EmissionParameters,
    GapParameters,
    HMMParameters,
    RNA_BASES,
    TransitionParameters,
)


def _build_uniform_params() -> HMMParameters:
    """Create a simple uniform-parameter HMM in log-space."""
    log_emit = math.log(1.0 / len(RNA_BASES))
    match = {x: {y: log_emit for y in RNA_BASES} for x in RNA_BASES}
    insert_x = {base: log_emit for base in RNA_BASES}
    insert_y = {base: log_emit for base in RNA_BASES}

    log_trans = math.log(1.0 / 3.0)
    transition_matrix = {
        state_from: {state_to: log_trans for state_to in ("M", "X", "Y")}
        for state_from in ("M", "X", "Y")
    }

    emissions = EmissionParameters(
        match=match,
        insert_x=insert_x,
        insert_y=insert_y,
    )
    transitions = TransitionParameters(matrix=transition_matrix)
    gaps = GapParameters(delta=0.1, epsilon=0.2, epsilon_x=0.2, epsilon_y=0.2)
    return HMMParameters(
        log_emissions=emissions,
        log_transitions=transitions,
        gaps=gaps,
    )


def test_pair_hmm_emission_helpers_return_expected_logs():
    """Test that the PairHMM emission helpers return expected log probabilities."""
    params = _build_uniform_params()
    hmm = PairHMM(params)

    expected_emit = math.log(0.25)
    assert math.isclose(hmm.log_emit_M("A", "C"), expected_emit)
    assert math.isclose(hmm.log_emit_X("G"), expected_emit)
    assert math.isclose(hmm.log_emit_Y("U"), expected_emit)


def test_pair_hmm_transition_helper_and_defaults():
    """Test that the PairHMM transition helper and defaults return expected values."""
    params = _build_uniform_params()
    hmm = PairHMM(params)

    expected_trans = math.log(1 / 3)
    assert math.isclose(hmm.log_trans("M", "X"), expected_trans)
    assert math.isclose(hmm.start_log_probs["M"], math.log(1 / 3))
    assert math.isclose(hmm.end_log_probs["Y"], 0.0)


def test_pair_hmm_rejects_unknown_bases_or_states():
    """Test that the PairHMM rejects unknown bases or states."""
    params = _build_uniform_params()
    hmm = PairHMM(params)

    with pytest.raises(ValueError):
        hmm.log_emit_X("Z")

    with pytest.raises(ValueError):
        hmm.log_trans("M", "Q")

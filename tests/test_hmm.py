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


def test_pair_hmm_all_emission_helpers():
    """Test all PairHMM emission helper methods return expected log probs."""
    params = _build_uniform_params()
    hmm = PairHMM(params)

    expected_emit = math.log(0.25)  # Uniform over 4 bases

    # Test all match pairs
    for x_base in ("A", "C", "G", "U"):
        for y_base in ("A", "C", "G", "U"):
            assert math.isclose(hmm.log_emit_M(x_base, y_base), expected_emit)

    # Test all insert_x bases
    for x_base in ("A", "C", "G", "U"):
        assert math.isclose(hmm.log_emit_X(x_base), expected_emit)

    # Test all insert_y bases
    for y_base in ("A", "C", "G", "U"):
        assert math.isclose(hmm.log_emit_Y(y_base), expected_emit)


def test_pair_hmm_all_transitions():
    """Test all PairHMM transition helper methods."""
    params = _build_uniform_params()
    hmm = PairHMM(params)

    expected_trans = math.log(1.0 / 3.0)  # Uniform over 3 states

    for from_state in ("M", "X", "Y"):
        for to_state in ("M", "X", "Y"):
            assert math.isclose(hmm.log_trans(from_state, to_state), expected_trans)


def test_pair_hmm_properties():
    """Test that PairHMM exposes expected properties."""
    params = _build_uniform_params()
    hmm = PairHMM(params)

    # Check properties exist and have correct structure
    assert isinstance(hmm.log_match, dict)
    assert isinstance(hmm.log_insert_x, dict)
    assert isinstance(hmm.log_insert_y, dict)
    assert isinstance(hmm.log_transitions, dict)
    assert hmm.gap_params is not None

    # Check dimensions
    assert len(hmm.log_match) == 4  # 4 RNA bases
    assert len(hmm.log_insert_x) == 4
    assert len(hmm.log_insert_y) == 4
    assert len(hmm.log_transitions) == 3  # 3 HMM states


def test_pair_hmm_start_end_probabilities():
    """Test start and end log probabilities are properly set."""
    params = _build_uniform_params()

    # Test with defaults
    hmm1 = PairHMM(params)
    assert len(hmm1.start_log_probs) == 3
    assert len(hmm1.end_log_probs) == 3
    assert all(state in hmm1.start_log_probs for state in ("M", "X", "Y"))
    assert all(state in hmm1.end_log_probs for state in ("M", "X", "Y"))

    # Test with custom start and end probs
    custom_start = {"M": math.log(0.7), "X": math.log(0.2), "Y": math.log(0.1)}
    custom_end = {"M": math.log(0.5), "X": math.log(0.3), "Y": math.log(0.2)}
    hmm2 = PairHMM(params, start_log_probs=custom_start, end_log_probs=custom_end)

    assert math.isclose(hmm2.start_log_probs["M"], math.log(0.7))
    assert math.isclose(hmm2.end_log_probs["X"], math.log(0.3))


def test_pair_hmm_frozen_dataclass():
    """Test that PairHMM is frozen and cannot be modified after initialization."""
    params = _build_uniform_params()
    hmm = PairHMM(params)

    with pytest.raises((TypeError, AttributeError)):
        hmm.start_log_probs = {"M": 0.0, "X": 0.0, "Y": 0.0}


def test_pair_hmm_invalid_emission_tables():
    """Test that EmissionParameters validates invalid emission parameters."""
    # Missing key in match emissions - should raise at EmissionParameters level
    bad_match = {
        "A": {"A": 0, "C": 0, "G": 0},  # Missing U
        "C": {"A": 0, "C": 0, "G": 0, "U": 0},
        "G": {"A": 0, "C": 0, "G": 0, "U": 0},
        "U": {"A": 0, "C": 0, "G": 0, "U": 0},
    }

    # Should raise an error during EmissionParameters validation
    with pytest.raises(ValueError):
        EmissionParameters(
            match=bad_match,
            insert_x={base: 0 for base in ("A", "C", "G", "U")},
            insert_y={base: 0 for base in ("A", "C", "G", "U")},
        )


def test_pair_hmm_invalid_transition_matrix():
    """Test that TransitionParameters validates invalid transition matrices."""
    # Missing transition state - should raise at TransitionParameters level
    with pytest.raises(ValueError):
        TransitionParameters(
            matrix={
                "M": {"M": 0, "X": 0},  # Missing Y
                "X": {"M": 0, "X": 0, "Y": 0},
                "Y": {"M": 0, "X": 0, "Y": 0},
            }
        )


def test_pair_hmm_handles_edge_cases():
    """Test that PairHMM handles edge cases in base and state validation."""
    params = _build_uniform_params()
    hmm = PairHMM(params)

    # Test with lowercase (should still work as RNASequence uppercases)
    with pytest.raises(ValueError):
        hmm.log_emit_X("n")  # Invalid base

    # Test invalid state transitions
    with pytest.raises(ValueError):
        hmm.log_trans("START", "M")

    with pytest.raises(ValueError):
        hmm.log_trans("M", "END")

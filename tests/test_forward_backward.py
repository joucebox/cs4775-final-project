"""Unit tests for forward/backward scaffolding utilities."""

from __future__ import annotations

import math

import pytest

from src.algorithms.forward_backward import (
    compute_backward,
    compute_forward,
    logsumexp,
)
from src.algorithms.hmm import PairHMM
from src.types.parameters import (
    EmissionParameters,
    GapParameters,
    HMMParameters,
    RNA_BASES,
    TransitionParameters,
)
from src.types.sequence import RNASequence


def _build_uniform_params() -> HMMParameters:
    """Create a simple uniform-parameter HMM in log-space.

    - Emissions:
        * Match: P(x,y | M) = 0.25 for all x,y
        * Insert X: P(x | X) = 0.25 for all x
        * Insert Y: P(y | Y) = 0.25 for all y
    - Transitions:
        * From any state, P(next_state) = 1/3 for M,X,Y
    - Gaps:
        * Arbitrary values, not used directly by the DP.
    """
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


def _toy_hmm() -> PairHMM:
    """Convenience constructor for a PairHMM with uniform parameters."""
    params = _build_uniform_params()
    return PairHMM(params)


def test_logsumexp_handles_negative_infs():
    """Test that the logsumexp function handles negative infinities."""
    values = [math.log(0.2), math.log(0.3), float("-inf")]
    expected = math.log(0.5)
    assert math.isclose(logsumexp(values), expected)
    assert logsumexp([]) == float("-inf")


def test_forward_backward_shapes_and_consistency():
    """Forward/backward matrices should have correct shapes and matching logZ."""
    hmm = _toy_hmm()
    # aligned=True since we’re in alignment / HMM-land (A,C,G,U, -, . allowed)
    x_seq = RNASequence(identifier="x", residues=list("AC"), description=None, aligned=True)
    y_seq = RNASequence(identifier="y", residues=list("GU"), description=None, aligned=True)

    F_M, F_X, F_Y, logZ_f = compute_forward(hmm, x_seq, y_seq)
    B_M, B_X, B_Y, logZ_b = compute_backward(hmm, x_seq, y_seq)

    n, m = len(x_seq), len(y_seq)

    # Shapes
    assert len(F_M) == len(F_X) == len(F_Y) == n + 1
    assert len(B_M) == len(B_X) == len(B_Y) == n + 1
    assert all(len(row) == m + 1 for row in F_M)
    assert all(len(row) == m + 1 for row in F_X)
    assert all(len(row) == m + 1 for row in F_Y)
    assert all(len(row) == m + 1 for row in B_M)
    assert all(len(row) == m + 1 for row in B_X)
    assert all(len(row) == m + 1 for row in B_Y)

    # logZ must be finite and consistent
    assert math.isfinite(logZ_f)
    assert math.isfinite(logZ_b)
    assert math.isclose(logZ_f, logZ_b, rel_tol=1e-9, abs_tol=1e-9)


def test_forward_backward_single_base_pair_probability():
    """Analytically check probability for n=1, m=1 under the toy uniform HMM.

    For sequences of length 1 and 1, the state paths that emit both symbols
    exactly once (and then stop) are:

        1) Start -> M -> End
        2) Start -> X -> Y -> End
        3) Start -> Y -> X -> End

    Using:
        π_s       = 1/3 for s in {M, X, Y}
        a_{s->t}  = 1/3 for all s,t
        e_M(x,y)  = 0.25
        e_X(x)    = 0.25
        e_Y(y)    = 0.25
        a_{s->End}= 1 (log 0) for all s

    We get:
        P1 = (1/3) * 0.25                        = 1/12
        P2 = (1/3) * 0.25 * (1/3) * 0.25         = 1/144
        P3 = P2                                   = 1/144

        Total P = 1/12 + 2*(1/144) = 7/72

    We verify that logZ from forward/backward equals log(7/72).
    """
    hmm = _toy_hmm()
    x_seq = RNASequence(identifier="x", residues=list("A"), description=None, aligned=True)
    y_seq = RNASequence(identifier="y", residues=list("U"), description=None, aligned=True)

    _, _, _, logZ_f = compute_forward(hmm, x_seq, y_seq)
    _, _, _, logZ_b = compute_backward(hmm, x_seq, y_seq)

    expected_prob = 7.0 / 72.0
    expected_logZ = math.log(expected_prob)

    assert math.isclose(logZ_f, expected_logZ, rel_tol=1e-8, abs_tol=1e-8)
    assert math.isclose(logZ_b, expected_logZ, rel_tol=1e-8, abs_tol=1e-8)


def test_forward_backward_single_insertion_x_probability():
    """Analytically check probability for n=1, m=0 (only X insertion).

    With n=1, m=0, the only valid path that emits exactly one x-symbol and no y-symbols is:
        Start -> X -> End

    Probability:
        P = π_X * e_X(x) * a_{X->End}
          = (1/3) * 0.25 * 1
          = 1/12
    """
    hmm = _toy_hmm()
    x_seq = RNASequence(identifier="x", residues=list("A"), description=None, aligned=True)
    # No residues in y
    y_seq = RNASequence(identifier="y", residues=[], description=None, aligned=True)

    _, _, _, logZ_f = compute_forward(hmm, x_seq, y_seq)
    _, _, _, logZ_b = compute_backward(hmm, x_seq, y_seq)

    expected_prob = 1.0 / 12.0
    expected_logZ = math.log(expected_prob)

    assert math.isclose(logZ_f, expected_logZ, rel_tol=1e-8, abs_tol=1e-8)
    assert math.isclose(logZ_b, expected_logZ, rel_tol=1e-8, abs_tol=1e-8)


def test_empty_sequences_logZ_is_negative_infinity_and_matches():
    """When both sequences are empty, logZ should be -inf and consistent.

    Under the current implementation, empty sequences produce no emissions,
    and we do not include an explicit Start->End probability, so logZ is -inf.
    Forward and backward should agree on this.
    """
    hmm = _toy_hmm()
    x_seq = RNASequence(identifier="x", residues=[], description=None, aligned=True)
    y_seq = RNASequence(identifier="y", residues=[], description=None, aligned=True)

    _, _, _, logZ_f = compute_forward(hmm, x_seq, y_seq)
    _, _, _, logZ_b = compute_backward(hmm, x_seq, y_seq)

    assert logZ_f == float("-inf")
    assert logZ_b == float("-inf")
"""Tests for forward/backward scaffolding utilities."""

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
    TransitionParameters,
)
from src.types.sequence import RNASequence


def _toy_hmm() -> PairHMM:
    log_emit = math.log(0.25)
    match = {
        base: {b: log_emit for b in ("A", "C", "G", "U")}
        for base in ("A", "C", "G", "U")
    }
    insert = {base: log_emit for base in ("A", "C", "G", "U")}
    transitions = {
        state: {s: math.log(1 / 3) for s in ("M", "X", "Y")}
        for state in ("M", "X", "Y")
    }
    params = HMMParameters(
        log_emissions=EmissionParameters(match=match, insert_x=insert, insert_y=insert),
        log_transitions=TransitionParameters(matrix=transitions),
        gaps=GapParameters(delta=0.1, epsilon=0.2, epsilon_x=0.2, epsilon_y=0.2),
    )
    return PairHMM(params)


def test_logsumexp_handles_negative_infs():
    """Test that the logsumexp function handles negative infinities."""
    values = [math.log(0.2), math.log(0.3), float("-inf")]
    expected = math.log(0.5)
    assert math.isclose(logsumexp(values), expected)
    assert logsumexp([]) == float("-inf")


@pytest.mark.parametrize(
    "compute_fn",
    [compute_forward, compute_backward],
)
def test_forward_backward_raise_not_implemented(compute_fn):
    """Test that the forward and backward algorithms raise NotImplementedError."""
    hmm = _toy_hmm()
    x_seq = RNASequence(identifier="x", residues=list("AC"), description=None)
    y_seq = RNASequence(identifier="y", residues=list("GT"), description=None)

    with pytest.raises(NotImplementedError):
        compute_fn(hmm, x_seq, y_seq)

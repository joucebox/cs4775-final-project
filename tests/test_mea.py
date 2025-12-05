"""Unit tests for MEAAligner (Maximum Expected Accuracy decoding)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.algorithms.hmm import PairHMM
from src.algorithms.mea import MEAAligner
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
        * Arbitrary values, not used directly by the MEA DP.
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


def test_mea_gamma_must_be_in_valid_range():
    """MEAAligner should reject gamma values outside valid range for each method."""
    # Power method (default): gamma must be > 0 (no upper bound)
    with pytest.raises(ValueError):
        MEAAligner(gamma=0.0)
    with pytest.raises(ValueError):
        MEAAligner(gamma=-1.0)
    MEAAligner(gamma=2.0)  # Should be fine (power allows > 1)

    # Threshold method: gamma must be in (0, 1]
    with pytest.raises(ValueError):
        MEAAligner(gamma=0.0, method="threshold")
    with pytest.raises(ValueError):
        MEAAligner(gamma=-1.0, method="threshold")
    with pytest.raises(ValueError):
        MEAAligner(gamma=1.5, method="threshold")

    # Log-odds method: gamma must be in (0, 1]
    with pytest.raises(ValueError):
        MEAAligner(gamma=0.0, method="log_odds")
    with pytest.raises(ValueError):
        MEAAligner(gamma=1.5, method="log_odds")

    # ProbCons method: gamma must be > 0 (no upper bound)
    with pytest.raises(ValueError):
        MEAAligner(gamma=0.0, method="probcons")
    with pytest.raises(ValueError):
        MEAAligner(gamma=-1.0, method="probcons")
    MEAAligner(gamma=2.0, method="probcons")  # Should be fine

    # Valid cases for power (default)
    MEAAligner(gamma=1.0)
    MEAAligner(gamma=0.5)


def test_mea_method_selection():
    """MEAAligner should accept different methods."""
    # All methods should be valid
    MEAAligner(gamma=0.5, method="power")
    MEAAligner(gamma=0.5, method="threshold")
    MEAAligner(gamma=0.5, method="log_odds")
    MEAAligner(gamma=0.75, method="probcons")

    # Invalid method should raise
    with pytest.raises(ValueError):
        MEAAligner(gamma=0.5, method="invalid")


def test_mea_alignment_basic_properties():
    """MEAAligner should return a well-formed AlignmentResult and Alignment."""
    hmm = _toy_hmm()
    aligner = MEAAligner(gamma=1.0)

    # Use only A,C,G so they're valid for aligned=False and for RNA_BASES
    x_seq = RNASequence(
        identifier="x", residues=list("AC"), description=None, aligned=False
    )
    y_seq = RNASequence(
        identifier="y", residues=list("AG"), description=None, aligned=False
    )

    result = aligner.align(hmm, x_seq, y_seq)

    # AlignmentResult basics
    assert result.alignment is not None
    assert isinstance(result.score, float)
    assert result.posteriors is not None

    alignment = result.alignment

    # Alignment should contain exactly two sequences
    assert alignment.num_sequences == 2
    aligned_x, aligned_y = alignment.aligned_sequences
    orig_x, orig_y = alignment.original_sequences

    # Original sequences should be the same objects we passed in
    assert orig_x.residues == x_seq.residues
    assert orig_y.residues == y_seq.residues
    assert orig_x.aligned is False
    assert orig_y.aligned is False
    assert orig_x.normalized is False
    assert orig_y.normalized is False

    # Aligned sequences should be marked aligned=True and have same column length
    assert aligned_x.aligned is True
    assert aligned_y.aligned is True
    assert len(aligned_x) == len(aligned_y) == alignment.columns

    # Posterior matrix shape should be (n+1, m+1)
    n, m = len(x_seq), len(y_seq)
    assert result.posteriors.shape == (n + 1, m + 1)


def test_mea_posteriors_are_probabilities():
    """Posterior match matrix should contain values in [0, 1]."""
    hmm = _toy_hmm()
    aligner = MEAAligner(gamma=1.0)
    x_seq = RNASequence(
        identifier="x", residues=list("ACG"), description=None, aligned=False
    )
    y_seq = RNASequence(
        identifier="y", residues=list("AGG"), description=None, aligned=False
    )

    result = aligner.align(hmm, x_seq, y_seq)
    post = result.posteriors

    assert post is not None
    assert np.all(post >= 0.0)
    assert np.all(post <= 1.0)


def test_mea_single_base_pair_posterior_and_score():
    """Analytically check posterior and score for n=1, m=1 under the toy HMM.

    As derived in the forward/backward tests, for sequences of length 1 and 1
    under the uniform toy HMM:

        - Total probability P(x,y) = 7/72
        - Probability that we're in state M at (1,1) is 1/12
        - Therefore posterior P(M at (1,1) | x,y) = (1/12) / (7/72) = 6/7

    For MEA with gamma=1:
        - The only non-zero weight is w[1][1] = 6/7
        - The optimal alignment matches the two bases (no gaps)
        - The MEA score is then 6/7.
    """
    hmm = _toy_hmm()
    aligner = MEAAligner(gamma=1.0)

    # Single-base sequences; identity doesn't matter for the uniform HMM.
    x_seq = RNASequence(identifier="x", residues=["A"], description=None, aligned=False)
    y_seq = RNASequence(identifier="y", residues=["C"], description=None, aligned=False)

    result = aligner.align(hmm, x_seq, y_seq)

    post = result.posteriors
    assert post is not None

    # Posterior for M at (1,1) should be 6/7
    expected_post = 6.0 / 7.0
    assert math.isclose(post[1, 1], expected_post, rel_tol=1e-8, abs_tol=1e-8)

    # Score should be equal to that posterior (no other matches possible)
    assert math.isclose(result.score, expected_post, rel_tol=1e-8, abs_tol=1e-8)

    # Alignment should be gap-free of length 1
    aligned_x, aligned_y = result.alignment.aligned_sequences
    assert len(aligned_x) == len(aligned_y) == 1
    assert aligned_x.residues[0] == "A"
    assert aligned_y.residues[0] == "C"


def test_mea_identical_sequences_prefers_gap_free_alignment():
    """MEA should choose a gap-free alignment for identical short sequences."""
    hmm = _toy_hmm()
    aligner = MEAAligner(gamma=1.0)

    x_seq = RNASequence(
        identifier="x", residues=list("AC"), description=None, aligned=False
    )
    y_seq = RNASequence(
        identifier="y", residues=list("AC"), description=None, aligned=False
    )

    result = aligner.align(hmm, x_seq, y_seq)
    aligned_x, aligned_y = result.alignment.aligned_sequences

    # No gaps expected in either aligned sequence
    assert "-" not in aligned_x.residues
    assert "-" not in aligned_y.residues

    # Lengths should match originals (global-style alignment)
    assert len(aligned_x) == len(x_seq)
    assert len(aligned_y) == len(y_seq)

    # Positions should be aligned one-to-one
    assert aligned_x.residues == aligned_y.residues

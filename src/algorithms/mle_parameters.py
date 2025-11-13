"""
Maximum Likelihood Estimation (MLE) of Hidden Markov Model (HMM) parameters from alignments.

This module provides functionality to estimate the parameters of an affine-gap HMM—specifically,
emission and transition probabilities—directly from observed aligned RNA sequence data.
The process follows these broad steps:

- Given a collection of pairwise or multiple sequence alignments (over A, C, G, U, and gaps),
  the code:
    - Iterates through aligned columns to count:
        * Base-pair matches (emissions) by state (Match, Insert_X, Insert_Y)
        * State transitions along the alignment path (e.g. M → X, X → Y, etc.)
        * Gap open and extend events between columns
    - Aggregates counts across all alignments
    - Converts observed counts to probabilities for model emission and transition parameters,
      using relative frequencies
- Estimated parameters include:
    - Match emission matrix (P(x, y | state = M))
    - Insertion emission probabilities (for insertions in either sequence)
    - State transition probability matrix (P(next_state | current_state))
    - Gap opening and extension probabilities (affine gap model)

Emission and transition results are returned in log-space (natural logarithm) for numerical
stability, while gap parameters remain in probability space.
"""

from __future__ import annotations

import itertools
import math
import random
from typing import Dict, List, Optional, Sequence, Tuple

from src.types import Alignment, SequenceType
from src.types.parameters import (
    EmissionParameters,
    TransitionParameters,
    GapParameters,
    HMMParameters,
    RNA_BASES,
    HMM_STATES,
)

BASES: Tuple[str, ...] = RNA_BASES
STATES: Tuple[str, ...] = HMM_STATES

MatchCountMatrix = Dict[str, Dict[str, int]]
MatchProbMatrix = Dict[str, Dict[str, float]]
TransitionCountMatrix = Dict[str, Dict[str, int]]
TransitionProbMatrix = Dict[str, Dict[str, float]]


def _state_at(x_i: str, y_i: str) -> Optional[str]:
    """Return HMM state label for a pair of aligned residues."""
    if x_i == "." or y_i == ".":
        return None
    if x_i == "-" and y_i == "-":
        return None
    if x_i in BASES and y_i == "-":
        return "X"
    if x_i == "-" and y_i in BASES:
        return "Y"
    if x_i in BASES and y_i in BASES:
        return "M"
    return None


def _update_counts_from_pair(
    x_aln: SequenceType,
    y_aln: SequenceType,
    match_counts: MatchCountMatrix,
    insert_x_counts: Dict[str, int],
    insert_y_counts: Dict[str, int],
    transition_counts: TransitionCountMatrix,
) -> None:
    """Accumulate emission and transition counts for one aligned sequence pair."""
    if len(x_aln) != len(y_aln):
        raise ValueError("Aligned sequences must share the same length.")

    states: List[Optional[str]] = [None] * len(x_aln)
    for idx in range(len(x_aln)):
        states[idx] = _state_at(x_aln.residues[idx], y_aln.residues[idx])

    for idx, state in enumerate(states):
        x_i, y_i = x_aln.residues[idx], y_aln.residues[idx]
        if state == "M":
            if x_i in BASES and y_i in BASES:
                match_counts[x_i][y_i] += 1
        elif state == "X":
            if x_i in BASES:
                insert_x_counts[x_i] += 1
        elif state == "Y":
            if y_i in BASES:
                insert_y_counts[y_i] += 1

    prev_state: Optional[str] = None
    for state in states:
        if state is None:
            prev_state = None
            continue
        if prev_state is not None:
            transition_counts[prev_state][state] += 1
        prev_state = state


def _accumulate_counts_from_alignment(
    alignment: Alignment,
    match_counts: MatchCountMatrix,
    insert_x_counts: Dict[str, int],
    insert_y_counts: Dict[str, int],
    transition_counts: TransitionCountMatrix,
    max_pairs: Optional[int] = None,
) -> None:
    """Aggregate counts across all (or sampled) sequence pairs within an alignment."""

    sequences = alignment.aligned_sequences
    if len(sequences) < 2:
        return

    pair_indices = list(itertools.combinations(range(len(sequences)), 2))
    if max_pairs is not None and max_pairs < len(pair_indices):
        pair_indices = random.sample(pair_indices, k=max_pairs)

    for i, j in pair_indices:
        _update_counts_from_pair(
            sequences[i],
            sequences[j],
            match_counts,
            insert_x_counts,
            insert_y_counts,
            transition_counts,
        )


def _normalize_emissions(
    match_counts: MatchCountMatrix,
    insert_x_counts: Dict[str, int],
    insert_y_counts: Dict[str, int],
    pseudocount: float = 0.0,
) -> EmissionParameters:
    """Normalize emission counts and return log-space emission parameters."""
    match_probs: MatchProbMatrix = {x: {} for x in BASES}
    for x in BASES:
        row_total = sum(match_counts[x][y] + pseudocount for y in BASES)
        for y in BASES:
            numerator = match_counts[x][y] + pseudocount
            if row_total == 0:
                match_probs[x][y] = 1.0 / len(BASES)
            else:
                match_probs[x][y] = numerator / row_total

    total_x = sum(insert_x_counts[base] + pseudocount for base in BASES)
    insert_x_probs: Dict[str, float] = {}
    for base in BASES:
        numerator = insert_x_counts[base] + pseudocount
        insert_x_probs[base] = numerator / total_x if total_x > 0 else 1.0 / len(BASES)

    total_y = sum(insert_y_counts[base] + pseudocount for base in BASES)
    insert_y_probs: Dict[str, float] = {}
    for base in BASES:
        numerator = insert_y_counts[base] + pseudocount
        insert_y_probs[base] = numerator / total_y if total_y > 0 else 1.0 / len(BASES)

    match_logs: MatchProbMatrix = {x: {} for x in BASES}
    for x in BASES:
        for y in BASES:
            prob = match_probs[x][y]
            match_logs[x][y] = math.log(prob) if prob > 0.0 else float("-inf")

    insert_x_logs: Dict[str, float] = {}
    for base, prob in insert_x_probs.items():
        insert_x_logs[base] = math.log(prob) if prob > 0.0 else float("-inf")

    insert_y_logs: Dict[str, float] = {}
    for base, prob in insert_y_probs.items():
        insert_y_logs[base] = math.log(prob) if prob > 0.0 else float("-inf")

    return EmissionParameters(
        match=match_logs,
        insert_x=insert_x_logs,
        insert_y=insert_y_logs,
    )


def _normalize_transitions(
    transition_counts: TransitionCountMatrix,
    pseudocount: float = 0.0,
) -> Tuple[TransitionParameters, GapParameters]:
    """Normalize transition counts; return log transitions and gap parameters in probability space."""
    transition_probs: TransitionProbMatrix = {state: {} for state in STATES}
    for state in STATES:
        outgoing_total = sum(
            transition_counts[state][target] + pseudocount for target in STATES
        )
        for target in STATES:
            numerator = transition_counts[state][target] + pseudocount
            if outgoing_total == 0:
                transition_probs[state][target] = 0.0
            else:
                transition_probs[state][target] = numerator / outgoing_total

    transition_logs: TransitionProbMatrix = {state: {} for state in STATES}
    for state in STATES:
        for target in STATES:
            prob = transition_probs[state][target]
            transition_logs[state][target] = (
                math.log(prob) if prob > 0.0 else float("-inf")
            )

    gap_open = transition_probs["M"]["X"] + transition_probs["M"]["Y"]
    epsilon_x = transition_probs["X"]["X"]
    epsilon_y = transition_probs["Y"]["Y"]
    epsilon = 0.5 * (epsilon_x + epsilon_y)

    transition_params = TransitionParameters(matrix=transition_logs)
    gap_params = GapParameters(
        delta=gap_open,
        epsilon=epsilon,
        epsilon_x=epsilon_x,
        epsilon_y=epsilon_y,
    )
    return transition_params, gap_params


def estimate_params_from_alignments(
    alignments: Sequence[Alignment],
    max_pairs_per_alignment: Optional[int] = None,
    pseudocount: float = 0.0,
) -> HMMParameters:
    """Estimate HMM parameters by aggregating over a collection of alignments."""
    match_counts: MatchCountMatrix = {x: {y: 0 for y in BASES} for x in BASES}
    insert_x_counts: Dict[str, int] = {base: 0 for base in BASES}
    insert_y_counts: Dict[str, int] = {base: 0 for base in BASES}
    transition_counts: TransitionCountMatrix = {
        state: {target: 0 for target in STATES} for state in STATES
    }

    for alignment in alignments:
        _accumulate_counts_from_alignment(
            alignment,
            match_counts,
            insert_x_counts,
            insert_y_counts,
            transition_counts,
            max_pairs=max_pairs_per_alignment,
        )

    emissions = _normalize_emissions(
        match_counts, insert_x_counts, insert_y_counts, pseudocount=pseudocount
    )
    transitions, gaps = _normalize_transitions(
        transition_counts, pseudocount=pseudocount
    )

    return HMMParameters(
        log_emissions=emissions, log_transitions=transitions, gaps=gaps
    )


__all__ = ["estimate_params_from_alignments"]

"""Forward/backward dynamic programming scaffolding for the Pair HMM."""

from __future__ import annotations

import math
from typing import List, Tuple

from src.algorithms.hmm import PairHMM
from src.types.sequence import SequenceType


def logsumexp(values: List[float]) -> float:
    """Compute log(sum(exp(values))) in a numerically stable way."""
    if not values:
        return float("-inf")
    max_val = max(values)
    if max_val == float("-inf"):
        return float("-inf")
    total = sum(math.exp(v - max_val) for v in values)
    return max_val + math.log(total)


def compute_forward(
    hmm: PairHMM, x_seq: SequenceType, y_seq: SequenceType
) -> Tuple[List[List[float]], List[List[float]], List[List[float]], float]:
    """Compute forward DP matrices for the Pair HMM (log-space).

    Shapes:
        - `F_M`, `F_X`, `F_Y` all have dimensions `(len(x_seq) + 1) x (len(y_seq) + 1)`.
        - Entry `(i, j)` corresponds to prefixes `x_seq[:i]` and `y_seq[:j]`.

    Initialization (log-space):
        F_M[0][0] = logsumexp([... start transitions to M ...])  # see TODO below
        F_X[0][0] = -inf
        F_Y[0][0] = -inf

    Recurrences (all computations in log-space):
        # Match state consumes one symbol from each sequence
        F_M[i][j] = logsumexp([
            F_M[i-1][j-1] + hmm.log_trans("M", "M"),
            F_X[i-1][j-1] + hmm.log_trans("X", "M"),
            F_Y[i-1][j-1] + hmm.log_trans("Y", "M"),
        ]) + hmm.log_emit_M(x_i, y_j)

        # Insert X state consumes one symbol from x only
        F_X[i][j] = logsumexp([
            F_M[i-1][j] + hmm.log_trans("M", "X"),
            F_X[i-1][j] + hmm.log_trans("X", "X"),
            F_Y[i-1][j] + hmm.log_trans("Y", "X"),
        ]) + hmm.log_emit_X(x_i)

        # Insert Y state consumes one symbol from y only
        F_Y[i][j] = logsumexp([
            F_M[i][j-1] + hmm.log_trans("M", "Y"),
            F_X[i][j-1] + hmm.log_trans("X", "Y"),
            F_Y[i][j-1] + hmm.log_trans("Y", "Y"),
        ]) + hmm.log_emit_Y(y_j)

    Termination:
        logZ = logsumexp([
            F_M[len(x)][len(y)] + hmm.end_log_probs["M"],
            F_X[len(x)][len(y)] + hmm.end_log_probs["X"],
            F_Y[len(x)][len(y)] + hmm.end_log_probs["Y"],
        ])

    TODO:
        Implement the recurrence using the formulas above, working entirely in log-space.
    """
    raise NotImplementedError("TODO: implement forward algorithm using formulas above.")


def compute_backward(
    hmm: PairHMM, x_seq: SequenceType, y_seq: SequenceType
) -> Tuple[List[List[float]], List[List[float]], List[List[float]], float]:
    """Compute backward DP matrices for the Pair HMM (log-space).

    Shapes mirror the forward matrices `(len(x_seq) + 1) x (len(y_seq) + 1)`.

    Initialization (log-space):
        B_M[len(x)][len(y)] = hmm.end_log_probs["M"]
        B_X[len(x)][len(y)] = hmm.end_log_probs["X"]
        B_Y[len(x)][len(y)] = hmm.end_log_probs["Y"]

    Recurrences (reverse direction):
        B_M[i][j] = logsumexp([
            hmm.log_trans("M", "M") + hmm.log_emit_M(x_{i+1}, y_{j+1}) + B_M[i+1][j+1],
            hmm.log_trans("M", "X") + hmm.log_emit_X(x_{i+1}) + B_X[i+1][j],
            hmm.log_trans("M", "Y") + hmm.log_emit_Y(y_{j+1}) + B_Y[i][j+1],
        ])

        B_X[i][j] = logsumexp([
            hmm.log_trans("X", "M") + hmm.log_emit_M(x_{i+1}, y_{j+1}) + B_M[i+1][j+1],
            hmm.log_trans("X", "X") + hmm.log_emit_X(x_{i+1}) + B_X[i+1][j],
            hmm.log_trans("X", "Y") + hmm.log_emit_Y(y_{j+1}) + B_Y[i][j+1],
        ])

        B_Y[i][j] = logsumexp([
            hmm.log_trans("Y", "M") + hmm.log_emit_M(x_{i+1}, y_{j+1}) + B_M[i+1][j+1],
            hmm.log_trans("Y", "X") + hmm.log_emit_X(x_{i+1}) + B_X[i+1][j],
            hmm.log_trans("Y", "Y") + hmm.log_emit_Y(y_{j+1}) + B_Y[i][j+1],
        ])

    Termination:
        logZ = logsumexp([
            hmm.start_log_probs["M"] + hmm.log_emit_M(x_1, y_1) + B_M[1][1],
            hmm.start_log_probs["X"] + hmm.log_emit_X(x_1) + B_X[1][0],
            hmm.start_log_probs["Y"] + hmm.log_emit_Y(y_1) + B_Y[0][1],
        ])

    TODO:
        Implement the backward recurrence mirroring the forward algorithm.
    """
    raise NotImplementedError(
        "TODO: implement backward algorithm using formulas above."
    )


__all__ = ["logsumexp", "compute_forward", "compute_backward"]

"""Forward/backward dynamic programming scaffolding for the Pair HMM.

This module assumes a 3-state pair-HMM with states:
    - M: match/mismatch (consumes one symbol from x and one from y)
    - X: insertion in X / gap in Y (consumes one symbol from x only)
    - Y: insertion in Y / gap in X (consumes one symbol from y only)

All dynamic programming is done in LOG-SPACE.
"""

from __future__ import annotations

import math
from typing import List, Tuple

from src.algorithms.hmm import PairHMM
from src.types.sequence import SequenceType


def logsumexp(values: List[float]) -> float:
    """Compute log(sum(exp(values))) in a numerically stable way.

    Returns -inf if the list is empty or all entries are -inf.
    """
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
    """Compute forward DP matrices for the Pair HMM in log-space.

    See module docstring and function docstring in template for detailed
    semantics and recurrences.
    """
    n = len(x_seq)
    m = len(y_seq)
    x = x_seq.residues
    y = y_seq.residues

    # Allocate and initialize DP tables with -inf
    F_M = [[float("-inf")] * (m + 1) for _ in range(n + 1)]
    F_X = [[float("-inf")] * (m + 1) for _ in range(n + 1)]
    F_Y = [[float("-inf")] * (m + 1) for _ in range(n + 1)]

    # Short-circuit corner case: empty sequences (no emissions at all)
    if n == 0 and m == 0:
        # No emissions; probability mass would come from start/end only.
        # We keep DP tables as -inf and logZ as -inf (no path emitting anything).
        logZ = float("-inf")
        return F_M, F_X, F_Y, logZ

    start = hmm.start_log_probs
    end = hmm.end_log_probs

    # --- Initialization from start distribution ---

    # First insertion in X at (1, 0)
    if n > 0:
        F_X[1][0] = start["X"] + hmm.log_emit_X(x[0])

    # First insertion in Y at (0, 1)
    if m > 0:
        F_Y[0][1] = start["Y"] + hmm.log_emit_Y(y[0])

    # First match at (1, 1)
    if n > 0 and m > 0:
        F_M[1][1] = start["M"] + hmm.log_emit_M(x[0], y[0])

    # --- Main DP loops ---

    # We iterate in increasing i, j and apply recurrences, being careful
    # not to overwrite the explicitly initialized "start" cells:
    #   (1, 0) for X, (0, 1) for Y, (1, 1) for M.
    for i in range(0, n + 1):
        for j in range(0, m + 1):
            # Skip the pre-start dummy cell
            if i == 0 and j == 0:
                continue

            # --- Match state: consumes x[i-1], y[j-1] ---
            if i > 0 and j > 0:
                if not (i == 1 and j == 1):
                    prev_vals = [
                        F_M[i - 1][j - 1] + hmm.log_trans("M", "M"),
                        F_X[i - 1][j - 1] + hmm.log_trans("X", "M"),
                        F_Y[i - 1][j - 1] + hmm.log_trans("Y", "M"),
                    ]
                    F_M[i][j] = logsumexp(prev_vals)
                    if F_M[i][j] > float("-inf"):
                        F_M[i][j] += hmm.log_emit_M(x[i - 1], y[j - 1])

            # --- Insert X state: consumes x[i-1] only ---
            if i > 0 and not (i == 1 and j == 0):
                prev_vals = [
                    F_M[i - 1][j] + hmm.log_trans("M", "X"),
                    F_X[i - 1][j] + hmm.log_trans("X", "X"),
                    F_Y[i - 1][j] + hmm.log_trans("Y", "X"),
                ]
                F_X[i][j] = logsumexp(prev_vals)
                if F_X[i][j] > float("-inf"):
                    F_X[i][j] += hmm.log_emit_X(x[i - 1])

            # --- Insert Y state: consumes y[j-1] only ---
            if j > 0 and not (i == 0 and j == 1):
                prev_vals = [
                    F_M[i][j - 1] + hmm.log_trans("M", "Y"),
                    F_X[i][j - 1] + hmm.log_trans("X", "Y"),
                    F_Y[i][j - 1] + hmm.log_trans("Y", "Y"),
                ]
                F_Y[i][j] = logsumexp(prev_vals)
                if F_Y[i][j] > float("-inf"):
                    F_Y[i][j] += hmm.log_emit_Y(y[j - 1])

    # --- Termination: sum over ending in any state at (n, m) ---
    logZ = logsumexp(
        [
            F_M[n][m] + end["M"],
            F_X[n][m] + end["X"],
            F_Y[n][m] + end["Y"],
        ]
    )

    return F_M, F_X, F_Y, logZ


def compute_backward(
    hmm: PairHMM, x_seq: SequenceType, y_seq: SequenceType
) -> Tuple[List[List[float]], List[List[float]], List[List[float]], float]:
    """Compute backward DP matrices for the Pair HMM in log-space.

    See module docstring and function docstring in template for detailed
    semantics and recurrences.
    """
    n = len(x_seq)
    m = len(y_seq)
    x = x_seq.residues
    y = y_seq.residues

    # Allocate and initialize DP tables with -inf
    B_M = [[float("-inf")] * (m + 1) for _ in range(n + 1)]
    B_X = [[float("-inf")] * (m + 1) for _ in range(n + 1)]
    B_Y = [[float("-inf")] * (m + 1) for _ in range(n + 1)]

    end = hmm.end_log_probs
    start = hmm.start_log_probs

    # --- Initialization at the end (n, m): only transition to End remains ---
    B_M[n][m] = end["M"]
    B_X[n][m] = end["X"]
    B_Y[n][m] = end["Y"]

    # --- Main DP loops (reverse order) ---
    for i in range(n, -1, -1):
        for j in range(m, -1, -1):
            # Skip the terminal cell (already initialized)
            if i == n and j == m:
                continue

            # --- From state M at (i, j) ---
            terms_M: List[float] = []

            # Next state M: consumes x[i], y[j] and goes to (i+1, j+1)
            if i < n and j < m:
                terms_M.append(
                    hmm.log_trans("M", "M")
                    + hmm.log_emit_M(x[i], y[j])
                    + B_M[i + 1][j + 1]
                )

            # Next state X: consumes x[i], goes to (i+1, j)
            if i < n:
                terms_M.append(
                    hmm.log_trans("M", "X")
                    + hmm.log_emit_X(x[i])
                    + B_X[i + 1][j]
                )

            # Next state Y: consumes y[j], goes to (i, j+1)
            if j < m:
                terms_M.append(
                    hmm.log_trans("M", "Y")
                    + hmm.log_emit_Y(y[j])
                    + B_Y[i][j + 1]
                )

            B_M[i][j] = logsumexp(terms_M)

            # --- From state X at (i, j) ---
            terms_X: List[float] = []

            if i < n and j < m:
                terms_X.append(
                    hmm.log_trans("X", "M")
                    + hmm.log_emit_M(x[i], y[j])
                    + B_M[i + 1][j + 1]
                )
            if i < n:
                terms_X.append(
                    hmm.log_trans("X", "X")
                    + hmm.log_emit_X(x[i])
                    + B_X[i + 1][j]
                )
            if j < m:
                terms_X.append(
                    hmm.log_trans("X", "Y")
                    + hmm.log_emit_Y(y[j])
                    + B_Y[i][j + 1]
                )

            B_X[i][j] = logsumexp(terms_X)

            # --- From state Y at (i, j) ---
            terms_Y: List[float] = []

            if i < n and j < m:
                terms_Y.append(
                    hmm.log_trans("Y", "M")
                    + hmm.log_emit_M(x[i], y[j])
                    + B_M[i + 1][j + 1]
                )
            if i < n:
                terms_Y.append(
                    hmm.log_trans("Y", "X")
                    + hmm.log_emit_X(x[i])
                    + B_X[i + 1][j]
                )
            if j < m:
                terms_Y.append(
                    hmm.log_trans("Y", "Y")
                    + hmm.log_emit_Y(y[j])
                    + B_Y[i][j + 1]
                )

            B_Y[i][j] = logsumexp(terms_Y)

    # --- Termination: sum over possible first states and first emissions ---

    terms_start: List[float] = []

    if n > 0 and m > 0:
        term_M = (
            start["M"]
            + hmm.log_emit_M(x[0], y[0])
            + B_M[1][1]
        )
        terms_start.append(term_M)

    if n > 0:
        term_X = start["X"] + hmm.log_emit_X(x[0]) + B_X[1][0]
        terms_start.append(term_X)

    if m > 0:
        term_Y = start["Y"] + hmm.log_emit_Y(y[0]) + B_Y[0][1]
        terms_start.append(term_Y)

    logZ = logsumexp(terms_start)

    return B_M, B_X, B_Y, logZ


__all__ = ["logsumexp", "compute_forward", "compute_backward"]

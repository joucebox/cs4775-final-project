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

    Let:
        n = len(x_seq)
        m = len(y_seq)

    We allocate three (n+1) x (m+1) matrices:
        F_M[i][j]: log P(being in state M after consuming x[:i], y[:j])
        F_X[i][j]: log P(being in state X after consuming x[:i], y[:j])
        F_Y[i][j]: log P(being in state Y after consuming x[:i], y[:j])

    Indexing convention (IMPORTANT):
        - i ranges from 0..n (prefix length of x)
        - j ranges from 0..m (prefix length of y)
        - When i > 0, the “current” x character for emissions is x_seq[i-1]
        - When j > 0, the “current” y character for emissions is y_seq[j-1]

    State semantics:
        - M(i, j) CONSUMES x[i-1] and y[j-1] (requires i > 0 and j > 0)
        - X(i, j) CONSUMES x[i-1] only (requires i > 0; j unchanged)
        - Y(i, j) CONSUMES y[j-1] only (requires j > 0; i unchanged)

    The HMM provides:
        - hmm.log_trans(s_from, s_to): log a_{s_from -> s_to}
        - hmm.log_emit_M(x_char, y_char): log e_M(x, y)
        - hmm.log_emit_X(x_char): log e_X(x)
        - hmm.log_emit_Y(y_char): log e_Y(y)
        - hmm.start_log_probs[s]: log π_s (probability of FIRST state s)
        - hmm.end_log_probs[s]: log a_{s -> End} (probability of ending from s)

    Initialization (log-space):
        - All F_* are initialized to -inf.
        - There is NO real state at (0,0); that is “before emitting anything”.
        - We create the first usable cells by stepping from the start distribution
          into the appropriate states and emitting the first symbol(s).

        Carefully handle the case where n == 0 or m == 0:
          - If n > 0:
              # first insertion in X at (1, 0)
              F_X[1][0] = hmm.start_log_probs["X"] \
                          + hmm.log_emit_X(x_seq[0])
              # extend along i > 1 on j == 0 using X -> X transitions
          - If m > 0:
              # first insertion in Y at (0, 1)
              F_Y[0][1] = hmm.start_log_probs["Y"] \
                          + hmm.log_emit_Y(y_seq[0])
              # extend along j > 1 on i == 0 using Y -> Y transitions
          - If n > 0 and m > 0:
              # first match at (1, 1)
              F_M[1][1] = hmm.start_log_probs["M"] \
                          + hmm.log_emit_M(x_seq[0], y_seq[0])
              # interior cells will be filled by the recurrence.

        BE CAREFUL:
          - Do NOT read x_seq[-1] or y_seq[-1] when i == 0 or j == 0.
          - When on the first row (i == 0), only Y and possibly M (if i>0,j>0)
            are valid, similarly for the first column.

    Recurrences (for 1 <= i <= n, 1 <= j <= m, respecting state semantics):

        # Match state consumes one symbol from each sequence
        F_M[i][j] = logsumexp([
            F_M[i-1][j-1] + hmm.log_trans("M", "M"),
            F_X[i-1][j-1] + hmm.log_trans("X", "M"),
            F_Y[i-1][j-1] + hmm.log_trans("Y", "M"),
        ]) + hmm.log_emit_M(x_seq[i-1], y_seq[j-1])

        # Insert X state consumes one symbol from x only (i > 0)
        F_X[i][j] = logsumexp([
            F_M[i-1][j] + hmm.log_trans("M", "X"),
            F_X[i-1][j] + hmm.log_trans("X", "X"),
            F_Y[i-1][j] + hmm.log_trans("Y", "X"),
        ]) + hmm.log_emit_X(x_seq[i-1])

        # Insert Y state consumes one symbol from y only (j > 0)
        F_Y[i][j] = logsumexp([
            F_M[i][j-1] + hmm.log_trans("M", "Y"),
            F_X[i][j-1] + hmm.log_trans("X", "Y"),
            F_Y[i][j-1] + hmm.log_trans("Y", "Y"),
        ]) + hmm.log_emit_Y(y_seq[j-1])

        On boundaries (i == 0 or j == 0), only recurrences that stay in
        bounds make sense; invalid predecessors should effectively contribute
        -inf (e.g., F_M[0][j-1] is invalid and should be treated as -inf).

        BE CAREFUL:
          - Guard all array accesses so that you never index i-1 < 0 or j-1 < 0.
          - Equivalently, initialize rows/cols with -inf and only apply
            recurrences where the predecessor indices are valid.

    Termination:
        At the end, we have consumed all characters: (n, m).
        The total log-probability of the sequences is:

            logZ = logsumexp([
                F_M[n][m] + hmm.end_log_probs["M"],
                F_X[n][m] + hmm.end_log_probs["X"],
                F_Y[n][m] + hmm.end_log_probs["Y"],
            ])

        This matches the backward definition if both are implemented correctly.

    Returns:
        F_M, F_X, F_Y, logZ_forward

    NOTE:
        - All values are in log-space.
        - You should be able to verify the implementation later by checking
          that logZ_forward == logZ_backward (within numerical tolerance).
    """
    raise NotImplementedError("TODO: implement forward algorithm using formulas above.")


def compute_backward(
    hmm: PairHMM, x_seq: SequenceType, y_seq: SequenceType
) -> Tuple[List[List[float]], List[List[float]], List[List[float]], float]:
    """Compute backward DP matrices for the Pair HMM in log-space.

    Let:
        n = len(x_seq)
        m = len(y_seq)

    We allocate three (n+1) x (m+1) matrices:
        B_M[i][j]: log P(generating the SUFFIX from (i,j) onward,
                         starting *after* we are in state M at (i,j))
        B_X[i][j]: same, for state X at (i,j)
        B_Y[i][j]: same, for state Y at (i,j)

    Indexing convention is the same as forward:
        - i is prefix length of x (0..n), j is prefix length of y (0..m)
        - State at (i,j) means we have ALREADY consumed x[:i], y[:j].
        - The NEXT emission(s) will involve x_seq[i] (if i < n) and/or
          y_seq[j] (if j < m).

    Semantics of B_s(i,j):
        B_s[i][j] = log P( all remaining emissions and transitions
                           from state s at cell (i,j) until End ),
                    where emissions at (i,j) are NOT included (they
                    were handled in the forward pass at that cell).

    Initialization at the end (i == n, j == m):
        - There are no further emissions; only the transition to End.
        - So for each state s in {M, X, Y}:

            B_s[n][m] = hmm.end_log_probs[s]

        (All other entries will be filled by the recurrence.)

    Recurrences (reverse direction):

        From a state s at (i,j), we consider all possible NEXT states t:

            - t = "M": consumes x[i] and y[j], moves to (i+1, j+1)
                        (only if i < n and j < m)
            - t = "X": consumes x[i], moves to (i+1, j)
                        (only if i < n)
            - t = "Y": consumes y[j], moves to (i, j+1)
                        (only if j < m)

        For each s in {M, X, Y}:

            B_M[i][j] = logsumexp(valid terms among:)
                - hmm.log_trans("M", "M")
                  + hmm.log_emit_M(x_seq[i], y_seq[j])
                  + B_M[i+1][j+1]          # if i < n and j < m
                - hmm.log_trans("M", "X")
                  + hmm.log_emit_X(x_seq[i])
                  + B_X[i+1][j]            # if i < n
                - hmm.log_trans("M", "Y")
                  + hmm.log_emit_Y(y_seq[j])
                  + B_Y[i][j+1]            # if j < m

            B_X[i][j] = logsumexp(valid terms among:)
                - hmm.log_trans("X", "M")
                  + hmm.log_emit_M(x_seq[i], y_seq[j])
                  + B_M[i+1][j+1]          # if i < n and j < m
                - hmm.log_trans("X", "X")
                  + hmm.log_emit_X(x_seq[i])
                  + B_X[i+1][j]            # if i < n
                - hmm.log_trans("X", "Y")
                  + hmm.log_emit_Y(y_seq[j])
                  + B_Y[i][j+1]            # if j < m

            B_Y[i][j] = logsumexp(valid terms among:)
                - hmm.log_trans("Y", "M")
                  + hmm.log_emit_M(x_seq[i], y_seq[j])
                  + B_M[i+1][j+1]          # if i < n and j < m
                - hmm.log_trans("Y", "X")
                  + hmm.log_emit_X(x_seq[i])
                  + B_X[i+1][j]            # if i < n
                - hmm.log_trans("Y", "Y")
                  + hmm.log_emit_Y(y_seq[j])
                  + B_Y[i][j+1]            # if j < m

        BE CAREFUL:
          - Only add a term if its (i+1, j), (i, j+1), or (i+1, j+1)
            indices are in bounds.
          - If n == 0 or m == 0, many of these will be invalid; handle
            those cases explicitly.

        Implementation detail:
          - Typically, you will fill B_* in decreasing i and j order:
              for i in range(n, -1, -1):
                  for j in range(m, -1, -1):
                      # skip (n,m) (already initialized)
                      # apply recurrences elsewhere

    Termination (computing logZ from backward):

        The total log-probability of the sequences is obtained by summing
        over all possible FIRST states and their first emission:

            If n > 0 and m > 0:
                term_M = hmm.start_log_probs["M"] \
                         + hmm.log_emit_M(x_seq[0], y_seq[0]) \
                         + B_M[1][1]
            If n > 0:
                term_X = hmm.start_log_probs["X"] \
                         + hmm.log_emit_X(x_seq[0]) \
                         + B_X[1][0]
            If m > 0:
                term_Y = hmm.start_log_probs["Y"] \
                         + hmm.log_emit_Y(y_seq[0]) \
                         + B_Y[0][1]

            logZ = logsumexp(all valid terms among term_M, term_X, term_Y)

        This should match the logZ computed in the forward pass:

            logZ_forward = logsumexp([
                F_M[n][m] + end_log_probs["M"],
                F_X[n][m] + end_log_probs["X"],
                F_Y[n][m] + end_log_probs["Y"],
            ])

        up to small numerical differences.

    Returns:
        B_M, B_X, B_Y, logZ_backward

    NOTE:
        - All values are in log-space.
        - You should later verify that |logZ_forward - logZ_backward|
          is small (e.g., < 1e-6) on test cases.
    """
    raise NotImplementedError(
        "TODO: implement backward algorithm using formulas above."
    )


__all__ = ["logsumexp", "compute_forward", "compute_backward"]

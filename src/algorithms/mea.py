"""Maximum Expected Accuracy (MEA) algorithm for pairwise sequence alignment."""

from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np

from src.algorithms.base import PairwiseAligner
from src.algorithms.hmm import PairHMM
from src.algorithms.forward_backward import compute_forward, compute_backward
from src.types import AlignmentResult, SequenceType
from src.types.alignment import Alignment


class MEAAligner(PairwiseAligner):
    """Maximum Expected Accuracy (MEA) alignment algorithm."""

    def __init__(self, gamma: float = 1.0) -> None:
        """Initialize the MEA aligner with a configurable gamma parameter.

        Interpretation of gamma:
            - We first compute posterior match probabilities P(M at (i,j) | x,y).
            - The DP reward for aligning i with j is:

                    w[i][j] = (posterior_M[i][j]) ** gamma

              so:
                * gamma > 1 accentuates high-confidence matches,
                * gamma < 1 flattens differences between high/low posteriors,
                * gamma = 1 uses the raw posterior probabilities.
        """
        if gamma <= 0:
            raise ValueError("gamma must be positive.")
        self.gamma = gamma

    def _compute_match_posteriors(
        self,
        hmm: PairHMM,
        x_seq: SequenceType,
        y_seq: SequenceType,
    ) -> np.ndarray:
        """Run forward/backward algorithms and return match posteriors."""
        F_M, _, _, logZ_f = compute_forward(hmm, x_seq, y_seq)
        B_M, _, _, logZ_b = compute_backward(hmm, x_seq, y_seq)

        if logZ_f == float("-inf") or logZ_b == float("-inf"):
            raise ValueError("No valid paths found.")

        logZ = logZ_f
        n = len(x_seq)
        m = len(y_seq)
        post_M = np.zeros((n + 1, m + 1), dtype=float)

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                joint_log = F_M[i][j] + B_M[i][j]
                if joint_log == float("-inf"):
                    post_M[i, j] = 0.0
                else:
                    post_M[i, j] = math.exp(joint_log - logZ)

        return post_M

    def _build_weight_matrix(self, post_M: np.ndarray) -> np.ndarray:
        """Raise posterior matrix to gamma power to form MEA weights."""
        return np.power(post_M, self.gamma)

    def _initialize_dp_tables(
        self,
        n: int,
        m: int,
    ) -> Tuple[List[List[float]], List[List[str | None]]]:
        """Allocate DP score and pointer tables with boundary conditions."""
        dp: List[List[float]] = [[0.0] * (m + 1) for _ in range(n + 1)]
        ptr: List[List[str | None]] = [[None] * (m + 1) for _ in range(n + 1)]

        for i in range(1, n + 1):
            dp[i][0] = dp[i - 1][0]
            ptr[i][0] = "X"

        for j in range(1, m + 1):
            dp[0][j] = dp[0][j - 1]
            ptr[0][j] = "Y"

        return dp, ptr

    def _run_mea_dp(
        self,
        w: np.ndarray,
        dp: List[List[float]],
        ptr: List[List[str | None]],
        n: int,
        m: int,
    ) -> float:
        """Fill DP tables according to the MEA recurrence."""
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                score_match = dp[i - 1][j - 1] + w[i, j]
                score_gap_x = dp[i - 1][j]  # gap in y (X)
                score_gap_y = dp[i][j - 1]  # gap in x (Y)

                best_score = score_match
                best_move = "M"
                if score_gap_x > best_score:
                    best_score = score_gap_x
                    best_move = "X"
                if score_gap_y > best_score:
                    best_score = score_gap_y
                    best_move = "Y"

                dp[i][j] = best_score
                ptr[i][j] = best_move

        return dp[n][m]

    def _traceback_alignment(
        self,
        ptr: List[List[str | None]],
        x: List[str],
        y: List[str],
        n: int,
        m: int,
    ) -> Tuple[List[str], List[str]]:
        """Trace back through pointer table to recover the alignment."""
        aligned_x_chars: List[str] = []
        aligned_y_chars: List[str] = []
        i, j = n, m

        while i > 0 or j > 0:
            move = ptr[i][j]
            if move == "M":
                aligned_x_chars.append(x[i - 1])
                aligned_y_chars.append(y[j - 1])
                i -= 1
                j -= 1
            elif move == "X":
                aligned_x_chars.append(x[i - 1])
                aligned_y_chars.append("-")
                i -= 1
            elif move == "Y":
                aligned_x_chars.append("-")
                aligned_y_chars.append(y[j - 1])
                j -= 1
            else:
                break

        aligned_x_chars.reverse()
        aligned_y_chars.reverse()

        return aligned_x_chars, aligned_y_chars

    def _build_alignment(
        self,
        x_seq: SequenceType,
        y_seq: SequenceType,
        aligned_x_chars: List[str],
        aligned_y_chars: List[str],
    ) -> Alignment:
        """Wrap aligned character lists in an Alignment object."""
        aligned_x_seq = type(x_seq)(
            identifier=x_seq.identifier,
            residues=aligned_x_chars,
            description=x_seq.description,
            aligned=True,
        )
        aligned_y_seq = type(y_seq)(
            identifier=y_seq.identifier,
            residues=aligned_y_chars,
            description=y_seq.description,
            aligned=True,
        )

        return Alignment(
            name=None,
            aligned_sequences=[aligned_x_seq, aligned_y_seq],
            original_sequences=[x_seq.denormalize(), y_seq.denormalize()],
        )

    def align(
        self,
        hmm: PairHMM,
        x_seq: SequenceType,
        y_seq: SequenceType,
    ) -> AlignmentResult:
        """Align two sequences by maximizing expected accuracy."""
        x_seq = x_seq.normalize()
        y_seq = y_seq.normalize()

        n = len(x_seq)
        m = len(y_seq)
        x = x_seq.residues
        y = y_seq.residues

        post_M = self._compute_match_posteriors(hmm, x_seq, y_seq)
        w = self._build_weight_matrix(post_M)
        dp, ptr = self._initialize_dp_tables(n, m)
        score = self._run_mea_dp(w, dp, ptr, n, m)
        aligned_x_chars, aligned_y_chars = self._traceback_alignment(ptr, x, y, n, m)
        alignment = self._build_alignment(
            x_seq, y_seq, aligned_x_chars, aligned_y_chars
        )

        return AlignmentResult(
            alignment=alignment,
            score=score,
            posteriors=post_M,
        )


__all__ = ["MEAAligner"]

"""Maximum Expected Accuracy (MEA) algorithm for pairwise sequence alignment."""

from __future__ import annotations

import math
from typing import List

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

    def align(
        self,
        hmm: PairHMM,
        x_seq: SequenceType,
        y_seq: SequenceType,
    ) -> AlignmentResult:
        """Align two sequences by maximizing expected accuracy."""
        n = len(x_seq)
        m = len(y_seq)
        x = x_seq.residues
        y = y_seq.residues

        # --- 1. Forward/backward to get posteriors on M(i,j) ---

        F_M, _, _, logZ_f = compute_forward(hmm, x_seq, y_seq)
        B_M, _, _, logZ_b = compute_backward(hmm, x_seq, y_seq)

        # If no valid paths, return a trivial alignment with score 0.
        if logZ_f == float("-inf") or logZ_b == float("-inf"):
            alignment = self._build_trivial_alignment(x_seq, y_seq)
            return AlignmentResult(
                alignment=alignment,
                score=0.0,
                posteriors=None,
            )

        # In principle logZ_f and logZ_b should agree; in practice we trust logZ_f.
        logZ = logZ_f

        # Posterior matrix post_M[i][j] for 0..n, 0..m (we only fill 1..n,1..m)
        post_M = np.zeros((n + 1, m + 1), dtype=float)
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                joint_log = F_M[i][j] + B_M[i][j]
                if joint_log == float("-inf"):
                    post_M[i, j] = 0.0
                else:
                    post_M[i, j] = math.exp(joint_log - logZ)

        # --- 2. Build MEA weight matrix w[i][j] = post_M[i][j] ** gamma ---

        w = np.power(post_M, self.gamma)

        # --- 3. MEA DP over w[i][j] ---

        dp: List[List[float]] = [[0.0] * (m + 1) for _ in range(n + 1)]
        # traceback pointers: "M" (match), "X" (gap in y), "Y" (gap in x)
        ptr: List[List[str | None]] = [[None] * (m + 1) for _ in range(n + 1)]

        # Initialize first column: gaps in y (X moves)
        for i in range(1, n + 1):
            dp[i][0] = dp[i - 1][0]
            ptr[i][0] = "X"

        # Initialize first row: gaps in x (Y moves)
        for j in range(1, m + 1):
            dp[0][j] = dp[0][j - 1]
            ptr[0][j] = "Y"

        # Fill interior
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                score_match = dp[i - 1][j - 1] + w[i, j]
                score_gap_x = dp[i - 1][j]  # gap in y (X)
                score_gap_y = dp[i][j - 1]  # gap in x (Y)

                # Choose best; deterministic tie-breaking: M > X > Y
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

        score = dp[n][m]

        # --- 4. Traceback to build aligned sequences ---

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
                # Should only happen at (0,0), but guard defensively.
                break

        aligned_x_chars.reverse()
        aligned_y_chars.reverse()

        # --- 5. Wrap into Alignment + AlignmentResult ---

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

        alignment = Alignment(
            name=None,
            aligned_sequences=[aligned_x_seq, aligned_y_seq],
            original_sequences=[x_seq, y_seq],
        )

        return AlignmentResult(
            alignment=alignment,
            score=score,
            posteriors=post_M,
        )

    @staticmethod
    def _build_trivial_alignment(
        x_seq: SequenceType,
        y_seq: SequenceType,
    ) -> Alignment:
        """Fallback alignment when logZ is -inf: align everything against gaps.

        This should rarely happen in sane parameter settings, but provides
        a defined object rather than raising.
        """
        x = x_seq.residues
        y = y_seq.residues

        aligned_x: List[str] = []
        aligned_y: List[str] = []

        # Simple global-style: x against gaps, then gaps against y
        for ch in x:
            aligned_x.append(ch)
            aligned_y.append("-")
        for ch in y:
            aligned_x.append("-")
            aligned_y.append(ch)

        aligned_x_seq = type(x_seq)(
            identifier=x_seq.identifier,
            residues=aligned_x,
            description=x_seq.description,
            aligned=True,
        )
        aligned_y_seq = type(y_seq)(
            identifier=y_seq.identifier,
            residues=aligned_y,
            description=y_seq.description,
            aligned=True,
        )

        return Alignment(
            name=None,
            aligned_sequences=[aligned_x_seq, aligned_y_seq],
            original_sequences=[x_seq, y_seq],
        )


__all__ = ["MEAAligner"]
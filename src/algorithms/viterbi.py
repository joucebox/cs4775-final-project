"""Viterbi alignment implementation for the affine-gap pair-HMM."""

from __future__ import annotations

from typing import List, Optional, Tuple

from src.algorithms.base import PairwiseAligner
from src.algorithms.hmm import PairHMM
from src.types import Alignment, AlignmentResult, SequenceType

NEG_INF = float("-inf")
VITERBI_STATES = ("M", "X", "Y")


class ViterbiAligner(PairwiseAligner):
    """Maximum a posteriori alignment using the Viterbi algorithm."""

    def _initialize_dp_matrices(
        self, n: int, m: int
    ) -> Tuple[List[List[float]], List[List[float]], List[List[float]]]:
        """Allocate DP tables for states M, X, Y and seed with -inf."""
        V_M = [[NEG_INF] * (m + 1) for _ in range(n + 1)]
        V_X = [[NEG_INF] * (m + 1) for _ in range(n + 1)]
        V_Y = [[NEG_INF] * (m + 1) for _ in range(n + 1)]
        return V_M, V_X, V_Y

    def _initialize_backpointers(self, n: int, m: int) -> Tuple[
        List[List[Optional[str]]],
        List[List[Optional[str]]],
        List[List[Optional[str]]],
    ]:
        """Prepare backpointer grids for traceback."""
        Psi_M = [[None] * (m + 1) for _ in range(n + 1)]
        Psi_X = [[None] * (m + 1) for _ in range(n + 1)]
        Psi_Y = [[None] * (m + 1) for _ in range(n + 1)]
        return Psi_M, Psi_X, Psi_Y

    def _initialize_from_start(
        self,
        V_M: List[List[float]],
        V_X: List[List[float]],
        V_Y: List[List[float]],
        hmm: PairHMM,
        x: List[str],
        y: List[str],
        n: int,
        m: int,
    ) -> None:
        """Seed the first reachable cells from the start distribution."""
        if n > 0:
            V_X[1][0] = hmm.start_log_probs["X"] + hmm.log_emit_X(x[0])
        if m > 0:
            V_Y[0][1] = hmm.start_log_probs["Y"] + hmm.log_emit_Y(y[0])
        if n > 0 and m > 0:
            V_M[1][1] = hmm.start_log_probs["M"] + hmm.log_emit_M(x[0], y[0])

    def _fill_boundaries(
        self,
        V_X: List[List[float]],
        V_Y: List[List[float]],
        Psi_X: List[List[Optional[str]]],
        Psi_Y: List[List[Optional[str]]],
        hmm: PairHMM,
        x: List[str],
        y: List[str],
        n: int,
        m: int,
    ) -> None:
        """Extend pure gap runs along the first column and first row."""
        for i in range(2, n + 1):
            if V_X[i - 1][0] == NEG_INF:
                continue
            candidate = V_X[i - 1][0] + hmm.log_trans("X", "X")
            if candidate == NEG_INF:
                continue
            V_X[i][0] = candidate + hmm.log_emit_X(x[i - 1])
            Psi_X[i][0] = "X"

        for j in range(2, m + 1):
            if V_Y[0][j - 1] == NEG_INF:
                continue
            candidate = V_Y[0][j - 1] + hmm.log_trans("Y", "Y")
            if candidate == NEG_INF:
                continue
            V_Y[0][j] = candidate + hmm.log_emit_Y(y[j - 1])
            Psi_Y[0][j] = "Y"

    def _fill_interior(
        self,
        V_M: List[List[float]],
        V_X: List[List[float]],
        V_Y: List[List[float]],
        Psi_M: List[List[Optional[str]]],
        Psi_X: List[List[Optional[str]]],
        Psi_Y: List[List[Optional[str]]],
        hmm: PairHMM,
        x: List[str],
        y: List[str],
        n: int,
        m: int,
    ) -> None:
        """Run Viterbi recurrences through the interior of the grid."""
        for i in range(0, n + 1):
            for j in range(0, m + 1):
                if i == 0 and j == 0:
                    continue

                if i > 0 and j > 0 and not (i == 1 and j == 1):
                    candidates = (
                        V_M[i - 1][j - 1] + hmm.log_trans("M", "M"),
                        V_X[i - 1][j - 1] + hmm.log_trans("X", "M"),
                        V_Y[i - 1][j - 1] + hmm.log_trans("Y", "M"),
                    )
                    max_val = max(candidates)
                    if max_val != NEG_INF:
                        best_prev = VITERBI_STATES[candidates.index(max_val)]
                        V_M[i][j] = max_val + hmm.log_emit_M(x[i - 1], y[j - 1])
                        Psi_M[i][j] = best_prev

                if i > 0 and not (i == 1 and j == 0):
                    candidates = (
                        V_M[i - 1][j] + hmm.log_trans("M", "X"),
                        V_X[i - 1][j] + hmm.log_trans("X", "X"),
                    )
                    max_val = max(candidates)
                    if max_val != NEG_INF:
                        best_prev = ("M", "X")[candidates.index(max_val)]
                        V_X[i][j] = max_val + hmm.log_emit_X(x[i - 1])
                        Psi_X[i][j] = best_prev

                if j > 0 and not (i == 0 and j == 1):
                    candidates = (
                        V_M[i][j - 1] + hmm.log_trans("M", "Y"),
                        V_Y[i][j - 1] + hmm.log_trans("Y", "Y"),
                    )
                    max_val = max(candidates)
                    if max_val != NEG_INF:
                        best_prev = ("M", "Y")[candidates.index(max_val)]
                        V_Y[i][j] = max_val + hmm.log_emit_Y(y[j - 1])
                        Psi_Y[i][j] = best_prev

    def _compute_termination(
        self,
        V_M: List[List[float]],
        V_X: List[List[float]],
        V_Y: List[List[float]],
        hmm: PairHMM,
        n: int,
        m: int,
    ) -> Tuple[float, str]:
        """Determine best ending state at (n, m) including end distribution."""
        end_vals = {
            "M": V_M[n][m] + hmm.end_log_probs["M"],
            "X": V_X[n][m] + hmm.end_log_probs["X"],
            "Y": V_Y[n][m] + hmm.end_log_probs["Y"],
        }
        best_state = max(end_vals, key=end_vals.get)
        return end_vals[best_state], best_state

    def _traceback(
        self,
        Psi_M: List[List[Optional[str]]],
        Psi_X: List[List[Optional[str]]],
        Psi_Y: List[List[Optional[str]]],
        x_seq: SequenceType,
        y_seq: SequenceType,
        best_state: str,
    ) -> Alignment:
        """Follow backpointers to reconstruct the optimal alignment path."""
        i, j = len(x_seq), len(y_seq)
        state = best_state
        aligned_x: List[str] = []
        aligned_y: List[str] = []

        while i > 0 or j > 0:
            if state == "M":
                aligned_x.append(x_seq.residues[i - 1])
                aligned_y.append(y_seq.residues[j - 1])
                prev_state = Psi_M[i][j]
                i -= 1
                j -= 1
            elif state == "X":
                aligned_x.append(x_seq.residues[i - 1])
                aligned_y.append("-")
                prev_state = Psi_X[i][j]
                i -= 1
            else:  # state == "Y"
                aligned_x.append("-")
                aligned_y.append(y_seq.residues[j - 1])
                prev_state = Psi_Y[i][j]
                j -= 1

            if prev_state is None:
                state = "M"
                continue
            state = prev_state

        aligned_x.reverse()
        aligned_y.reverse()

        seq_cls = type(x_seq)
        aligned_seq_x = seq_cls(
            identifier=x_seq.identifier,
            residues=aligned_x,
            description=x_seq.description,
            aligned=True,
        )
        seq_cls_y = type(y_seq)
        aligned_seq_y = seq_cls_y(
            identifier=y_seq.identifier,
            residues=aligned_y,
            description=y_seq.description,
            aligned=True,
        )

        return Alignment(
            name=f"Viterbi_{x_seq.identifier}_vs_{y_seq.identifier}",
            aligned_sequences=[aligned_seq_x, aligned_seq_y],
            original_sequences=[x_seq.denormalize(), y_seq.denormalize()],
        )

    def align(
        self,
        hmm: PairHMM,
        x_seq: SequenceType,
        y_seq: SequenceType,
    ) -> AlignmentResult:
        """Compute the Viterbi alignment for the provided sequences."""
        x_seq = x_seq.normalize()
        y_seq = y_seq.normalize()

        n = len(x_seq)
        m = len(y_seq)
        x = x_seq.residues
        y = y_seq.residues

        V_M, V_X, V_Y = self._initialize_dp_matrices(n, m)
        Psi_M, Psi_X, Psi_Y = self._initialize_backpointers(n, m)

        self._initialize_from_start(V_M, V_X, V_Y, hmm, x, y, n, m)
        self._fill_boundaries(V_X, V_Y, Psi_X, Psi_Y, hmm, x, y, n, m)
        self._fill_interior(V_M, V_X, V_Y, Psi_M, Psi_X, Psi_Y, hmm, x, y, n, m)

        score, best_state = self._compute_termination(V_M, V_X, V_Y, hmm, n, m)
        alignment = self._traceback(Psi_M, Psi_X, Psi_Y, x_seq, y_seq, best_state)

        return AlignmentResult(alignment=alignment, score=score, posteriors=None)


__all__ = ["ViterbiAligner"]

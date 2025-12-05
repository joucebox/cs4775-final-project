"""Maximum Expected Accuracy (MEA) algorithm for pairwise sequence alignment."""

from __future__ import annotations

import math
from typing import Callable, List, Literal, Tuple

import numpy as np

from src.algorithms.base import PairwiseAligner
from src.algorithms.hmm import PairHMM
from src.algorithms.forward_backward import compute_forward, compute_backward
from src.types import AlignmentResult, SequenceType
from src.types.alignment import Alignment

# Type alias for weight matrix functions
WeightFunction = Callable[[np.ndarray, float], np.ndarray]
MEAMethod = Literal["power", "threshold", "probcons", "log_odds"]


def weight_power(post_M: np.ndarray, gamma: float) -> np.ndarray:
    """Power-based MEA weights (original formulation).

    Formula: w[i,j] = P_M(i,j) ^ gamma

    Raises posteriors to the power of gamma:
        - gamma > 1: accentuates high-confidence matches
        - gamma < 1: flattens differences between posteriors
        - gamma = 1: raw posteriors (no transformation)

    Note: No threshold effect - all positive posteriors contribute.

    Args:
        post_M: Posterior probability matrix (n+1 x m+1).
        gamma: Exponent parameter, must be > 0.

    Returns:
        Weight matrix of same shape.
    """
    return np.power(post_M, gamma)


def weight_threshold(post_M: np.ndarray, gamma: float) -> np.ndarray:
    """Threshold-based MEA weights.

    Formula: w[i,j] = P_M(i,j) - (1 - gamma)

    Match is beneficial when P_M(i,j) > (1 - gamma):
        - gamma = 1.0: threshold > 0 (high recall)
        - gamma = 0.5: threshold > 0.5 (balanced)
        - gamma = 0.1: threshold > 0.9 (high precision)

    Args:
        post_M: Posterior probability matrix (n+1 x m+1).
        gamma: Threshold parameter in (0, 1].

    Returns:
        Weight matrix of same shape.
    """
    return post_M - (1.0 - gamma)


def weight_probcons(post_M: np.ndarray, gamma: float) -> np.ndarray:
    """ProbCons-style MEA weights (Do et al., 2005).

    Formula: w[i,j] = 2 * gamma * P_M(i,j) - 1

    Match is beneficial when P_M(i,j) > 1 / (2 * gamma):
        - gamma = 1.0: threshold > 0.5
        - gamma = 0.75: threshold > 0.67
        - gamma = 0.5: threshold > 1.0 (no matches)

    Note: Requires gamma > 0.5 to produce any matches.

    Args:
        post_M: Posterior probability matrix (n+1 x m+1).
        gamma: Scaling parameter, typically > 0.5.

    Returns:
        Weight matrix of same shape.
    """
    return 2.0 * gamma * post_M - 1.0


def weight_log_odds(post_M: np.ndarray, gamma: float) -> np.ndarray:
    """Log-odds MEA weights (Bradley et al., 2009 - FSA).

    Formula: w[i,j] = log(P / (1-P)) + log(gamma / (1-gamma))

    Interprets gamma as prior probability of match. Numerically stable
    for extreme posteriors.

    Match is beneficial when P_M(i,j) > (1 - gamma):
        - gamma = 0.9: threshold > 0.1
        - gamma = 0.5: threshold > 0.5
        - gamma = 0.1: threshold > 0.9

    Args:
        post_M: Posterior probability matrix (n+1 x m+1).
        gamma: Prior match probability in (0, 1).

    Returns:
        Weight matrix of same shape.
    """
    eps = 1e-10  # Numerical stability
    P = np.clip(post_M, eps, 1.0 - eps)
    log_odds_posterior = np.log(P / (1.0 - P))
    log_odds_prior = math.log(gamma / (1.0 - gamma))
    return log_odds_posterior + log_odds_prior


# Registry of weight functions
WEIGHT_FUNCTIONS: dict[MEAMethod, WeightFunction] = {
    "power": weight_power,
    "threshold": weight_threshold,
    "probcons": weight_probcons,
    "log_odds": weight_log_odds,
}


class MEAAligner(PairwiseAligner):
    """Maximum Expected Accuracy (MEA) alignment algorithm.

    Supports four formulations for the weight matrix:
        - "power": w = P^gamma (default, original formulation)
        - "threshold": w = P - (1 - gamma), match when P > (1 - gamma)
        - "probcons": w = 2*gamma*P - 1, match when P > 1/(2*gamma)
        - "log_odds": w = log(P/(1-P)) + log(gamma/(1-gamma))
    """

    def __init__(self, gamma: float = 1.0, method: MEAMethod = "power") -> None:
        """Initialize the MEA aligner.

        Args:
            gamma: Parameter controlling alignment behavior.
                   For "power": gamma > 0 (exponent, 1.0 = raw posteriors).
                   For "threshold" and "log_odds": gamma in (0, 1].
                   For "probcons": gamma > 0 (but > 0.5 needed for matches).
            method: Weight function ("power", "threshold", "probcons", "log_odds").
        """
        if method not in WEIGHT_FUNCTIONS:
            raise ValueError(
                f"Unknown method: {method}. Choose from {list(WEIGHT_FUNCTIONS.keys())}"
            )

        if method in ("threshold", "log_odds"):
            if gamma <= 0 or gamma > 1:
                raise ValueError(f"gamma must be in (0, 1] for method '{method}'.")
        else:  # power, probcons
            if gamma <= 0:
                raise ValueError(f"gamma must be positive for method '{method}'.")

        self.gamma = gamma
        self.method = method
        self._weight_fn = WEIGHT_FUNCTIONS[method]

    def _compute_match_posteriors(
        self,
        hmm: PairHMM,
        x_seq: SequenceType,
        y_seq: SequenceType,
    ) -> np.ndarray:
        """Run forward/backward algorithms and return match posteriors."""
        F_M, _, _, logZ_f = compute_forward(hmm, x_seq, y_seq)
        B_M, _, _, logZ_b = compute_backward(hmm, x_seq, y_seq)

        if math.isfinite(logZ_f) and math.isfinite(logZ_b):
            if abs(logZ_f - logZ_b) > 1e-5:
                raise ValueError(
                    f"Forward and backward log-normalizers disagree: {logZ_f} vs {logZ_b}"
                )
            logZ = (logZ_f + logZ_b) * 0.5
        else:
            raise ValueError(
                f"Forward and backward log-normalizers are not finite: {logZ_f} vs {logZ_b}"
            )

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
        """Build MEA weight matrix using the selected method."""
        return self._weight_fn(post_M, self.gamma)

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


__all__ = [
    "MEAAligner",
    "MEAMethod",
    "weight_power",
    "weight_threshold",
    "weight_probcons",
    "weight_log_odds",
]

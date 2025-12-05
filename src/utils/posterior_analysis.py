"""Posterior probability analysis utilities for pairwise alignments."""

from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import mean
from typing import Dict, Optional, Tuple

import numpy as np

from src.algorithms.forward_backward import compute_forward, compute_backward
from src.algorithms.hmm import PairHMM
from src.algorithms.mea import MEAAligner, MEAMethod
from src.algorithms.viterbi import ViterbiAligner
from src.types import SequenceType


class PosteriorCache:
    """Cache for posterior matrices, Viterbi, and MEA alignment pairs."""

    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.posteriors_dir = cache_dir / "posteriors"
        self.viterbi_dir = cache_dir / "viterbi_pairs"
        self.mea_dir = cache_dir / "mea_pairs"

    def _ensure_dirs(self) -> None:
        self.posteriors_dir.mkdir(parents=True, exist_ok=True)
        self.viterbi_dir.mkdir(parents=True, exist_ok=True)
        self.mea_dir.mkdir(parents=True, exist_ok=True)

    def _safe_id(self, pair_id: str) -> str:
        return pair_id.replace("/", "_")

    def _posterior_path(self, pair_id: str) -> Path:
        return self.posteriors_dir / f"{self._safe_id(pair_id)}.npy"

    def _viterbi_path(self, pair_id: str) -> Path:
        return self.viterbi_dir / f"{self._safe_id(pair_id)}.json"

    def _mea_path(
        self, pair_id: str, gamma: float, method: MEAMethod = "power"
    ) -> Path:
        return (
            self.mea_dir / f"{self._safe_id(pair_id)}_{method}_gamma_{gamma:.4f}.json"
        )

    def _load_pairs(self, path: Path) -> Optional[set[Tuple[int, int]]]:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                pairs_list = json.load(f)
            return {tuple(p) for p in pairs_list}
        return None

    def _save_pairs(self, path: Path, pairs: set[Tuple[int, int]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(list(pairs), f)

    def load_posteriors(self, pair_id: str) -> Optional[np.ndarray]:
        path = self._posterior_path(pair_id)
        if path.exists():
            return np.load(path)
        return None

    def save_posteriors(self, pair_id: str, post_M: np.ndarray) -> None:
        self._ensure_dirs()
        np.save(self._posterior_path(pair_id), post_M)

    def load_viterbi_pairs(self, pair_id: str) -> Optional[set[Tuple[int, int]]]:
        return self._load_pairs(self._viterbi_path(pair_id))

    def save_viterbi_pairs(self, pair_id: str, pairs: set[Tuple[int, int]]) -> None:
        self._ensure_dirs()
        self._save_pairs(self._viterbi_path(pair_id), pairs)

    def load_mea_pairs(
        self, pair_id: str, gamma: float, method: MEAMethod = "power"
    ) -> Optional[set[Tuple[int, int]]]:
        return self._load_pairs(self._mea_path(pair_id, gamma, method))

    def save_mea_pairs(
        self,
        pair_id: str,
        gamma: float,
        pairs: set[Tuple[int, int]],
        method: MEAMethod = "power",
    ) -> None:
        self._ensure_dirs()
        self._save_pairs(self._mea_path(pair_id, gamma, method), pairs)

    def get_or_compute_posteriors(
        self,
        pair_id: str,
        hmm: PairHMM,
        x_seq: SequenceType,
        y_seq: SequenceType,
    ) -> np.ndarray:
        """Load posteriors from cache or compute and save."""
        cached = self.load_posteriors(pair_id)
        if cached is not None:
            return cached
        post_M = compute_posteriors(hmm, x_seq, y_seq)
        self.save_posteriors(pair_id, post_M)
        return post_M

    def get_or_compute_viterbi(
        self,
        pair_id: str,
        hmm: PairHMM,
        x_seq: SequenceType,
        y_seq: SequenceType,
    ) -> set[Tuple[int, int]]:
        """Load Viterbi pairs from cache or compute and save."""
        cached = self.load_viterbi_pairs(pair_id)
        if cached is not None:
            return cached
        aligner = ViterbiAligner()
        result = aligner.align(hmm, x_seq, y_seq)
        pairs = extract_alignment_pairs(
            result.alignment.aligned_sequences[0],
            result.alignment.aligned_sequences[1],
        )
        self.save_viterbi_pairs(pair_id, pairs)
        return pairs

    def get_or_compute_mea(
        self,
        pair_id: str,
        gamma: float,
        hmm: PairHMM,
        x_seq: SequenceType,
        y_seq: SequenceType,
        method: MEAMethod = "power",
    ) -> set[Tuple[int, int]]:
        """Load MEA pairs from cache or compute and save.

        Args:
            pair_id: Unique identifier for the sequence pair.
            gamma: Gamma parameter for MEA.
            hmm: The pair HMM model.
            x_seq: First sequence.
            y_seq: Second sequence.
            method: MEA weight function ("power", "threshold", "probcons", "log_odds").

        Returns:
            Set of aligned position pairs.
        """
        cached = self.load_mea_pairs(pair_id, gamma, method)
        if cached is not None:
            return cached
        aligner = MEAAligner(gamma=gamma, method=method)
        result = aligner.align(hmm, x_seq, y_seq)
        pairs = extract_alignment_pairs(
            result.alignment.aligned_sequences[0],
            result.alignment.aligned_sequences[1],
        )
        self.save_mea_pairs(pair_id, gamma, pairs, method)
        return pairs


def compute_posteriors(
    hmm: PairHMM,
    x_seq: SequenceType,
    y_seq: SequenceType,
) -> np.ndarray:
    """Compute posterior match probability matrix for a pair of sequences.

    Uses the forward-backward algorithm to compute P(M_{i,j} | x, y) for all
    positions (i, j), representing the posterior probability that positions
    i and j are aligned in a match state.

    Args:
        hmm: The pair HMM model.
        x_seq: First sequence (will be normalized if not already).
        y_seq: Second sequence (will be normalized if not already).

    Returns:
        np.ndarray of shape (n+1, m+1) where n=len(x_seq), m=len(y_seq).
        post_M[i, j] is the posterior probability of matching position i with j.
        Row/col 0 are padding (always 0).

    Raises:
        ValueError: If forward and backward log-normalizers disagree or are not finite.
    """
    # Normalize sequences if needed
    if getattr(x_seq, "normalized", False) is False:
        x_seq = x_seq.normalize()
    if getattr(y_seq, "normalized", False) is False:
        y_seq = y_seq.normalize()

    F_M, _, _, logZ_f = compute_forward(hmm, x_seq, y_seq)
    B_M, _, _, logZ_b = compute_backward(hmm, x_seq, y_seq)

    if math.isfinite(logZ_f) and math.isfinite(logZ_b):
        if abs(logZ_f - logZ_b) > 1e-5:
            raise ValueError(
                f"Forward and backward log-normalizers disagree: {logZ_f} vs {logZ_b}"
            )
        logZ = 0.5 * (logZ_f + logZ_b)
    else:
        raise ValueError("Forward/backward log-normalizers are not finite")

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


def extract_alignment_pairs(
    aligned_seq1: SequenceType,
    aligned_seq2: SequenceType,
) -> set[Tuple[int, int]]:
    """Extract 0-based aligned residue index pairs from two aligned sequences.

    Walks through two aligned sequences and records which original residue
    positions are aligned (both non-gap).

    Args:
        aligned_seq1: First aligned sequence (with gaps).
        aligned_seq2: Second aligned sequence (with gaps).

    Returns:
        Set of (i, j) tuples where i is the 0-based position in the original
        first sequence and j is the 0-based position in the original second
        sequence, representing aligned pairs.
    """
    pairs: set[Tuple[int, int]] = set()
    cur_i = 0
    cur_j = 0
    for c1, c2 in zip(aligned_seq1.residues, aligned_seq2.residues):
        is_res1 = c1 != "-"
        is_res2 = c2 != "-"
        if is_res1 and is_res2:
            pairs.add((cur_i, cur_j))
        if is_res1:
            cur_i += 1
        if is_res2:
            cur_j += 1
    return pairs


def compute_posterior_metrics(
    post_M: np.ndarray,
    mea_pairs: set[Tuple[int, int]],
    vit_pairs: set[Tuple[int, int]],
) -> Dict[str, float]:
    """Compute comprehensive posterior mass metrics comparing MEA and Viterbi alignments.

    Args:
        post_M: Posterior probability matrix of shape (n+1, m+1).
        mea_pairs: Set of (i, j) tuples for MEA alignment.
        vit_pairs: Set of (i, j) tuples for Viterbi alignment.

    Returns:
        Dictionary with the following metrics:
        - total_mass: Sum of all posterior probabilities
        - mass_mea: Total posterior mass captured by MEA
        - mass_vit: Total posterior mass captured by Viterbi
        - delta_mass: mass_mea - mass_vit
        - delta_mass_pct: (mass_mea - mass_vit) / total_mass * 100
        - num_mea, num_vit: Number of aligned pairs
        - num_shared: Number of pairs in common
        - mass_shared: Posterior mass of shared pairs
        - mass_mea_only: Posterior mass of MEA-only pairs
        - mass_vit_only: Posterior mass of Viterbi-only pairs
        - mean_post_mea, mean_post_vit: Average posterior per aligned pair
        - efficiency_mea, efficiency_vit: mass / num_pairs (posterior per position)
        - jaccard: Jaccard similarity between alignments
    """
    n, m = post_M.shape[0] - 1, post_M.shape[1] - 1

    def _get_mass(pairs: set[Tuple[int, int]]) -> float:
        """Sum posterior probabilities for a set of pairs."""
        total = 0.0
        for i, j in pairs:
            if 0 <= i < n and 0 <= j < m:
                total += float(post_M[i + 1, j + 1])
        return total

    def _get_posteriors(pairs: set[Tuple[int, int]]) -> list[float]:
        """Get list of posterior values for a set of pairs."""
        vals = []
        for i, j in pairs:
            if 0 <= i < n and 0 <= j < m:
                vals.append(float(post_M[i + 1, j + 1]))
        return vals

    # Partition the pairs
    shared = mea_pairs & vit_pairs
    mea_only = mea_pairs - vit_pairs
    vit_only = vit_pairs - mea_pairs
    union = mea_pairs | vit_pairs

    # Counts
    num_mea = len(mea_pairs)
    num_vit = len(vit_pairs)
    num_shared = len(shared)
    num_union = len(union)

    # Mass calculations
    total_mass = float(np.sum(post_M[1:, 1:]))
    mass_mea = _get_mass(mea_pairs)
    mass_vit = _get_mass(vit_pairs)
    mass_shared = _get_mass(shared)
    mass_mea_only = _get_mass(mea_only)
    mass_vit_only = _get_mass(vit_only)

    # Delta calculations
    delta_mass = mass_mea - mass_vit
    delta_mass_pct = (delta_mass / total_mass * 100) if total_mass > 0 else float("nan")

    # Mean posteriors
    mea_posteriors = _get_posteriors(mea_pairs)
    vit_posteriors = _get_posteriors(vit_pairs)
    mean_post_mea = mean(mea_posteriors) if mea_posteriors else float("nan")
    mean_post_vit = mean(vit_posteriors) if vit_posteriors else float("nan")

    # Efficiency (mass per aligned pair)
    efficiency_mea = mass_mea / num_mea if num_mea > 0 else float("nan")
    efficiency_vit = mass_vit / num_vit if num_vit > 0 else float("nan")

    # Jaccard similarity
    jaccard = num_shared / num_union if num_union > 0 else float("nan")

    return {
        "total_mass": total_mass,
        "mass_mea": mass_mea,
        "mass_vit": mass_vit,
        "delta_mass": delta_mass,
        "delta_mass_pct": delta_mass_pct,
        "num_mea": num_mea,
        "num_vit": num_vit,
        "num_shared": num_shared,
        "mass_shared": mass_shared,
        "mass_mea_only": mass_mea_only,
        "mass_vit_only": mass_vit_only,
        "mean_post_mea": mean_post_mea,
        "mean_post_vit": mean_post_vit,
        "efficiency_mea": efficiency_mea,
        "efficiency_vit": efficiency_vit,
        "jaccard": jaccard,
    }


__all__ = [
    "compute_posteriors",
    "extract_alignment_pairs",
    "compute_posterior_metrics",
    "PosteriorCache",
    "MEAMethod",
]

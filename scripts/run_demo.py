#!/usr/bin/env python3
"""Run the Viterbi aligner on a single Stockholm alignment and print results."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Tuple

from .constants import ALIGNMENTS_FOLDER, HMM_YAML

# Ensure repository modules are importable when invoked as a script
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.algorithms.viterbi import ViterbiAligner
from src.algorithms.mea import MEAAligner
from src.types import SequenceType
from src.utils.stockholm import read_rna_stockholm
from src.utils import load_pair_hmm


def format_alignment(seqs: Tuple[SequenceType, SequenceType]) -> str:
    """Return a human-readable two-line alignment string."""
    lines = []
    for seq in seqs:
        residues = "".join(seq.residues)
        lines.append(f"{seq.identifier:>20}: {residues}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Viterbi alignment on a Stockholm example."
    )
    parser.add_argument(
        "-s",
        "--stockholm",
        type=str,
        required=True,
        help="Stockholm (.sto) file containing the reference alignment.",
    )
    args = parser.parse_args()

    stockholm_path = ALIGNMENTS_FOLDER / f"{args.stockholm}.sto"

    if not stockholm_path.exists():
        raise FileNotFoundError(f"Stockholm file not found: {stockholm_path}")

    # read_rna_stockholm now returns a list of pairwise alignments
    alignments = read_rna_stockholm(str(stockholm_path))
    if not alignments:
        raise ValueError("No pairwise alignments found in Stockholm file.")

    # Use the first pairwise alignment by default
    reference = alignments[0]

    print(f"Found {len(alignments)} pairwise alignment(s) in Stockholm file.")
    print(f"Using alignment: {reference.name}\n")

    hmm = load_pair_hmm(HMM_YAML)
    aligner = ViterbiAligner()

    seq_x, seq_y = reference.original_sequences
    result = aligner.align(hmm, seq_x, seq_y)
    viterbi_alignment = result.alignment.aligned_sequences

    mea_aligner = MEAAligner()
    mea_result = mea_aligner.align(hmm, seq_x, seq_y)
    mea_alignment = mea_result.alignment.aligned_sequences

    print("Reference alignment (Stockholm):")
    print(format_alignment(tuple(reference.aligned_sequences)))

    print("\n ======================== Viterbi ========================")
    print("\nViterbi alignment:")
    print(format_alignment(tuple(viterbi_alignment)))
    print(f"\nViterbi log-score: {result.score:.6f}")
    print(f"\nViterbi posteriors: {result.posteriors}")

    print("\n ======================== MEA ========================")
    print("\nMEA alignment:")
    print(format_alignment(tuple(mea_alignment)))
    print(f"\nMEA log-score: {mea_result.score:.6f}")
    print(f"\nMEA posteriors: {mea_result.posteriors}")


if __name__ == "__main__":
    main()

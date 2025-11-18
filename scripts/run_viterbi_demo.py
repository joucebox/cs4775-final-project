#!/usr/bin/env python3
"""Run the Viterbi aligner on a single Stockholm alignment and print results."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Tuple

import yaml

# Ensure repository modules are importable when invoked as a script
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.algorithms.hmm import PairHMM  # pylint: disable=C0413
from src.algorithms.viterbi import ViterbiAligner  # pylint: disable=C0413
from src.types import SequenceType  # pylint: disable=C0413
from src.types.parameters import (  # pylint: disable=C0413
    EmissionParameters,
    GapParameters,
    HMMParameters,
    TransitionParameters,
)
from src.utils.stockholm import read_rna_stockholm  # pylint: disable=C0413

PROJECT_ROOT = Path(__file__).parent.parent
ALIGNMENTS_FOLDER = PROJECT_ROOT / "data" / "alignments"
FASTA_FOLDER = PROJECT_ROOT / "data" / "fasta"
HMM_YAML = PROJECT_ROOT / "results" / "parameters" / "hmm.yaml"


def load_pair_hmm(yaml_path: Path) -> PairHMM:
    """Load PairHMM parameters from the provided YAML file."""
    with yaml_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)

    params_dict = payload.get("parameters", payload)
    emissions_dict = params_dict["log_emissions"]
    transitions_dict = params_dict["log_transitions"]["matrix"]
    gaps_dict = params_dict["gaps"]

    params = HMMParameters(
        log_emissions=EmissionParameters(
            match=emissions_dict["match"],
            insert_x=emissions_dict["insert_x"],
            insert_y=emissions_dict["insert_y"],
        ),
        log_transitions=TransitionParameters(matrix=transitions_dict),
        gaps=GapParameters(**gaps_dict),
    )
    return PairHMM(params)


def format_alignment(seqs: Tuple[SequenceType, SequenceType]) -> str:
    """Return a human-readable two-line alignment string."""
    lines = []
    for seq in seqs:
        residues = "".join(seq.residues)
        lines.append(f"{seq.identifier:>10}: {residues}")
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

    reference = read_rna_stockholm(str(stockholm_path))
    if len(reference.original_sequences) != 2:
        raise ValueError(
            "Expected a pairwise alignment; found "
            f"{len(reference.original_sequences)} sequences."
        )

    hmm = load_pair_hmm(HMM_YAML)
    aligner = ViterbiAligner()

    seq_x, seq_y = reference.original_sequences
    result = aligner.align(hmm, seq_x, seq_y)
    viterbi_alignment = result.alignment.aligned_sequences

    print("Reference alignment (Stockholm):")
    print(format_alignment(tuple(reference.aligned_sequences)))
    print("\nViterbi alignment:")
    print(format_alignment(tuple(viterbi_alignment)))
    print(f"\nViterbi log-score: {result.score:.6f}")


if __name__ == "__main__":
    main()

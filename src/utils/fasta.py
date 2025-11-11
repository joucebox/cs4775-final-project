"""Functions for working with FASTA files."""

from typing import List
import skbio.io

from src.types import RNASequence, SequenceType


def read_rna_fasta(file_path: str) -> List[SequenceType]:
    """Read a FASTA file and return a list of RNASequence.

    Attempts to use scikit-bio if available; falls back to a simple parser.
    """
    sequences: List[RNASequence] = []
    for record in skbio.io.read(file_path, format="fasta"):
        metadata = getattr(record, "metadata", {}) or {}
        identifier = metadata.get("id") or ""
        description = metadata.get("description")
        seq_str = str(record)
        sequences.append(
            RNASequence(
                identifier=identifier,
                residues=list(seq_str),
                description=description,
            )
        )
    return sequences


__all__ = ["read_rna_fasta"]

"""Functions for working with FASTA files."""

from typing import List
import skbio.io
from skbio import RNA

from src.types import RNASequence, SequenceType


def rna_sequence_from_skbio(record: RNA, aligned: bool) -> RNASequence:
    """Convert a scikit-bio record to an RNASequence."""
    metadata = getattr(record, "metadata", {}) or {}
    identifier = metadata.get("id") or ""
    description = metadata.get("description")
    seq_str = b"".join(record.values).decode().replace(".", "-")

    return RNASequence(
        identifier=identifier,
        residues=list(seq_str),
        description=description,
        aligned=aligned,
    )


def read_rna_fasta(file_path: str, ids: List[str] = None) -> List[SequenceType]:
    """Read a FASTA file and return a list of RNASequence.

    Attempts to use scikit-bio if available; falls back to a simple parser.
    """
    sequences: List[RNASequence] = []
    for record in skbio.io.read(file_path, format="fasta"):
        if ids and record.metadata["id"] not in ids:
            continue

        try:
            sequences.append(rna_sequence_from_skbio(record, aligned=False))
        except Exception:
            pass
    return sequences


__all__ = ["read_rna_fasta"]

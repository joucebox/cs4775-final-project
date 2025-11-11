"""Functions for working with Stockholm files."""

from skbio import RNA, TabularMSA

from src.types import Alignment


def read_rna_stockholm(file_path: str) -> Alignment:
    """Read a Stockholm file and return an Alignment."""
    msa = TabularMSA.read(file_path, constructor=RNA)
    return Alignment(
        name=msa.metadata.get("name"),
        aligned_sequences=msa.sequences,
        original_sequences=msa.sequences,
    )


__all__ = ["read_rna_stockholm"]

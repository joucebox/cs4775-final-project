"""Functions for working with Stockholm files."""

import io

from skbio import RNA, TabularMSA
from src.types import Alignment
from .fasta import rna_sequence_from_skbio


def _uppercase_stockholm_sequences(file_path: str) -> io.StringIO:
    """Return a file-like object of the Stockholm with sequences uppercased.

    Only uppercases aligned sequence lines (not metadata lines starting with '#'
    and not the footer line '//'). This avoids altering GR/GC feature data.
    """
    out_lines = []
    with open(file_path, "r", encoding="utf-8") as fh:
        for raw in fh:
            stripped = raw.rstrip("\n")
            if stripped and not stripped.startswith("#") and stripped.rstrip() != "//":
                parts = stripped.split(None, 1)
                if len(parts) == 2:
                    name, seq = parts
                    out_lines.append(f"{name} {seq.upper()}\n")
                else:
                    out_lines.append(raw)
            else:
                out_lines.append(raw)
    return io.StringIO("".join(out_lines))


def read_rna_stockholm(file_path: str) -> Alignment:
    """Read a Stockholm file and return an Alignment."""
    # Preprocess to uppercase aligned sequence characters to satisfy RNA alphabet
    fh = _uppercase_stockholm_sequences(file_path)
    msa = TabularMSA.read(fh, constructor=RNA)

    aligned_sequences = []
    for seq in msa:
        aligned_sequences.append(rna_sequence_from_skbio(seq))

    return Alignment(
        name=msa.metadata.get("name"),
        aligned_sequences=aligned_sequences,
        original_sequences=aligned_sequences,
    )


__all__ = ["read_rna_stockholm"]

"""Utilities for writing and parsing Stockholm format files."""

import io
from typing import Dict

from src.utils.stockholm import uppercase_fileio

from skbio import RNA, TabularMSA


def parse_stockholm_to_dict(file_content: str) -> Dict[str, str]:
    """Parse Stockholm format text into a dictionary."""
    # Use skbio to parse the Stockholm format
    fh = io.StringIO(file_content)
    fh = uppercase_fileio(fh)
    msa = TabularMSA.read(fh, constructor=RNA, format="stockholm")

    result = {}
    for seq in msa:
        metadata = getattr(seq, "metadata", {}) or {}
        identifier = metadata.get("id") or ""
        aligned_seq = str(seq)

        if identifier:
            result[identifier] = aligned_seq

    return result


def write_stockholm_pairwise(
    output_path: str,
    seq1_name: str,
    seq2_name: str,
    aligned1: str,
    aligned2: str,
) -> None:
    """Write a pairwise Stockholm alignment file."""
    with open(output_path, "w", encoding="utf-8") as f:
        # Write Stockholm header
        f.write("# STOCKHOLM 1.0\n")
        f.write("#=GF AU Downloaded from Rfam\n")
        f.write("\n")

        # Write aligned sequences
        # Pad sequence names to align nicely (use max length + 2 spaces)
        max_name_len = max(len(seq1_name), len(seq2_name))
        f.write(f"{seq1_name:<{max_name_len}}  {aligned1}\n")
        f.write(f"{seq2_name:<{max_name_len}}  {aligned2}\n")

        # Write end marker
        f.write("//\n")


def write_fasta_pairwise(
    output_path: str,
    seq1_name: str,
    seq2_name: str,
    unaligned1: str,
    unaligned2: str,
) -> None:
    """Write a pairwise FASTA file."""
    with open(output_path, "w", encoding="utf-8") as f:
        # Write first sequence
        header1 = f">{seq1_name}"
        f.write(f"{header1}\n")
        f.write(f"{unaligned1}\n")

        # Write second sequence
        header2 = f">{seq2_name}"
        f.write(f"{header2}\n")
        f.write(f"{unaligned2}\n")


__all__ = [
    "parse_stockholm_to_dict",
    "write_stockholm_pairwise",
    "write_fasta_pairwise",
]

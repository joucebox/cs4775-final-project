"""Functions for working with Stockholm files."""

import io

from skbio import RNA, TabularMSA
from src.types import Alignment
from .fasta import rna_sequence_from_skbio


def _uppercase_stockholm_sequences(file_path: str) -> io.StringIO:
    """
    Return a file-like object of the Stockholm with sequences uppercased.

    Only uppercases aligned sequence lines (not metadata lines starting with '#'
    and not the footer line '//'). This avoids altering GR/GC feature data.
    """
    out_lines = []
    with open(file_path, "r", encoding="utf-8") as fh:
        for raw in fh:
            stripped = raw.rstrip("\n")
            # Only process "sequence" lines (not # metadata and not // end marker)
            if stripped and not stripped.startswith("#") and stripped.rstrip() != "//":
                # Split line into identifier and sequence; only uppercase the sequence part
                parts = stripped.split(None, 1)
                if len(parts) == 2:
                    name, seq = parts
                    # Preserve the exact spacing between fields by reconstructing with original separator
                    space_index = raw.find(" ")
                    if space_index != -1:
                        before = raw[:space_index]
                        after = raw[space_index + 1 :].rstrip("\n")
                        out_lines.append(f"{before}{raw[space_index]}{after.upper()}\n")
                    else:
                        # fallback if no space found (shouldn't happen with valid lines)
                        out_lines.append(f"{name} {seq.upper()}\n")
                else:
                    # keep malformed/unexpected lines unchanged
                    out_lines.append(raw)
            else:
                # Copy metadata lines and end-of-block marker as is
                out_lines.append(raw)
    # Save the new Stockholm lines back to the same file
    with open(file_path, "w", encoding="utf-8") as fh:
        fh.writelines(out_lines)
    # Return also the file-like StringIO for further in-memory use
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

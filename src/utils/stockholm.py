"""Functions for working with Stockholm files."""

import io
import os
from typing import List

from skbio import RNA, TabularMSA

from src.types import Alignment
from .fasta import rna_sequence_from_skbio, read_rna_fasta


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
                    # Preserve the exact spacing between fields
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

    # Determine matching FASTA path with same basename under sibling 'fasta' dir
    sto_path = os.path.abspath(file_path)
    base = os.path.splitext(os.path.basename(sto_path))[0]
    data_root = os.path.dirname(os.path.dirname(sto_path))  # .../data
    fasta_path = os.path.join(data_root, "fasta", f"{base}.fa")

    # Read FASTA and build a lookup by identifier
    fasta_sequences = read_rna_fasta(fasta_path)
    fasta_by_id = {seq.identifier: seq for seq in fasta_sequences}
    allowed_ids = set(fasta_by_id.keys())

    # Build aligned sequences from MSA that are present in FASTA
    aligned_sequences = []
    for seq in msa:
        identifier = (getattr(seq, "metadata", {}) or {}).get("id") or ""
        if identifier in allowed_ids:
            aligned_sequences.append(rna_sequence_from_skbio(seq, aligned=True))

    # Original (unaligned) sequences filtered to the same identifiers
    original_sequences = [
        fasta_by_id[s.identifier]
        for s in aligned_sequences
        if s.identifier in fasta_by_id
    ]

    if len(aligned_sequences) != len(original_sequences) or len(aligned_sequences) != 2:
        raise ValueError(f"Expected 2 aligned sequences, got {len(aligned_sequences)}")

    return _clean_alignment(
        Alignment(
            name=os.path.basename(file_path),
            aligned_sequences=aligned_sequences,
            original_sequences=original_sequences,
        )
    )


def _clean_alignment(alignment: Alignment) -> Alignment:
    """Clean the alignment by removing columns where all aligned sequences have a gap."""
    seqs = [list(seq.residues) for seq in alignment.aligned_sequences]
    if not seqs:
        return alignment

    seq_len = len(seqs[0])

    # Identify columns where all entries are gaps
    columns_to_remove = []
    for col_idx in range(seq_len):
        column = [seq[col_idx] for seq in seqs]
        if all(residue in ("-", ".") for residue in column):
            columns_to_remove.append(col_idx)

    # Build new sequences with those columns removed
    cleaned_seqs = []
    for seq_idx, seq in enumerate(seqs):
        new_residues = [
            seq[col_idx]
            for col_idx in range(seq_len)
            if col_idx not in columns_to_remove
        ]
        # Replace the residues in the original aligned sequence to keep metadata/id
        old_seq = alignment.aligned_sequences[seq_idx]
        cleaned_seq = type(old_seq)(
            identifier=old_seq.identifier,
            residues=new_residues,
            description=old_seq.description,
            aligned=old_seq.aligned,
        )
        cleaned_seqs.append(cleaned_seq)

    return Alignment(
        name=alignment.name,
        aligned_sequences=cleaned_seqs,
        original_sequences=alignment.original_sequences,
    )


def collect_alignments(folder_path: str) -> List[Alignment]:
    """Collect all alignments from the given folder."""
    alignments = []
    for file_path in os.listdir(folder_path):
        if file_path.lower().endswith(".sto"):
            alignments.append(read_rna_stockholm(os.path.join(folder_path, file_path)))
    return alignments


__all__ = ["collect_alignments"]

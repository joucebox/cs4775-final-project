"""Functions for working with Stockholm files."""

import io
import os
from typing import List
from itertools import combinations

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


def read_rna_stockholm(file_path: str) -> List[Alignment]:
    """Read a Stockholm file and return a list of Alignments for all pairs.

    This function:
    1. Reads all sequences from the corresponding FASTA file
    2. Generates all pairwise combinations of sequences
    3. For each pair, creates an Alignment from the Stockholm data
    4. Returns a list of all pairwise Alignments
    """
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
    aligned_sequences_dict = {}
    for seq in msa:
        identifier = (getattr(seq, "metadata", {}) or {}).get("id") or ""
        if identifier in allowed_ids:
            aligned_sequences_dict[identifier] = rna_sequence_from_skbio(
                seq, aligned=True
            )

    # Get list of identifiers for creating pairs
    identifiers = list(aligned_sequences_dict.keys())

    if len(identifiers) < 2:
        raise ValueError(f"Expected at least 2 sequences, got {len(identifiers)}")

    # Generate all pairwise combinations
    pairs = list(combinations(identifiers, 2))

    # Create an Alignment for each pair
    alignments = []
    for i, (id1, id2) in enumerate(pairs):
        pair_aligned_sequences = [
            aligned_sequences_dict[id1],
            aligned_sequences_dict[id2],
        ]
        pair_original_sequences = [fasta_by_id[id1], fasta_by_id[id2]]

        alignment_name = f"{os.path.basename(file_path)}_{i}"
        alignments.append(
            _clean_alignment(
                Alignment(
                    name=alignment_name,
                    aligned_sequences=pair_aligned_sequences,
                    original_sequences=pair_original_sequences,
                )
            )
        )

    return alignments


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
    """Collect all alignments from the given folder.

    Since each Stockholm file now generates multiple pairwise alignments,
    this returns a flattened list of all pairwise alignments from all files.
    """
    alignments = []
    for file_path in os.listdir(folder_path):
        if file_path.lower().endswith(".sto"):
            # read_rna_stockholm now returns a list of alignments
            pairwise_alignments = read_rna_stockholm(
                os.path.join(folder_path, file_path)
            )
            alignments.extend(pairwise_alignments)
    return alignments


__all__ = ["collect_alignments"]

"""Read the sequences and alignments from the data files."""

import os
import sys
from pathlib import Path

# Add the repository root to the Python path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils import read_rna_fasta, read_rna_stockholm  # pylint: disable=C0413

PROJECT_ROOT = Path(__file__).parent.parent
DATA_FOLDER = PROJECT_ROOT / "data"
FASTA_FOLDER = DATA_FOLDER / "fasta"
ALIGNMENTS_FOLDER = DATA_FOLDER / "alignments"


def print_fasta_sequences(folder_path):
    """Print the sequences in the FASTA files in the given folder."""
    print(f"Reading FASTA files from {folder_path}")
    for filename in os.listdir(folder_path):
        if filename.lower().endswith((".fa", ".fasta")):
            file_path = os.path.join(folder_path, filename)
            print(f"--- {filename} ---")
            sequences = read_rna_fasta(file_path)
            for seq in sequences:
                print(seq)


def print_stockholm_alignments(folder_path):
    """Print the alignments in the Stockholm files in the given folder."""
    print(f"Reading Stockholm files from {folder_path}")
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".sto"):
            file_path = os.path.join(folder_path, filename)
            print(f"--- {filename} ---")
            alignment = read_rna_stockholm(file_path)
            print(f"Name: {alignment.name}")
            print("Aligned Sequences:")
            for i, seq in enumerate(alignment.aligned_sequences):
                print(f"  Seq {i+1}: {seq}")
            print()


if __name__ == "__main__":
    print_fasta_sequences(FASTA_FOLDER)
    print_stockholm_alignments(ALIGNMENTS_FOLDER)

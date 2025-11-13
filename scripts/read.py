"""Read the sequences and alignments from the data files."""

import sys
from pathlib import Path

# Add the repository root to the Python path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils import collect_alignments  # pylint: disable=C0413

PROJECT_ROOT = Path(__file__).parent.parent
DATA_FOLDER = PROJECT_ROOT / "data"
FASTA_FOLDER = DATA_FOLDER / "fasta"
ALIGNMENTS_FOLDER = DATA_FOLDER / "alignments"


def print_stockholm_alignments(folder_path):
    """Print the alignments in the Stockholm files in the given folder."""
    print(f"Reading Stockholm files from {folder_path}")
    alignments = collect_alignments(folder_path)
    for alignment in alignments:
        print(f"--- {alignment.name} ---")
        print(alignment)


if __name__ == "__main__":
    print_stockholm_alignments(ALIGNMENTS_FOLDER)

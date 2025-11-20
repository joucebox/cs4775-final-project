"""Constants for the project."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_FOLDER = PROJECT_ROOT / "data"
FASTA_FOLDER = DATA_FOLDER / "fasta"
ALIGNMENTS_FOLDER = DATA_FOLDER / "alignments"
OUTPUT_FOLDER = PROJECT_ROOT / "results" / "parameters"
HMM_YAML = OUTPUT_FOLDER / "hmm.yaml"
CSV_PATH = PROJECT_ROOT / "results" / "gamma.csv"

PRECISION = 6
PSEUDOCOUNT = 0.5

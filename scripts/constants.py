"""Constants for the project."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_FOLDER = PROJECT_ROOT / "data"
FASTA_FOLDER = DATA_FOLDER / "fasta"
ALIGNMENTS_FOLDER = DATA_FOLDER / "alignments"
RFAM_FA_PATH = DATA_FOLDER / "all_sequences.fa"
OUTPUT_FOLDER = PROJECT_ROOT / "results" / "parameters"
HMM_YAML = OUTPUT_FOLDER / "hmm.yaml"
CSV_FOLDER = PROJECT_ROOT / "results" / "metrics"

PRECISION = 6
PSEUDOCOUNT = 0.5
GAMMA_VALUES = [0.01, 0.1, 0.25, 0.5, 1.0]

# Rfam download configuration
FULL_ALIGN_BASE_URL = (
    "https://ftp.ebi.ac.uk/pub/databases/Rfam/CURRENT/full_alignments/"
)
DEFAULT_NUM_FAMILIES = 10
MAX_PAIRS_PER_FAMILY = 5
MAX_SEQUENCES_PER_FAMILY = 25
RANDOM_SEED = 42

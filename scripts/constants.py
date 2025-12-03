"""Constants for the project."""

from pathlib import Path
from typing import Dict

PROJECT_ROOT = Path(__file__).parent.parent
DATA_FOLDER = PROJECT_ROOT / "data"
FASTA_FOLDER = DATA_FOLDER / "fasta"
ALIGNMENTS_FOLDER = DATA_FOLDER / "alignments"
RFAM_FA_PATH = DATA_FOLDER / "all_sequences.fa"
OUTPUT_FOLDER = PROJECT_ROOT / "results" / "parameters"
HMM_YAML = OUTPUT_FOLDER / "hmm.yaml"
CSV_FOLDER = PROJECT_ROOT / "results" / "metrics"
FIGURES_FOLDER = PROJECT_ROOT / "results" / "figures"
POSTERIORS_FOLDER = PROJECT_ROOT / "results" / "posteriors"
POSTERIOR_GAIN_FOLDER = PROJECT_ROOT / "results" / "posterior_gain"
CACHE_FOLDER = PROJECT_ROOT / ".cache"

PRECISION = 6
PSEUDOCOUNT = 0.5
GAMMA_VALUES = [0.01, 0.1, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]

# Rfam download configuration
FULL_ALIGN_BASE_URL = (
    "https://ftp.ebi.ac.uk/pub/databases/Rfam/CURRENT/full_alignments/"
)
DEFAULT_NUM_FAMILIES = 25
MAX_PAIRS_PER_FAMILY = 5
MAX_SEQUENCES_PER_FAMILY = 25
RANDOM_SEED = 42

# Plot styling
ALIGNER_COLORS: Dict[str, str] = {
    "mea": "#2E86AB",
    "viterbi": "#A23B72",
    "shared": "#0c6e17",
}
PLOT_DPI = 300
PLOT_GRID_ALPHA = 0.3
PLOT_XLABEL_FONTSIZE = 12
PLOT_YLABEL_FONTSIZE = 12
PLOT_TITLE_FONTSIZE = 14

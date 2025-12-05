"""Constants for the project."""

from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).parent.parent

# ============================================================================
# Data directories
# ============================================================================
DATA_FOLDER = PROJECT_ROOT / "data"
FASTA_FOLDER = DATA_FOLDER / "fasta"
ALIGNMENTS_FOLDER = DATA_FOLDER / "alignments"
RFAM_FA_PATH = DATA_FOLDER / "all_sequences.fa"

# ============================================================================
# Results directories
# ============================================================================
RESULTS_FOLDER = PROJECT_ROOT / "results"

# HMM parameter estimation outputs
HMM_YAML = RESULTS_FOLDER / "parameters" / "hmm.yaml"

# Alignment evaluation metrics (CSV files from evaluate_alignments.py)
EVALUATION_METRICS_FOLDER = RESULTS_FOLDER / "metrics"

# Metrics comparison plots (from plot_metrics.py)
METRICS_FIGURES_FOLDER = RESULTS_FOLDER / "figures"

# Posterior heatmap visualizations (from plot_posteriors.py)
POSTERIOR_HEATMAPS_FOLDER = RESULTS_FOLDER / "posteriors"

# Posterior mass gain analysis (from plot_posterior_gain.py)
POSTERIOR_GAIN_FOLDER = RESULTS_FOLDER / "posterior_gain"

# ============================================================================
# Cache directory
# ============================================================================
CACHE_FOLDER = PROJECT_ROOT / ".cache"

# ============================================================================
# Algorithm parameters
# ============================================================================
PRECISION = 6
PSEUDOCOUNT = 0.5
GAMMA_VALUES: List[float] = [
    0.01,
    0.1,
    0.125,
    0.25,
    0.375,
    0.5,
    0.625,
    0.75,
    0.875,
    1.0,
]
MEA_METHODS: List[str] = ["power", "threshold", "probcons", "log_odds"]

# ============================================================================
# Rfam download configuration
# ============================================================================
FULL_ALIGN_BASE_URL = (
    "https://ftp.ebi.ac.uk/pub/databases/Rfam/CURRENT/full_alignments/"
)
DEFAULT_NUM_FAMILIES = 25
MAX_PAIRS_PER_FAMILY = 5
MAX_SEQUENCES_PER_FAMILY = 25
RANDOM_SEED = 42

# ============================================================================
# Plot styling
# ============================================================================
ALIGNER_COLORS: Dict[str, str] = {
    "mea": "#2E86AB",
    "viterbi": "#A23B72",
    "shared": "#0c6e17",
}
METHOD_COLORS: Dict[str, str] = {
    "power": "#2ea1ab",
    "threshold": "#2e62ab",
    "probcons": "#2e3fab",
    "log_odds": "#2eab90",
}
PLOT_DPI = 300
PLOT_GRID_ALPHA = 0.3
PLOT_XLABEL_FONTSIZE = 12
PLOT_YLABEL_FONTSIZE = 12
PLOT_TITLE_FONTSIZE = 14

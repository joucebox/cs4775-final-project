"""Utility functions for the project."""

from .fasta import read_rna_fasta
from .stockholm import collect_alignments
from .serialization import parameters_to_dict, load_pair_hmm
from .posterior_analysis import (
    compute_posteriors,
    extract_alignment_pairs,
    compute_posterior_metrics,
    PosteriorCache,
    MEAMethod,
)

__all__ = [
    "read_rna_fasta",
    "collect_alignments",
    "parameters_to_dict",
    "load_pair_hmm",
    "compute_posteriors",
    "extract_alignment_pairs",
    "compute_posterior_metrics",
    "PosteriorCache",
    "MEAMethod",
]

"""CLI to estimate HMM parameters from Stockholm alignments and dump YAML."""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

from .constants import ALIGNMENTS_FOLDER, HMM_YAML, PRECISION, PSEUDOCOUNT

# Add the repository root to the Python path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils import collect_alignments, parameters_to_dict
from src.algorithms.mle_parameters import (
    estimate_params_from_alignments,
)


def load_alignments(folder: Path):
    """Load alignments from the given folder."""
    if not folder.exists():
        raise FileNotFoundError(f"Alignments directory does not exist: {folder}")
    alignments = collect_alignments(str(folder))
    if not alignments:
        raise ValueError(f"No Stockholm files found in {folder}")
    return alignments


def main() -> None:
    """Estimate HMM parameters from alignments and dump YAML."""
    alignments = load_alignments(ALIGNMENTS_FOLDER)

    params = estimate_params_from_alignments(alignments, pseudocount=PSEUDOCOUNT)

    HMM_YAML.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": {
            "alignments_dir": str(ALIGNMENTS_FOLDER),
            "num_alignments": len(alignments),
        },
        "parameters": parameters_to_dict(params, float_precision=PRECISION),
    }

    with HMM_YAML.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)

    print(
        f"Wrote parameters for {len(alignments)} alignments to {HMM_YAML}",
        file=sys.stdout,
    )


if __name__ == "__main__":
    main()

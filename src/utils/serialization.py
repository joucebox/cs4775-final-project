"""Serialization utilities for HMM parameters (load and save)."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict

import yaml

from src.algorithms.hmm import PairHMM
from src.types.parameters import (
    EmissionParameters,
    GapParameters,
    HMMParameters,
    TransitionParameters,
)


def _convert_values(value: Any, precision: int | None) -> Any:
    """
    Recursively convert dataclasses/dicts/lists and optionally round floats.
    """
    if is_dataclass(value):
        return _convert_values(asdict(value), precision)
    if isinstance(value, dict):
        return {key: _convert_values(val, precision) for key, val in value.items()}
    if isinstance(value, list):
        return [_convert_values(item, precision) for item in value]
    if isinstance(value, float) and precision is not None:
        return round(value, precision)
    return value


def parameters_to_dict(
    params: HMMParameters, float_precision: int | None = 6
) -> Dict[str, Any]:
    """
    Convert HMMParameters dataclass into a plain dictionary suitable for YAML.
    """
    return _convert_values(params, float_precision)


def load_pair_hmm(yaml_path: Path) -> PairHMM:
    """Load PairHMM parameters from a YAML file."""
    with yaml_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)

    params_dict = payload.get("parameters", payload)
    emissions_dict = params_dict["log_emissions"]
    transitions_dict = params_dict["log_transitions"]["matrix"]
    gaps_dict = params_dict["gaps"]

    params = HMMParameters(
        log_emissions=EmissionParameters(
            match=emissions_dict["match"],
            insert_x=emissions_dict["insert_x"],
            insert_y=emissions_dict["insert_y"],
        ),
        log_transitions=TransitionParameters(matrix=transitions_dict),
        gaps=GapParameters(**gaps_dict),
    )
    return PairHMM(params)


__all__ = ["parameters_to_dict", "load_pair_hmm"]

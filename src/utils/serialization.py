"""Serialization utilities for exporting HMM parameters."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Dict

from src.types.parameters import HMMParameters


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


__all__ = ["parameters_to_dict"]

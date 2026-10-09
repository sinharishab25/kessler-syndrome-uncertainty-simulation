"""Canonical model-time conversion helpers."""

from __future__ import annotations

import math
import warnings


CANONICAL_MODEL_TIME_UNIT = "years"
SUPPORTED_MODEL_TIME_UNITS = ("days", "years")


def model_time_to_years(
    value: float,
    unit: str,
    days_per_year: float,
) -> float:
    """Convert a supported model-time value into canonical model years."""
    numeric_value = float(value)
    normalized_unit = _normalize_model_time_unit(unit)
    if normalized_unit == "years":
        return numeric_value
    if normalized_unit == "days":
        return numeric_value / float(days_per_year)

    raise ValueError(f"Unsupported model-time unit {unit!r}.")


def model_years_to_unit(
    value_years: float,
    unit: str,
    days_per_year: float,
) -> float:
    """Convert canonical model years into a supported display/config unit."""
    numeric_value = float(value_years)
    normalized_unit = _normalize_model_time_unit(unit)
    if normalized_unit == "years":
        return numeric_value
    if normalized_unit == "days":
        return numeric_value * float(days_per_year)

    raise ValueError(f"Unsupported model-time unit {unit!r}.")


def calculate_step_count(duration_years: float, dt_years: float) -> int:
    """Return the step count needed without overshooting a finite duration."""
    numeric_duration = float(duration_years)
    numeric_dt = float(dt_years)
    if numeric_duration <= 0.0:
        raise ValueError("duration_years must be greater than 0.")
    if numeric_dt <= 0.0:
        raise ValueError("dt_years must be greater than 0.")

    return max(1, int(math.ceil(numeric_duration / numeric_dt)))


def warn_deprecated_alias(alias_name: str, replacement_name: str) -> None:
    """Emit a consistent deprecation warning for old seconds-based settings."""
    warnings.warn(
        f"{alias_name} is deprecated. Use {replacement_name} with explicit "
        "model-time units instead.",
        DeprecationWarning,
        stacklevel=3,
    )


def _normalize_model_time_unit(unit: str) -> str:
    """Normalize plural/singular supported model-time unit names."""
    if not isinstance(unit, str):
        raise ValueError(f"Model-time unit must be a string. Current value: {unit!r}.")

    normalized = unit.strip().lower()
    aliases = {
        "day": "days",
        "days": "days",
        "year": "years",
        "years": "years",
    }
    if normalized not in aliases:
        supported_units = ", ".join(SUPPORTED_MODEL_TIME_UNITS)
        raise ValueError(
            f"Unsupported model-time unit {unit!r}. Choose one of: {supported_units}."
        )

    return aliases[normalized]

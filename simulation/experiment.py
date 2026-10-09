"""Experiment output directories and metadata for Version 1.5.15 runs."""

from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
import json
import sys
import uuid

import config


def create_experiment_output_directory(run_mode: str) -> Path:
    """Create and activate a unique output directory when configured."""
    output_root = Path(config.OUTPUT_DIRECTORY)
    output_root.mkdir(parents=True, exist_ok=True)

    if not config.USE_UNIQUE_EXPERIMENT_DIRECTORIES:
        return output_root

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    experiment_id = f"{timestamp}_{run_mode}_{uuid.uuid4().hex[:8]}"
    output_directory = output_root / experiment_id
    output_directory.mkdir(parents=True, exist_ok=False)
    config.OUTPUT_DIRECTORY = str(output_directory)
    return output_directory


def write_experiment_metadata(
    output_directory: Path,
    run_mode: str,
    status: str,
    failure_message: str = "",
    results: list | None = None,
    extra_metadata: dict[str, object] | None = None,
) -> Path:
    """Write effective configuration, command, seeds, status, and failures."""
    output_directory.mkdir(parents=True, exist_ok=True)
    metadata_path = output_directory / config.EXPERIMENT_METADATA_FILENAME
    metadata = {
        "code_version": config.CODE_VERSION,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command": " ".join(sys.argv),
        "run_mode": run_mode,
        "output_directory": str(output_directory.resolve()),
        "status": status,
        "failure_message": failure_message,
        "effective_config": _get_effective_config(),
        "seeds": _get_seed_records(results or []),
    }
    if extra_metadata:
        metadata.update(_to_json_safe(extra_metadata))
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return metadata_path


def _get_seed_records(results: list) -> list[dict[str, object]]:
    """Return per-run seed and status records for completed result objects."""
    seed_records = []
    for result in results:
        seed_records.append(
            {
                "run_index": getattr(result, "run_index", None),
                "repetition_index": getattr(result, "repetition_index", None),
                "sigma_index": getattr(result, "sigma_index", None),
                "sigma_km": getattr(result, "position_uncertainty_sigma_km", None),
                "experiment_condition_id": getattr(
                    result,
                    "experiment_condition_id",
                    None,
                ),
                "population_value": getattr(result, "population_value", None),
                "run_seed": getattr(result, "run_seed", None),
                "observation_seed": getattr(result, "observation_seed", None),
                "fragmentation_seed": getattr(result, "fragmentation_seed", None),
                "status": getattr(result, "status", None),
                "failure_message": getattr(result, "failure_message", ""),
            }
        )
    return seed_records


def _get_effective_config() -> dict[str, object]:
    """Return public uppercase config values converted to JSON-safe objects."""
    effective_config = {}
    for name in sorted(dir(config)):
        if not name.isupper():
            continue
        value = getattr(config, name)
        if callable(value):
            continue
        effective_config[name] = _to_json_safe(value)
    return effective_config


def _to_json_safe(value):
    """Convert config values and result fragments to JSON-safe structures."""
    if is_dataclass(value):
        return _to_json_safe(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _to_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)






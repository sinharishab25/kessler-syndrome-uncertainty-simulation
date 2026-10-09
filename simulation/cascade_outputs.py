"""Normalized cascade output tables and validators for Version 1.5.15."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
import csv
import json

import config
from analysis.cascade_graphs import plot_cascade_time_series_graphs
from simulation.cascade_metrics import calculate_cascade_level_metric_rows
from simulation.result import SimulationResult


EVENT_FIELDNAMES = [
    "run_uid",
    "population_value",
    "run_index",
    "event_id",
    "model_time_years",
    "parent_i",
    "parent_j",
    "parent_event_i",
    "parent_event_j",
    "generation",
    "fragments_created",
    "first_fragment_id",
    "last_fragment_id",
    "population_before",
    "population_after",
    "debris_population_before",
    "debris_population_after",
    "event_layer",
]

RUN_FIELDNAMES = [
    "run_uid",
    "population_value",
    "run_index",
    "run_seed",
    "observation_seed",
    "fragmentation_seed",
    "initialization_fingerprint",
    "configured_duration_model_years",
    "actual_duration_model_years",
    "dt_model_years",
    "position_uncertainty_sigma_km",
    "initial_active_population",
    "final_active_population",
    "peak_active_population",
    "initial_debris_population",
    "final_debris_population",
    "peak_debris_population",
    "collision_count",
    "maximum_cascade_depth",
    "true_cascade_detected",
    "true_onset_time_model_years",
    "observed_cascade_event_count",
    "observed_cascade_detected",
    "observed_onset_time_model_years",
    "observed_cascade_onset_generation",
    "cascade_level_classification",
    "onset_time_bias_model_years",
    "termination_reason",
    "execution_time_seconds",
    "status",
    "failure_message",
]

OBSERVED_EVENT_FIELDNAMES = [
    "run_uid",
    "population_value",
    "run_index",
    "event_id",
    "model_time_years",
    "parent_i",
    "parent_j",
    "parent_event_i",
    "parent_event_j",
    "generation",
    "event_layer",
]

METRIC_FIELDNAMES = [
    "metric",
    "numerator",
    "denominator",
    "estimate",
    "wilson_lower",
    "wilson_upper",
]

TIME_BIN_FIELDNAMES = [
    "run_uid",
    "population_value",
    "run_index",
    "bin_index",
    "bin_start_model_years",
    "bin_end_model_years",
    "bin_width_model_years",
    "sample_time_model_years",
    "N_active",
    "N_debris",
    "collisions_C_k",
    "collision_rate_lambda_per_model_year",
    "cumulative_collisions",
    "maximum_cascade_depth_to_time",
    "rolling_collision_rate_per_model_year",
    "debris_growth_rate_per_model_year",
]


def save_cascade_output_tables(
    simulation_results: list[SimulationResult],
) -> list[Path]:
    """Write normalized cascade tables, manifest, and derived graphs."""
    output_directory = Path(config.OUTPUT_DIRECTORY)
    output_directory.mkdir(parents=True, exist_ok=True)

    event_rows = build_cascade_event_rows(simulation_results)
    observed_event_rows = build_observed_cascade_event_rows(simulation_results)
    run_rows = build_cascade_run_rows(simulation_results, event_rows)
    time_bin_rows = build_cascade_time_bin_rows(run_rows, event_rows)
    metric_rows = calculate_cascade_level_metric_rows(simulation_results)

    validate_run_rows(run_rows)
    validate_event_rows(event_rows, run_rows)
    validate_time_bin_rows(time_bin_rows, run_rows)

    event_path = output_directory / config.CASCADE_EVENT_TABLE_FILENAME
    observed_event_path = output_directory / config.CASCADE_OBSERVED_EVENT_TABLE_FILENAME
    run_path = output_directory / config.CASCADE_RUN_TABLE_FILENAME
    time_bin_path = output_directory / config.CASCADE_TIME_BIN_TABLE_FILENAME
    metric_path = output_directory / config.CASCADE_LEVEL_METRICS_FILENAME
    manifest_path = output_directory / config.CASCADE_CONFIG_MANIFEST_FILENAME

    _write_csv(event_path, EVENT_FIELDNAMES, event_rows)
    _write_csv(observed_event_path, OBSERVED_EVENT_FIELDNAMES, observed_event_rows)
    _write_csv(run_path, RUN_FIELDNAMES, run_rows)
    _write_csv(time_bin_path, TIME_BIN_FIELDNAMES, time_bin_rows)
    _write_csv(metric_path, METRIC_FIELDNAMES, metric_rows)
    _write_manifest(
        manifest_path,
        event_path,
        observed_event_path,
        run_path,
        time_bin_path,
        metric_path,
    )
    graph_paths = plot_cascade_time_series_graphs(time_bin_path)

    return [
        event_path,
        observed_event_path,
        run_path,
        time_bin_path,
        metric_path,
        manifest_path,
        *graph_paths,
    ]


def build_cascade_event_rows(
    simulation_results: list[SimulationResult],
) -> list[dict[str, object]]:
    """Return one normalized row per true collision event."""
    rows = []
    for result in simulation_results:
        run_uid = _run_uid(result)
        population_before = int(result.initial_particle_count)
        debris_before = 0
        next_fragment_id = int(result.initial_particle_count)
        for event in sorted(_event_dicts(result), key=lambda row: int(row["event_id"])):
            fragments_created = int(event["fragments_created"])
            first_fragment_id = next_fragment_id if fragments_created > 0 else None
            last_fragment_id = (
                next_fragment_id + fragments_created - 1
                if fragments_created > 0
                else None
            )
            debris_parents = int(event["parent_event_i"] != -1) + int(
                event["parent_event_j"] != -1
            )
            debris_after = debris_before - debris_parents + fragments_created
            rows.append(
                {
                    "run_uid": run_uid,
                    "population_value": result.population_value,
                    "run_index": result.run_index,
                    "event_id": int(event["event_id"]),
                    "model_time_years": float(event["model_time_years"]),
                    "parent_i": int(event["parent_i"]),
                    "parent_j": int(event["parent_j"]),
                    "parent_event_i": int(event["parent_event_i"]),
                    "parent_event_j": int(event["parent_event_j"]),
                    "generation": int(event["generation"]),
                    "fragments_created": fragments_created,
                    "first_fragment_id": first_fragment_id,
                    "last_fragment_id": last_fragment_id,
                    "population_before": population_before,
                    "population_after": int(event["population_after"]),
                    "debris_population_before": debris_before,
                    "debris_population_after": debris_after,
                    "event_layer": event.get("event_layer", "true"),
                }
            )
            population_before = int(event["population_after"])
            debris_before = debris_after
            next_fragment_id += fragments_created
    return rows


def build_observed_cascade_event_rows(
    simulation_results: list[SimulationResult],
) -> list[dict[str, object]]:
    """Return one normalized row per deduplicated observed cascade event."""
    rows = []
    for result in simulation_results:
        run_uid = _run_uid(result)
        for event in sorted(
            _observed_event_dicts(result),
            key=lambda row: int(row["event_id"]),
        ):
            rows.append(
                {
                    "run_uid": run_uid,
                    "population_value": result.population_value,
                    "run_index": result.run_index,
                    "event_id": int(event["event_id"]),
                    "model_time_years": float(event["model_time_years"]),
                    "parent_i": int(event["parent_i"]),
                    "parent_j": int(event["parent_j"]),
                    "parent_event_i": int(event["parent_event_i"]),
                    "parent_event_j": int(event["parent_event_j"]),
                    "generation": int(event["generation"]),
                    "event_layer": event.get("event_layer", "observed"),
                }
            )
    return rows


def build_cascade_run_rows(
    simulation_results: list[SimulationResult],
    event_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Return one normalized row per simulation run, including zero-event runs."""
    events_by_run = _group_rows_by_run(event_rows)
    rows = []
    for result in simulation_results:
        run_uid = _run_uid(result)
        events = events_by_run.get(run_uid, [])
        active_values = [int(result.initial_particle_count)]
        active_values.extend(int(row["population_after"]) for row in events)
        debris_values = [0]
        debris_values.extend(int(row["debris_population_after"]) for row in events)
        maximum_depth = max((int(row["generation"]) for row in events), default=0)
        rows.append(
            {
                "run_uid": run_uid,
                "population_value": result.population_value,
                "run_index": result.run_index,
                "run_seed": result.run_seed,
                "observation_seed": result.observation_seed,
                "fragmentation_seed": result.fragmentation_seed,
                "initialization_fingerprint": result.initialization_fingerprint,
                "configured_duration_model_years": result.configured_duration_model_years,
                "actual_duration_model_years": result.actual_duration_model_years,
                "dt_model_years": result.dt_model_years,
                "position_uncertainty_sigma_km": result.position_uncertainty_sigma_km,
                "initial_active_population": result.initial_particle_count,
                "final_active_population": result.final_particle_count,
                "peak_active_population": max(active_values),
                "initial_debris_population": 0,
                "final_debris_population": debris_values[-1],
                "peak_debris_population": max(debris_values),
                "collision_count": len(events),
                "maximum_cascade_depth": maximum_depth,
                "true_cascade_detected": (
                    result.true_cascade_onset_time_model_years is not None
                ),
                "true_onset_time_model_years": result.true_cascade_onset_time_model_years,
                "observed_cascade_event_count": result.observed_cascade_event_count,
                "observed_cascade_detected": (
                    result.observed_cascade_onset_time_model_years is not None
                ),
                "observed_onset_time_model_years": (
                    result.observed_cascade_onset_time_model_years
                ),
                "observed_cascade_onset_generation": (
                    result.observed_cascade_onset_generation
                ),
                "cascade_level_classification": result.cascade_level_classification,
                "onset_time_bias_model_years": result.onset_time_bias_model_years,
                "termination_reason": result.cascade_termination_reason,
                "execution_time_seconds": result.execution_time_seconds,
                "status": result.status,
                "failure_message": result.failure_message,
            }
        )
    return rows


def build_cascade_time_bin_rows(
    run_rows: list[dict[str, object]],
    event_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Return one row per run per deterministic sampled time bin."""
    events_by_run = _group_rows_by_run(event_rows)
    bin_width = config.model_time_config_value_to_years(
        config.CASCADE_TIME_BIN_WIDTH_MODEL_TIME
    )
    rolling_window = config.model_time_config_value_to_years(
        config.CASCADE_ROLLING_WINDOW_MODEL_TIME
    )
    rows = []
    for run_row in run_rows:
        run_uid = str(run_row["run_uid"])
        events = sorted(
            events_by_run.get(run_uid, []),
            key=lambda row: (float(row["model_time_years"]), int(row["event_id"])),
        )
        sample_times = _sample_times(
            float(run_row["actual_duration_model_years"]),
            bin_width,
        )
        previous_time = 0.0
        previous_debris = int(run_row["initial_debris_population"])
        for bin_index, sample_time in enumerate(sample_times):
            interval_events = (
                []
                if bin_index == 0
                else _events_in_interval(events, previous_time, sample_time)
            )
            state_events = [] if bin_index == 0 else _events_to_time(events, sample_time)
            n_active = (
                int(state_events[-1]["population_after"])
                if state_events
                else int(run_row["initial_active_population"])
            )
            n_debris = (
                int(state_events[-1]["debris_population_after"])
                if state_events
                else int(run_row["initial_debris_population"])
            )
            width = sample_time - previous_time if bin_index > 0 else 0.0
            collision_count = len(interval_events)
            cumulative_collisions = len(state_events)
            max_depth = max((int(row["generation"]) for row in state_events), default=0)
            rows.append(
                {
                    "run_uid": run_uid,
                    "population_value": run_row["population_value"],
                    "run_index": run_row["run_index"],
                    "bin_index": bin_index,
                    "bin_start_model_years": previous_time if bin_index > 0 else 0.0,
                    "bin_end_model_years": sample_time,
                    "bin_width_model_years": width,
                    "sample_time_model_years": sample_time,
                    "N_active": n_active,
                    "N_debris": n_debris,
                    "collisions_C_k": collision_count,
                    "collision_rate_lambda_per_model_year": (
                        collision_count / width if width > 0.0 else 0.0
                    ),
                    "cumulative_collisions": cumulative_collisions,
                    "maximum_cascade_depth_to_time": max_depth,
                    "rolling_collision_rate_per_model_year": _rolling_rate(
                        events,
                        sample_time,
                        rolling_window,
                    ),
                    "debris_growth_rate_per_model_year": (
                        (n_debris - previous_debris) / width if width > 0.0 else 0.0
                    ),
                }
            )
            previous_time = sample_time
            previous_debris = n_debris
    return rows


def validate_run_rows(rows: list[dict[str, object]]) -> None:
    """Validate run-level cascade table rows."""
    seen = set()
    for row in rows:
        run_uid = str(row["run_uid"])
        if run_uid in seen:
            raise ValueError(f"Duplicate run_uid: {run_uid}.")
        seen.add(run_uid)
        for field in (
            "initial_active_population",
            "final_active_population",
            "peak_active_population",
            "initial_debris_population",
            "final_debris_population",
            "peak_debris_population",
            "collision_count",
            "maximum_cascade_depth",
            "actual_duration_model_years",
        ):
            if float(row[field]) < 0.0:
                raise ValueError(f"{field} must be nonnegative for {run_uid}.")
        if int(row["peak_active_population"]) < int(row["final_active_population"]):
            raise ValueError(f"Peak active population is too small for {run_uid}.")
        if int(row["peak_debris_population"]) < int(row["final_debris_population"]):
            raise ValueError(f"Peak debris population is too small for {run_uid}.")
        onset = row["true_onset_time_model_years"]
        if onset not in ("", None):
            onset_time = float(onset)
            if onset_time < 0.0 or onset_time > float(row["actual_duration_model_years"]):
                raise ValueError(f"Onset time is out of bounds for {run_uid}.")
        observed_onset = row["observed_onset_time_model_years"]
        if observed_onset not in ("", None):
            observed_onset_time = float(observed_onset)
            if (
                observed_onset_time < 0.0
                or observed_onset_time > float(row["actual_duration_model_years"])
            ):
                raise ValueError(f"Observed onset time is out of bounds for {run_uid}.")
        if row["cascade_level_classification"] not in ("TP", "FP", "FN", "TN"):
            raise ValueError(f"Invalid cascade-level classification for {run_uid}.")


def validate_event_rows(
    event_rows: list[dict[str, object]],
    run_rows: list[dict[str, object]],
) -> None:
    """Validate true-event IDs, ancestry, bounds, and population accounting."""
    run_by_uid = {str(row["run_uid"]): row for row in run_rows}
    grouped_rows = _group_rows_by_run(event_rows)
    for run_uid, run_row in run_by_uid.items():
        if len(grouped_rows.get(run_uid, [])) != int(run_row["collision_count"]):
            raise ValueError(f"Run/event collision counts disagree for {run_uid}.")
    for run_uid, rows in _group_rows_by_run(event_rows).items():
        if run_uid not in run_by_uid:
            raise ValueError(f"Event row references missing run_uid: {run_uid}.")
        run_row = run_by_uid[run_uid]
        sorted_rows = sorted(rows, key=lambda row: int(row["event_id"]))
        event_ids = [int(row["event_id"]) for row in sorted_rows]
        if event_ids != list(range(len(sorted_rows))):
            raise ValueError(f"Event IDs must be unique and monotonic for {run_uid}.")
        active_object_ids = set(range(int(run_row["initial_active_population"])))
        object_creator = {object_id: -1 for object_id in active_object_ids}
        object_generation = {object_id: 0 for object_id in active_object_ids}
        event_generation_by_id = {}
        next_object_id = int(run_row["initial_active_population"])
        for row in sorted_rows:
            _validate_one_event_row(
                row,
                run_row,
                active_object_ids,
                object_creator,
                object_generation,
                event_generation_by_id,
                next_object_id,
            )
            parent_i = int(row["parent_i"])
            parent_j = int(row["parent_j"])
            fragments_created = int(row["fragments_created"])
            active_object_ids.remove(parent_i)
            active_object_ids.remove(parent_j)
            event_id = int(row["event_id"])
            generation = int(row["generation"])
            event_generation_by_id[event_id] = generation
            for fragment_id in range(next_object_id, next_object_id + fragments_created):
                active_object_ids.add(fragment_id)
                object_creator[fragment_id] = event_id
                object_generation[fragment_id] = generation
            next_object_id += fragments_created
        if sorted_rows:
            last_row = sorted_rows[-1]
            if int(last_row["population_after"]) != int(run_row["final_active_population"]):
                raise ValueError(f"Final active population disagrees for {run_uid}.")
            if int(last_row["debris_population_after"]) != int(run_row["final_debris_population"]):
                raise ValueError(f"Final debris population disagrees for {run_uid}.")


def validate_time_bin_rows(
    time_bin_rows: list[dict[str, object]],
    run_rows: list[dict[str, object]],
) -> None:
    """Validate sampled time-bin table structure and nonnegative metrics."""
    run_by_uid = {str(row["run_uid"]): row for row in run_rows}
    for run_uid, rows in _group_rows_by_run(time_bin_rows).items():
        if run_uid not in run_by_uid:
            raise ValueError(f"Time-bin row references missing run_uid: {run_uid}.")
        rows = sorted(rows, key=lambda row: int(row["bin_index"]))
        if int(rows[0]["bin_index"]) != 0 or float(rows[0]["sample_time_model_years"]) != 0.0:
            raise ValueError(f"Time-bin rows must start at time zero for {run_uid}.")
        final_time = float(run_by_uid[run_uid]["actual_duration_model_years"])
        if abs(float(rows[-1]["sample_time_model_years"]) - final_time) > 1e-9:
            raise ValueError(f"Time-bin rows must include exact final horizon for {run_uid}.")
        previous_end = 0.0
        for expected_index, row in enumerate(rows):
            if int(row["bin_index"]) != expected_index:
                raise ValueError(f"Nonmonotonic bin index for {run_uid}.")
            start = float(row["bin_start_model_years"])
            end = float(row["bin_end_model_years"])
            width = float(row["bin_width_model_years"])
            if abs(start - (previous_end if expected_index > 0 else 0.0)) > 1e-9:
                raise ValueError(f"Bin start does not match previous end for {run_uid}.")
            if abs(end - float(row["sample_time_model_years"])) > 1e-9:
                raise ValueError(f"Bin end must equal sample time for {run_uid}.")
            if abs(width - (end - start if expected_index > 0 else 0.0)) > 1e-9:
                raise ValueError(f"Bin width is inconsistent for {run_uid}.")
            for field in (
                "N_active",
                "N_debris",
                "collisions_C_k",
                "collision_rate_lambda_per_model_year",
                "cumulative_collisions",
                "maximum_cascade_depth_to_time",
                "rolling_collision_rate_per_model_year",
            ):
                if float(row[field]) < 0.0:
                    raise ValueError(f"{field} must be nonnegative for {run_uid}.")
            previous_end = end


def _validate_one_event_row(
    row: dict[str, object],
    run_row: dict[str, object],
    active_object_ids: set[int],
    object_creator: dict[int, int],
    object_generation: dict[int, int],
    event_generation_by_id: dict[int, int],
    next_object_id: int,
) -> None:
    run_uid = str(row["run_uid"])
    event_id = int(row["event_id"])
    parent_i = int(row["parent_i"])
    parent_j = int(row["parent_j"])
    if row["event_layer"] != "true":
        raise ValueError(f"Event layer must be true for {run_uid}.")
    if parent_i == parent_j or parent_i < 0 or parent_j < 0:
        raise ValueError(f"Invalid parents for {run_uid} event {event_id}.")
    if parent_i not in active_object_ids or parent_j not in active_object_ids:
        raise ValueError(f"Parent is inactive or unknown for {run_uid} event {event_id}.")
    event_time = float(row["model_time_years"])
    if event_time < 0.0 or event_time > float(run_row["actual_duration_model_years"]):
        raise ValueError(f"Event time is out of bounds for {run_uid} event {event_id}.")
    parent_event_i = int(row["parent_event_i"])
    parent_event_j = int(row["parent_event_j"])
    if parent_event_i != object_creator[parent_i] or parent_event_j != object_creator[parent_j]:
        raise ValueError(f"Parent creator mismatch for {run_uid} event {event_id}.")
    for creator_event in (parent_event_i, parent_event_j):
        if creator_event >= event_id:
            raise ValueError(f"Creator event must precede child event for {run_uid}.")
        if creator_event != -1 and creator_event not in event_generation_by_id:
            raise ValueError(f"Unknown creator event for {run_uid} event {event_id}.")
    expected_generation = 1 + max(object_generation[parent_i], object_generation[parent_j])
    if int(row["generation"]) != expected_generation:
        raise ValueError(f"Generation is invalid for {run_uid} event {event_id}.")
    fragments_created = int(row["fragments_created"])
    if fragments_created < 0:
        raise ValueError(f"fragments_created must be nonnegative for {run_uid}.")
    if int(row["population_after"]) != int(row["population_before"]) - 2 + fragments_created:
        raise ValueError(f"Population accounting failed for {run_uid} event {event_id}.")
    debris_parents = int(parent_event_i != -1) + int(parent_event_j != -1)
    if int(row["debris_population_after"]) != int(row["debris_population_before"]) - debris_parents + fragments_created:
        raise ValueError(f"Debris accounting failed for {run_uid} event {event_id}.")
    if fragments_created == 0:
        if row["first_fragment_id"] not in ("", None) or row["last_fragment_id"] not in ("", None):
            raise ValueError(f"Zero-fragment event has fragment IDs for {run_uid}.")
    else:
        if int(row["first_fragment_id"]) != next_object_id:
            raise ValueError(f"First fragment ID is invalid for {run_uid}.")
        if int(row["last_fragment_id"]) != next_object_id + fragments_created - 1:
            raise ValueError(f"Last fragment ID is invalid for {run_uid}.")
    for field in (
        "generation",
        "fragments_created",
        "population_before",
        "population_after",
        "debris_population_before",
        "debris_population_after",
    ):
        if int(row[field]) < 0:
            raise ValueError(f"{field} must be nonnegative for {run_uid}.")


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field)) for field in fieldnames})


def _write_manifest(
    manifest_path: Path,
    event_path: Path,
    observed_event_path: Path,
    run_path: Path,
    time_bin_path: Path,
    metric_path: Path,
) -> None:
    bin_width_years = config.model_time_config_value_to_years(
        config.CASCADE_TIME_BIN_WIDTH_MODEL_TIME
    )
    rolling_window_years = config.model_time_config_value_to_years(
        config.CASCADE_ROLLING_WINDOW_MODEL_TIME
    )
    manifest = {
        "code_version": config.CODE_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "event_table": str(event_path.resolve()),
        "observed_event_table": str(observed_event_path.resolve()),
        "run_table": str(run_path.resolve()),
        "time_bin_table": str(time_bin_path.resolve()),
        "cascade_level_metrics_table": str(metric_path.resolve()),
        "bin_width_model_years": bin_width_years,
        "rolling_window_model_years": rolling_window_years,
        "bin_edge_rule": (
            "sample time zero is initial pre-event state; positive intervals use "
            "[0, end] for the first bin and (start, end] thereafter; exact final "
            "horizon is always included"
        ),
        "no_onset_encoding": "null in memory and blank in CSV",
        "observed_ancestry_rule": (
            "Observed ancestry follows observed object identity labels only. "
            "Initial visible IDs start at observed generation 0. A deduplicated "
            "observed event assigns generation 1 + max(parent observed generations) "
            "to its observed parents. Newly appearing IDs in the next observation "
            "snapshot inherit only from the most recent observed creator event by "
            "appearance timing, never from true parent-event fields."
        ),
        "model_scope": "scaled educational 2D model; not real-LEO predictive",
        "effective_cascade_output_config": {
            "CASCADE_TIME_BIN_WIDTH_MODEL_TIME": config.CASCADE_TIME_BIN_WIDTH_MODEL_TIME,
            "CASCADE_ROLLING_WINDOW_MODEL_TIME": config.CASCADE_ROLLING_WINDOW_MODEL_TIME,
            "MODEL_TIME_UNIT": config.MODEL_TIME_UNIT,
            "TIME_BASIS_DAYS_PER_YEAR": config.TIME_BASIS_DAYS_PER_YEAR,
            "CASCADE_PLOT_INDIVIDUAL_TRAJECTORIES": (
                config.CASCADE_PLOT_INDIVIDUAL_TRAJECTORIES
            ),
            "CASCADE_MAX_INDIVIDUAL_TRAJECTORIES": (
                config.CASCADE_MAX_INDIVIDUAL_TRAJECTORIES
            ),
            "CASCADE_ONSET_GENERATION": config.CASCADE_ONSET_GENERATION,
            "FRAGMENTATION_MODE": config.FRAGMENTATION_MODE,
            "MAX_ACTIVE_OBJECTS": config.MAX_ACTIVE_OBJECTS,
            "MAX_TRUE_COLLISION_EVENTS": config.MAX_TRUE_COLLISION_EVENTS,
            "CASCADE_TERMINATION_POLICY": config.CASCADE_TERMINATION_POLICY,
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _sample_times(final_time: float, bin_width: float) -> list[float]:
    if final_time < 0.0:
        raise ValueError("final_time must be nonnegative.")
    times = [0.0]
    next_time = bin_width
    while next_time < final_time - 1e-12:
        times.append(next_time)
        next_time += bin_width
    if final_time > 0.0 and abs(times[-1] - final_time) > 1e-12:
        times.append(final_time)
    return times


def _events_in_interval(
    events: list[dict[str, object]],
    start_time: float,
    end_time: float,
) -> list[dict[str, object]]:
    return [
        event
        for event in events
        if (
            (float(event["model_time_years"]) >= start_time - 1e-12 if start_time == 0.0 else float(event["model_time_years"]) > start_time + 1e-12)
            and float(event["model_time_years"]) <= end_time + 1e-12
        )
    ]


def _events_to_time(
    events: list[dict[str, object]],
    sample_time: float,
) -> list[dict[str, object]]:
    return [
        event
        for event in events
        if float(event["model_time_years"]) <= sample_time + 1e-12
    ]


def _rolling_rate(
    events: list[dict[str, object]],
    sample_time: float,
    rolling_window: float,
) -> float:
    if sample_time <= 0.0:
        return 0.0
    start_time = sample_time - rolling_window
    count = sum(
        1
        for event in events
        if float(event["model_time_years"]) > start_time + 1e-12
        and float(event["model_time_years"]) <= sample_time + 1e-12
    )
    return count / rolling_window


def _event_dicts(result: SimulationResult) -> list[dict[str, object]]:
    rows = []
    for event in getattr(result, "true_collision_events", []):
        rows.append(asdict(event) if is_dataclass(event) else dict(event))
    return rows


def _observed_event_dicts(result: SimulationResult) -> list[dict[str, object]]:
    rows = []
    for event in getattr(result, "observed_cascade_events", []):
        rows.append(asdict(event) if is_dataclass(event) else dict(event))
    return rows


def _group_rows_by_run(
    rows: list[dict[str, object]],
) -> dict[str, list[dict[str, object]]]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[str(row["run_uid"])].append(row)
    return grouped


def _run_uid(result: SimulationResult) -> str:
    if result.experiment_condition_id:
        return result.experiment_condition_id
    return f"population_{result.population_value}_run_{result.run_index}"


def _csv_value(value):
    if value is None:
        return ""
    return value

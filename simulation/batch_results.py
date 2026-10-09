"""CSV output and basic statistics for Version 1.5.15 Monte Carlo sweeps."""

from dataclasses import dataclass
from pathlib import Path
import csv
import json
import statistics

import config
from simulation.result import SimulationResult


@dataclass
class PopulationSweepStatistics:
    """Small summary for one population value in a headless sweep."""

    population_value: int
    sigma_km: float
    total_runs: int
    successful_runs: int
    runs_with_collisions: int
    collision_probability: float
    mean_collision_count: float
    median_collision_count: float
    mean_first_collision_time_model_years: float | None
    total_execution_time_seconds: float
    mean_execution_time_seconds: float


def save_monte_carlo_results_csv(
    simulation_results: list[SimulationResult],
) -> Path:
    """Save one raw CSV row for each headless Monte Carlo result."""
    output_directory = Path(config.OUTPUT_DIRECTORY)
    output_directory.mkdir(parents=True, exist_ok=True)
    output_path = output_directory / config.MONTE_CARLO_RESULTS_FILENAME

    fieldnames = [
        "experiment_condition_id",
        "experiment_stage",
        "population_index",
        "population_value",
        "sigma_index",
        "repetition_index",
        "run_index",
        "run_seed",
        "observation_seed",
        "fragmentation_seed",
        "initialization_fingerprint",
        "configured_duration_model_years",
        "actual_duration_model_years",
        "dt_model_years",
        "position_uncertainty_sigma_km",
        "steps_executed",
        "initial_particle_count",
        "final_particle_count",
        "pairs_per_step",
        "total_pair_step_comparisons",
        "TP",
        "FP",
        "FN",
        "TN",
        "true_positive_rate",
        "false_negative_rate",
        "false_positive_rate",
        "precision",
        "total_particle_collisions",
        "true_collision_event_count",
        "true_cascade_onset_time_model_years",
        "true_cascade_onset_generation",
        "observed_cascade_event_count",
        "observed_cascade_onset_time_model_years",
        "observed_cascade_onset_generation",
        "cascade_level_classification",
        "onset_time_bias_model_years",
        "cascade_termination_reason",
        "cap_termination",
        "observed_warning_event_count",
        "first_collision_time_model_years",
        "execution_time_seconds",
        "estimated_runtime_seconds",
        "estimated_output_size_bytes",
        "expected_condition_count",
        "expected_run_count",
        "model_time_unit",
        "model_dt",
        "model_duration",
        "time_basis_days_per_year",
        "box_width_km",
        "box_height_km",
        "particle_radius_km",
        "minimum_initial_spacing_km",
        "fragmentation_mode",
        "cascade_onset_generation",
        "max_active_objects",
        "max_true_collision_events",
        "cascade_termination_policy",
        "headless_backend",
        "fixed_scientific_config_json",
        "completed_successfully",
        "status",
        "failure_message",
    ]

    existing_rows = _read_existing_raw_rows(output_path, fieldnames)

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in existing_rows:
            writer.writerow(row)
        for simulation_result in simulation_results:
            writer.writerow(_simulation_result_to_csv_row(simulation_result))

    return output_path


def _read_existing_raw_rows(
    output_path: Path,
    fieldnames: list[str],
) -> list[dict[str, object]]:
    """Preserve existing raw rows when resuming into an existing directory."""
    if not config.EXPERIMENT_RESUME_WITHOUT_DUPLICATION or not output_path.exists():
        return []
    with output_path.open(newline="", encoding="utf-8") as csv_file:
        return [
            {field: row.get(field, "") for field in fieldnames}
            for row in csv.DictReader(csv_file)
        ]


def save_population_summary_csv(
    population_statistics: list[PopulationSweepStatistics],
) -> Path:
    """Save one summary CSV row for each population value."""
    output_directory = Path(config.OUTPUT_DIRECTORY)
    output_directory.mkdir(parents=True, exist_ok=True)
    output_path = output_directory / config.MONTE_CARLO_SUMMARY_FILENAME

    fieldnames = [
        "population_value",
        "sigma_km",
        "run_count",
        "collision_probability",
        "mean_collision_count",
        "median_collision_count",
        "mean_first_collision_time_model_years",
        "total_execution_time_seconds",
        "mean_execution_time_seconds",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for statistic in population_statistics:
            writer.writerow(_population_statistic_to_csv_row(statistic))

    return output_path


def calculate_population_sweep_statistics(
    simulation_results: list[SimulationResult],
) -> list[PopulationSweepStatistics]:
    """Calculate requested statistics independently for each population value."""
    statistics_by_population = []
    condition_values = sorted(
        {
            (result.population_value, result.position_uncertainty_sigma_km)
            for result in simulation_results
        }
    )

    for population_value, sigma_km in condition_values:
        population_results = [
            result
            for result in simulation_results
            if result.population_value == population_value
            and result.position_uncertainty_sigma_km == sigma_km
        ]
        statistics_by_population.append(
            _calculate_one_population_statistics(
                population_value,
                sigma_km,
                population_results,
            )
        )

    return statistics_by_population


def print_monte_carlo_batch_summary(
    simulation_results: list[SimulationResult],
    population_statistics: list[PopulationSweepStatistics],
    raw_csv_path: Path,
    summary_csv_path: Path,
    total_batch_execution_time_seconds: float,
) -> None:
    """Print identity checks, output paths, and population-level statistics."""
    print("Monte Carlo population sweep complete:")
    print(
        "- Population values: "
        + _format_result_values(
            [str(statistic.population_value) for statistic in population_statistics]
        )
    )
    print(
        "- Physical seeds per population: "
        + _format_result_values(
            [str(result.run_seed) for result in simulation_results]
        )
    )
    print(
        "- Observation seeds per population: "
        + _format_result_values(
            [str(result.observation_seed) for result in simulation_results]
        )
    )
    print(
        "- Initial states: "
        + _format_result_values(
            [result.initialization_fingerprint for result in simulation_results]
        )
    )
    print(f"- Raw CSV: {raw_csv_path.resolve()}")
    print(f"- Summary CSV: {summary_csv_path.resolve()}")
    print(f"- Total batch execution time: {total_batch_execution_time_seconds:.4f} seconds")

    for statistic in population_statistics:
        print(f"Population {statistic.population_value}:")
        print(f"- Run count: {statistic.total_runs}")
        print(f"- Successful runs: {statistic.successful_runs}")
        print(f"- Runs with collisions: {statistic.runs_with_collisions}")
        print(f"- Collision probability: {statistic.collision_probability:.4f}")
        print(f"- Mean collision count: {statistic.mean_collision_count:.4f}")
        print(f"- Median collision count: {statistic.median_collision_count:.4f}")
        print(
            "- Mean first-collision time: "
            + _format_optional_model_years(statistic.mean_first_collision_time_model_years)
        )
        print(f"- Total execution time: {statistic.total_execution_time_seconds:.4f} seconds")
        print(f"- Mean execution time: {statistic.mean_execution_time_seconds:.4f} seconds")


def _calculate_one_population_statistics(
    population_value: int,
    sigma_km: float,
    simulation_results: list[SimulationResult],
) -> PopulationSweepStatistics:
    """Calculate the summary row for one population value."""
    total_runs = len(simulation_results)
    successful_runs = sum(
        1 for result in simulation_results if result.completed_successfully
    )
    runs_with_collisions = sum(
        1 for result in simulation_results if result.total_particle_collisions > 0
    )
    collision_counts = [
        result.total_particle_collisions for result in simulation_results
    ]
    first_collision_times = [
        result.first_collision_time_model_years
        for result in simulation_results
        if result.first_collision_time_model_years is not None
    ]
    total_execution_time_seconds = sum(
        result.execution_time_seconds for result in simulation_results
    )

    return PopulationSweepStatistics(
        population_value=population_value,
        sigma_km=sigma_km,
        total_runs=total_runs,
        successful_runs=successful_runs,
        runs_with_collisions=runs_with_collisions,
        collision_probability=(
            runs_with_collisions / total_runs if total_runs > 0 else 0.0
        ),
        mean_collision_count=(
            statistics.mean(collision_counts) if collision_counts else 0.0
        ),
        median_collision_count=(
            float(statistics.median(collision_counts)) if collision_counts else 0.0
        ),
        mean_first_collision_time_model_years=(
            statistics.mean(first_collision_times) if first_collision_times else None
        ),
        total_execution_time_seconds=total_execution_time_seconds,
        mean_execution_time_seconds=(
            total_execution_time_seconds / total_runs if total_runs > 0 else 0.0
        ),
    )


def _simulation_result_to_csv_row(
    simulation_result: SimulationResult,
) -> dict[str, object]:
    """Convert one result to a plain row suitable for CSV writing."""
    return {
        "experiment_condition_id": simulation_result.experiment_condition_id,
        "experiment_stage": simulation_result.experiment_stage,
        "population_index": simulation_result.population_index,
        "population_value": simulation_result.population_value,
        "sigma_index": simulation_result.sigma_index,
        "repetition_index": simulation_result.repetition_index,
        "run_index": simulation_result.run_index,
        "run_seed": simulation_result.run_seed,
        "observation_seed": simulation_result.observation_seed,
        "fragmentation_seed": simulation_result.fragmentation_seed,
        "initialization_fingerprint": simulation_result.initialization_fingerprint,
        "configured_duration_model_years": simulation_result.configured_duration_model_years,
        "actual_duration_model_years": simulation_result.actual_duration_model_years,
        "dt_model_years": simulation_result.dt_model_years,
        "position_uncertainty_sigma_km": (
            simulation_result.position_uncertainty_sigma_km
        ),
        "steps_executed": simulation_result.steps_executed,
        "initial_particle_count": simulation_result.initial_particle_count,
        "final_particle_count": simulation_result.final_particle_count,
        "pairs_per_step": simulation_result.pairs_per_step,
        "total_pair_step_comparisons": (
            simulation_result.total_pair_step_comparisons
        ),
        "TP": simulation_result.true_positive_count,
        "FP": simulation_result.false_positive_count,
        "FN": simulation_result.false_negative_count,
        "TN": simulation_result.true_negative_count,
        "true_positive_rate": _format_optional_number(
            simulation_result.true_positive_rate
        ),
        "false_negative_rate": _format_optional_number(
            simulation_result.false_negative_rate
        ),
        "false_positive_rate": _format_optional_number(
            simulation_result.false_positive_rate
        ),
        "precision": _format_optional_number(simulation_result.precision),
        "total_particle_collisions": simulation_result.total_particle_collisions,
        "true_collision_event_count": simulation_result.true_collision_event_count,
        "true_cascade_onset_time_model_years": _format_optional_number(
            simulation_result.true_cascade_onset_time_model_years
        ),
        "true_cascade_onset_generation": _format_optional_number(
            simulation_result.true_cascade_onset_generation
        ),
        "observed_cascade_event_count": simulation_result.observed_cascade_event_count,
        "observed_cascade_onset_time_model_years": _format_optional_number(
            simulation_result.observed_cascade_onset_time_model_years
        ),
        "observed_cascade_onset_generation": _format_optional_number(
            simulation_result.observed_cascade_onset_generation
        ),
        "cascade_level_classification": simulation_result.cascade_level_classification,
        "onset_time_bias_model_years": _format_optional_number(
            simulation_result.onset_time_bias_model_years
        ),
        "cascade_termination_reason": simulation_result.cascade_termination_reason,
        "cap_termination": simulation_result.cap_termination,
        "observed_warning_event_count": simulation_result.observed_warning_event_count,
        "first_collision_time_model_years": _format_optional_number(
            simulation_result.first_collision_time_model_years
        ),
        "execution_time_seconds": simulation_result.execution_time_seconds,
        "estimated_runtime_seconds": simulation_result.estimated_runtime_seconds,
        "estimated_output_size_bytes": simulation_result.estimated_output_size_bytes,
        "expected_condition_count": simulation_result.expected_condition_count,
        "expected_run_count": simulation_result.expected_run_count,
        "model_time_unit": config.MODEL_TIME_UNIT,
        "model_dt": config.MODEL_DT,
        "model_duration": config.MODEL_DURATION,
        "time_basis_days_per_year": config.TIME_BASIS_DAYS_PER_YEAR,
        "box_width_km": config.BOX_WIDTH,
        "box_height_km": config.BOX_HEIGHT,
        "particle_radius_km": config.PARTICLE_RADIUS,
        "minimum_initial_spacing_km": config.MINIMUM_INITIAL_SPACING,
        "fragmentation_mode": config.FRAGMENTATION_MODE,
        "cascade_onset_generation": config.CASCADE_ONSET_GENERATION,
        "max_active_objects": config.MAX_ACTIVE_OBJECTS,
        "max_true_collision_events": config.MAX_TRUE_COLLISION_EVENTS,
        "cascade_termination_policy": config.CASCADE_TERMINATION_POLICY,
        "headless_backend": config.HEADLESS_BACKEND,
        "fixed_scientific_config_json": (
            simulation_result.fixed_scientific_config_json
            or json.dumps(_fixed_scientific_config(), sort_keys=True)
        ),
        "completed_successfully": simulation_result.completed_successfully,
        "status": simulation_result.status,
        "failure_message": simulation_result.failure_message,
    }


def _population_statistic_to_csv_row(
    statistic: PopulationSweepStatistics,
) -> dict[str, object]:
    """Convert one population summary to a CSV row."""
    return {
        "population_value": statistic.population_value,
        "sigma_km": statistic.sigma_km,
        "run_count": statistic.total_runs,
        "collision_probability": statistic.collision_probability,
        "mean_collision_count": statistic.mean_collision_count,
        "median_collision_count": statistic.median_collision_count,
        "mean_first_collision_time_model_years": _format_optional_number(
            statistic.mean_first_collision_time_model_years
        ),
        "total_execution_time_seconds": statistic.total_execution_time_seconds,
        "mean_execution_time_seconds": statistic.mean_execution_time_seconds,
    }


def _fixed_scientific_config() -> dict[str, object]:
    """Return fixed scientific settings copied into every raw run row."""
    return {
        "code_version": config.CODE_VERSION,
        "model_time_unit": config.MODEL_TIME_UNIT,
        "model_dt": config.MODEL_DT,
        "model_duration": config.MODEL_DURATION,
        "time_basis_days_per_year": config.TIME_BASIS_DAYS_PER_YEAR,
        "box_width_km": config.BOX_WIDTH,
        "box_height_km": config.BOX_HEIGHT,
        "particle_radius_km": config.PARTICLE_RADIUS,
        "minimum_initial_spacing_km": config.MINIMUM_INITIAL_SPACING,
        "headless_backend": config.HEADLESS_BACKEND,
        "fragmentation_mode": config.FRAGMENTATION_MODE,
        "cascade_onset_generation": config.CASCADE_ONSET_GENERATION,
        "max_active_objects": config.MAX_ACTIVE_OBJECTS,
        "max_true_collision_events": config.MAX_TRUE_COLLISION_EVENTS,
        "cascade_termination_policy": config.CASCADE_TERMINATION_POLICY,
    }


def _format_optional_number(value: float | None) -> str | float:
    """Keep missing collision times as empty CSV cells."""
    if value is None:
        return ""

    return value


def _format_optional_model_years(value: float | None) -> str:
    """Return readable text for statistics that only exist after collisions."""
    if value is None:
        return "not available"

    return f"{value:.4f} model years"


def _format_result_values(values: list[str]) -> str:
    """Return all values for small batches, or a compact preview for large ones."""
    if len(values) <= 20:
        return ", ".join(values)

    preview = [*values[:5], "...", *values[-5:]]
    return ", ".join(preview)







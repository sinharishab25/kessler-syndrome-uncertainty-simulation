"""Sequential Monte Carlo runner for Version 1.5.15 headless simulations."""

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
import csv
import json

import numpy as np

import config
from analysis.batch_graphs import plot_monte_carlo_batch_graphs
from analysis.experiment_analysis import generate_experiment_analysis_outputs
from simulation.backend import create_headless_simulation
from simulation.batch_results import (
    calculate_population_sweep_statistics,
    print_monte_carlo_batch_summary,
    save_population_summary_csv,
    save_monte_carlo_results_csv,
)
from simulation.cascade_outputs import save_cascade_output_tables
from simulation.experiment import (
    create_experiment_output_directory,
    write_experiment_metadata,
)
from simulation.headless import run_headless_simulation
from simulation.progress import should_print_progress, should_print_summary
from simulation.result import SimulationResult


@dataclass(frozen=True)
class ExperimentCondition:
    """One population x sigma x repetition condition."""

    population_index: int
    population_value: int
    sigma_index: int
    sigma_km: float
    repetition_index: int

    @property
    def run_index(self) -> int:
        return self.repetition_index

    @property
    def condition_id(self) -> str:
        sigma_label = str(self.sigma_km).replace(".", "p")
        return f"N{self.population_value}_sigma{sigma_label}_rep{self.repetition_index}"


def run_monte_carlo_simulations() -> list[SimulationResult]:
    """Run a sequential factorial experiment and save one primary raw row per run."""
    output_directory = create_experiment_output_directory("headless_batch")
    results = []
    conditions = build_experiment_conditions()
    completed_condition_ids = _read_completed_condition_ids(output_directory)
    runnable_conditions = [
        condition
        for condition in conditions
        if condition.condition_id not in completed_condition_ids
    ]
    total_runs = len(runnable_conditions)
    progress_interval = max(1, total_runs // 10)
    batch_start_time = perf_counter()
    completed_runs = 0
    estimate = estimate_experiment_workload(conditions)
    write_experiment_metadata(
        output_directory,
        "headless_batch",
        "running",
        extra_metadata={"experiment_estimate": estimate},
    )

    try:
        if not runnable_conditions:
            raw_csv_path = output_directory / config.MONTE_CARLO_RESULTS_FILENAME
            if raw_csv_path.exists():
                generate_experiment_analysis_outputs(raw_csv_path)
            write_experiment_metadata(
                output_directory,
                "headless_batch",
                "success",
                results=results,
                extra_metadata={
                    "experiment_estimate": estimate,
                    "skipped_completed_conditions": len(completed_condition_ids),
                    "new_conditions_run": 0,
                },
            )
            return results

        for condition in runnable_conditions:
            completed_runs += 1
            if should_print_progress(completed_runs, total_runs, progress_interval):
                print(
                    "Monte Carlo progress: "
                    f"run {completed_runs} of {total_runs} ({condition.condition_id})"
                )

            try:
                results.append(_run_one_headless_simulation(condition, len(conditions)))
            except Exception as error:
                results.append(
                    _create_failed_simulation_result(condition, len(conditions), error)
                )

        total_batch_execution_time_seconds = perf_counter() - batch_start_time
        raw_csv_path = save_monte_carlo_results_csv(results)
        population_statistics = calculate_population_sweep_statistics(results)
        summary_csv_path = save_population_summary_csv(population_statistics)
        save_cascade_output_tables(results)
        plot_monte_carlo_batch_graphs(raw_csv_path, summary_csv_path)
        generate_experiment_analysis_outputs(raw_csv_path)
        if should_print_summary():
            print_monte_carlo_batch_summary(
                results,
                population_statistics,
                raw_csv_path,
                summary_csv_path,
                total_batch_execution_time_seconds,
            )
        write_experiment_metadata(
            output_directory,
            "headless_batch",
            "success",
            results=results,
            extra_metadata={
                "experiment_estimate": estimate,
                "skipped_completed_conditions": len(completed_condition_ids),
                "new_conditions_run": len(runnable_conditions),
            },
        )
        return results
    except Exception as error:
        write_experiment_metadata(
            output_directory,
            "headless_batch",
            "failed",
            failure_message=str(error),
            results=results,
            extra_metadata={"experiment_estimate": estimate},
        )
        raise


def build_experiment_conditions() -> list[ExperimentCondition]:
    """Return deterministic population x sigma x repetition conditions."""
    conditions = []
    for population_index, population_value in enumerate(config.POPULATION_SWEEP_VALUES):
        for repetition_index in range(config.MONTE_CARLO_RUN_COUNT):
            for sigma_index, sigma_km in enumerate(config.UNCERTAINTY_SIGMA_SWEEP_VALUES):
                conditions.append(
                    ExperimentCondition(
                        population_index=population_index,
                        population_value=population_value,
                        sigma_index=sigma_index,
                        sigma_km=float(sigma_km),
                        repetition_index=repetition_index,
                    )
                )
    return conditions


def estimate_experiment_workload(
    conditions: list[ExperimentCondition] | None = None,
) -> dict[str, object]:
    """Estimate runtime and output size without running physics."""
    conditions = build_experiment_conditions() if conditions is None else conditions
    duration_years = config.get_model_duration_years()
    step_count = config.calculate_step_count(
        duration_years,
        config.get_effective_model_dt_years(),
    )
    pair_steps = sum(
        condition.population_value
        * (condition.population_value - 1)
        // 2
        * step_count
        for condition in conditions
    )
    bin_width_years = max(
        config.model_time_config_value_to_years(config.CASCADE_TIME_BIN_WIDTH_MODEL_TIME),
        1e-12,
    )
    time_bins_per_run = int(duration_years / bin_width_years) + 2
    estimated_output_size_bytes = len(conditions) * (
        config.EXPERIMENT_OUTPUT_ESTIMATE_BYTES_PER_RUN
        + time_bins_per_run * config.EXPERIMENT_OUTPUT_ESTIMATE_BYTES_PER_TIME_BIN
    )
    return {
        "stage": config.EXPERIMENT_STAGE,
        "condition_count": len(conditions),
        "expected_run_count": len(conditions),
        "step_count_per_run": step_count,
        "pair_step_comparisons": pair_steps,
        "estimated_runtime_seconds": (
            pair_steps * config.EXPERIMENT_RUNTIME_ESTIMATE_SECONDS_PER_PAIR_STEP
        ),
        "estimated_output_size_bytes": estimated_output_size_bytes,
    }


def _run_one_headless_simulation(
    condition: ExperimentCondition,
    expected_run_count: int,
) -> SimulationResult:
    """Create a fresh RNG and Simulation for one independent condition."""
    run_seed = config.get_run_seed(condition.repetition_index)
    random_generator = np.random.default_rng(run_seed)
    simulation = create_headless_simulation(
        random_generator=random_generator,
        run_seed=run_seed,
        run_index=condition.run_index,
        particle_groups=config.get_particle_groups_for_population(
            condition.population_value
        ),
        population_value=condition.population_value,
        observation_seed=config.get_condition_observation_seed(
            condition.population_index,
            condition.sigma_index,
            condition.repetition_index,
        ),
        position_uncertainty_sigma_km=condition.sigma_km,
    )
    result = run_headless_simulation(simulation)
    _attach_condition_metadata(result, condition, expected_run_count)
    return result


def _create_failed_simulation_result(
    condition: ExperimentCondition,
    expected_run_count: int,
    error: Exception,
) -> SimulationResult:
    """Return a primary-output row for a failed attempted condition."""
    observation_seed = config.get_condition_observation_seed(
        condition.population_index,
        condition.sigma_index,
        condition.repetition_index,
    )
    result = SimulationResult(
        run_index=condition.run_index,
        run_seed=config.get_run_seed(condition.repetition_index),
        observation_seed=observation_seed,
        population_value=condition.population_value,
        configured_duration_model_years=config.get_model_duration_years(),
        actual_duration_model_years=0.0,
        steps_executed=0,
        initial_particle_count=0,
        final_particle_count=0,
        initialization_fingerprint="",
        total_particle_collisions=0,
        first_collision_time_model_years=None,
        collision_times_model_years=[],
        true_collision_event_count=0,
        true_cascade_onset_time_model_years=None,
        true_cascade_onset_generation=None,
        position_uncertainty_sigma_km=condition.sigma_km,
        dt_model_years=config.get_effective_model_dt_years(),
        pairs_per_step=0,
        total_pair_step_comparisons=0,
        true_positive_count=0,
        false_positive_count=0,
        false_negative_count=0,
        true_negative_count=0,
        true_positive_rate=None,
        false_negative_rate=None,
        false_positive_rate=None,
        precision=None,
        observed_warning_event_count=0,
        execution_time_seconds=0.0,
        steps_per_second=0.0,
        completed_successfully=False,
        status="failed",
        failure_message=str(error),
        fragmentation_seed=config.get_fragmentation_seed(condition.repetition_index),
    )
    _attach_condition_metadata(result, condition, expected_run_count)
    return result


def _attach_condition_metadata(
    result: SimulationResult,
    condition: ExperimentCondition,
    expected_run_count: int,
) -> None:
    """Attach factorial metadata without changing scientific simulation fields."""
    result.repetition_index = condition.repetition_index
    result.population_index = condition.population_index
    result.sigma_index = condition.sigma_index
    result.experiment_stage = config.EXPERIMENT_STAGE
    result.experiment_condition_id = condition.condition_id
    result.expected_condition_count = expected_run_count
    result.expected_run_count = expected_run_count
    result.fixed_scientific_config_json = _fixed_scientific_config_json()
    result.cap_termination = "cap" in result.cascade_termination_reason.lower()
    result.estimated_runtime_seconds = _estimate_condition_runtime_seconds(condition)
    result.estimated_output_size_bytes = _estimate_condition_output_bytes()


def _read_completed_condition_ids(output_directory: Path) -> set[str]:
    """Return already successful condition IDs for resume-without-duplication."""
    if not config.EXPERIMENT_RESUME_WITHOUT_DUPLICATION:
        return set()
    raw_path = output_directory / config.MONTE_CARLO_RESULTS_FILENAME
    if not raw_path.exists():
        return set()

    with raw_path.open(newline="", encoding="utf-8") as raw_file:
        return {
            row.get("experiment_condition_id", "")
            for row in csv.DictReader(raw_file)
            if row.get("experiment_condition_id") and row.get("status") == "success"
        }


def _fixed_scientific_config_json() -> str:
    """Return fixed model parameters copied into each run row."""
    return json.dumps(
        {
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
        },
        sort_keys=True,
    )


def _estimate_condition_runtime_seconds(condition: ExperimentCondition) -> float:
    step_count = config.calculate_step_count(
        config.get_model_duration_years(),
        config.get_effective_model_dt_years(),
    )
    pair_count = condition.population_value * (condition.population_value - 1) // 2
    return (
        pair_count
        * step_count
        * config.EXPERIMENT_RUNTIME_ESTIMATE_SECONDS_PER_PAIR_STEP
    )


def _estimate_condition_output_bytes() -> int:
    duration_years = config.get_model_duration_years()
    bin_width_years = max(
        config.model_time_config_value_to_years(config.CASCADE_TIME_BIN_WIDTH_MODEL_TIME),
        1e-12,
    )
    time_bins = int(duration_years / bin_width_years) + 2
    return (
        config.EXPERIMENT_OUTPUT_ESTIMATE_BYTES_PER_RUN
        + time_bins * config.EXPERIMENT_OUTPUT_ESTIMATE_BYTES_PER_TIME_BIN
    )

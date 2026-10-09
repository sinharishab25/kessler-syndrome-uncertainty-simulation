"""Headless execution flow for Version 1.5.15 numerical simulation runs."""

from time import perf_counter

import config
from simulation.cascade_metrics import classify_cascade_level, onset_time_bias
from simulation.result import SimulationResult
from simulation.time_units import calculate_step_count, warn_deprecated_alias


def run_headless_simulation(
    simulation,
    duration_model_time: float | None = None,
    *,
    duration_seconds: float | None = None,
) -> SimulationResult:
    """Advance a simulation numerically without creating any Matplotlib UI."""
    if duration_seconds is not None:
        warn_deprecated_alias("duration_seconds", "duration_model_time")
        if duration_model_time is not None:
            raise ValueError(
                "Use only duration_model_time; duration_seconds is deprecated."
            )
        duration_model_time = duration_seconds

    target_duration_model_years = _get_headless_duration_model_years(
        duration_model_time
    )
    configured_step_model_years = float(simulation.time_step)
    step_count = calculate_step_count(
        target_duration_model_years,
        configured_step_model_years,
    )
    simulation.record_position_history = False
    simulation.position_history = []
    initial_time = simulation.current_time
    initial_frame = simulation.current_frame
    initial_particle_count = _get_active_particle_count(simulation)
    start_time = perf_counter()

    for _step_index in range(step_count):
        if getattr(simulation, "cascade_terminated", False):
            break
        elapsed_model_years = simulation.current_time - initial_time
        remaining_model_years = target_duration_model_years - elapsed_model_years
        if remaining_model_years <= 0.0:
            break
        simulation.step(min(configured_step_model_years, remaining_model_years))

    execution_time_seconds = perf_counter() - start_time
    steps_executed = simulation.current_frame - initial_frame
    steps_per_second = (
        steps_executed / execution_time_seconds
        if execution_time_seconds > 0.0
        else 0.0
    )
    collision_times_model_years = [
        record["simulation_time_model_years"]
        for record in simulation.particle_collision_records
    ]
    pair_step_summary = simulation.get_pair_step_summary()
    simulation.close_pair_step_diagnostic_data()
    true_onset = pair_step_summary["true_cascade_onset_time_model_years"]
    observed_onset = pair_step_summary["observed_cascade_onset_time_model_years"]

    return SimulationResult(
        run_index=simulation.run_index,
        run_seed=simulation.run_seed,
        observation_seed=simulation.observation_seed,
        population_value=simulation.population_value,
        configured_duration_model_years=target_duration_model_years,
        actual_duration_model_years=simulation.current_time - initial_time,
        steps_executed=steps_executed,
        initial_particle_count=initial_particle_count,
        final_particle_count=_get_active_particle_count(simulation),
        initialization_fingerprint=simulation.initialization_fingerprint,
        total_particle_collisions=simulation.particle_collision_count,
        first_collision_time_model_years=(
            collision_times_model_years[0] if collision_times_model_years else None
        ),
        collision_times_model_years=collision_times_model_years,
        true_collision_event_count=pair_step_summary["true_collision_event_count"],
        true_cascade_onset_time_model_years=(
            pair_step_summary["true_cascade_onset_time_model_years"]
        ),
        true_cascade_onset_generation=pair_step_summary[
            "true_cascade_onset_generation"
        ],
        observed_cascade_event_count=pair_step_summary["observed_cascade_event_count"],
        observed_cascade_onset_time_model_years=observed_onset,
        observed_cascade_onset_generation=pair_step_summary[
            "observed_cascade_onset_generation"
        ],
        position_uncertainty_sigma_km=simulation.position_uncertainty_sigma_km,
        dt_model_years=simulation.time_step,
        pairs_per_step=pair_step_summary["pairs_per_step"],
        total_pair_step_comparisons=pair_step_summary["total_pair_step_comparisons"],
        true_positive_count=pair_step_summary["TP"],
        false_positive_count=pair_step_summary["FP"],
        false_negative_count=pair_step_summary["FN"],
        true_negative_count=pair_step_summary["TN"],
        true_positive_rate=pair_step_summary["true_positive_rate"],
        false_negative_rate=pair_step_summary["false_negative_rate"],
        false_positive_rate=pair_step_summary["false_positive_rate"],
        precision=pair_step_summary["precision"],
        observed_warning_event_count=(
            pair_step_summary["observed_deduplicated_warning_event_count"]
        ),
        execution_time_seconds=execution_time_seconds,
        steps_per_second=steps_per_second,
        completed_successfully=True,
        fragmentation_seed=getattr(
            simulation,
            "fragmentation_seed",
            config.get_fragmentation_seed(simulation.run_index),
        ),
        cascade_termination_reason=getattr(simulation, "cascade_termination_reason", ""),
        true_collision_events=list(getattr(simulation, "true_collision_events", [])),
        cascade_level_classification=classify_cascade_level(true_onset, observed_onset),
        onset_time_bias_model_years=onset_time_bias(true_onset, observed_onset),
        observed_cascade_events=list(getattr(simulation, "observed_cascade_events", [])),
    )


def _get_headless_duration_model_years(duration_model_time: float | None) -> float:
    """Return a validated canonical model-year duration for a headless run."""
    if duration_model_time is None:
        return config.get_model_duration_years()

    numeric_duration = float(duration_model_time)
    if numeric_duration <= 0.0:
        raise ValueError("duration_model_time must be greater than 0.")

    return config.model_time_config_value_to_years(numeric_duration)


def _get_active_particle_count(simulation) -> int:
    """Return active count for either object or array backends."""
    if hasattr(simulation, "get_active_particle_count"):
        return simulation.get_active_particle_count()

    return len(simulation.particles)


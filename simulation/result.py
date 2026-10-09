"""Structured result object for Version 1.5.15 simulation runs."""

from dataclasses import dataclass, field


@dataclass
class SimulationResult:
    """Compact summary of one completed headless simulation run."""

    run_index: int
    run_seed: int
    observation_seed: int
    population_value: int
    configured_duration_model_years: float
    actual_duration_model_years: float
    steps_executed: int
    initial_particle_count: int
    final_particle_count: int
    initialization_fingerprint: str
    total_particle_collisions: int
    first_collision_time_model_years: float | None
    collision_times_model_years: list[float]
    true_collision_event_count: int
    true_cascade_onset_time_model_years: float | None
    true_cascade_onset_generation: int | None
    position_uncertainty_sigma_km: float
    dt_model_years: float
    pairs_per_step: int
    total_pair_step_comparisons: int
    true_positive_count: int
    false_positive_count: int
    false_negative_count: int
    true_negative_count: int
    true_positive_rate: float | None
    false_negative_rate: float | None
    false_positive_rate: float | None
    precision: float | None
    observed_warning_event_count: int
    execution_time_seconds: float = field(compare=False)
    steps_per_second: float = field(compare=False)
    completed_successfully: bool
    status: str = "success"
    failure_message: str = ""
    fragmentation_seed: int | None = None
    cascade_termination_reason: str = ""
    true_collision_events: list = field(default_factory=list, compare=False)
    observed_cascade_event_count: int = 0
    observed_cascade_onset_time_model_years: float | None = None
    observed_cascade_onset_generation: int | None = None
    cascade_level_classification: str = "TN"
    onset_time_bias_model_years: float | None = None
    observed_cascade_events: list = field(default_factory=list, compare=False)
    repetition_index: int = 0
    population_index: int = 0
    sigma_index: int = 0
    experiment_stage: str = ""
    experiment_condition_id: str = ""
    expected_condition_count: int = 0
    expected_run_count: int = 0
    fixed_scientific_config_json: str = ""
    cap_termination: bool = False
    estimated_runtime_seconds: float = 0.0
    estimated_output_size_bytes: int = 0

    @property
    def configured_duration_seconds(self) -> float:
        """Deprecated alias for the old ambiguous model-time field name."""
        from simulation.time_units import warn_deprecated_alias

        warn_deprecated_alias(
            "configured_duration_seconds",
            "configured_duration_model_years",
        )
        return self.configured_duration_model_years

    @property
    def actual_duration_seconds(self) -> float:
        """Deprecated alias for the old ambiguous model-time field name."""
        from simulation.time_units import warn_deprecated_alias

        warn_deprecated_alias(
            "actual_duration_seconds",
            "actual_duration_model_years",
        )
        return self.actual_duration_model_years

    @property
    def first_collision_time_seconds(self) -> float | None:
        """Deprecated alias for the old ambiguous model-time field name."""
        from simulation.time_units import warn_deprecated_alias

        warn_deprecated_alias(
            "first_collision_time_seconds",
            "first_collision_time_model_years",
        )
        return self.first_collision_time_model_years

    @property
    def collision_times_seconds(self) -> list[float]:
        """Deprecated alias for the old ambiguous model-time field name."""
        from simulation.time_units import warn_deprecated_alias

        warn_deprecated_alias(
            "collision_times_seconds",
            "collision_times_model_years",
        )
        return self.collision_times_model_years

    @property
    def dt_seconds(self) -> float:
        """Deprecated alias for the old ambiguous model-time field name."""
        from simulation.time_units import warn_deprecated_alias

        warn_deprecated_alias("dt_seconds", "dt_model_years")
        return self.dt_model_years

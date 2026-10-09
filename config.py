"""Central configuration for Version 1.5.15 Kessler-style LEO simulations.

All experiment parameters should live here so runs can be adjusted without
changing the simulation, physics, visualization, or analysis code.
"""

from pathlib import Path

from simulation.time_units import (
    SUPPORTED_MODEL_TIME_UNITS,
    calculate_step_count,
    model_time_to_years,
    model_years_to_unit,
    warn_deprecated_alias,
)


# Run mode
CODE_VERSION = "v1.5.15"
RUN_MODE = "visual"
SUPPORTED_RUN_MODES = ["visual", "headless", "benchmark", "profile"]
HEADLESS_BACKEND = "array"
SUPPORTED_HEADLESS_BACKENDS = ["array", "object"]


# Model geometry
# Coordinates are kilometers in a simplified LEO cross-section, not pixels.
BOX_WIDTH = 1000.0
BOX_HEIGHT = 600.0
DISTANCE_UNIT = "km"
AREA_UNIT = "km^2"
SIMULATION_REGION_NAME = "Simplified Low Earth Orbit cross-section"
SIMULATION_REGION_INTERPRETATION = (
    "A flat 1000 km by 600 km slice through a representative LEO shell. "
    "It is useful for density and relative-motion experiments, but it is not "
    "a full orbital mechanics model around a curved Earth."
)
REPRESENTATIVE_LEO_ALTITUDE_RANGE_KM = (160, 2000)
REPRESENTATIVE_LEO_SPEED_RANGE_KM_PER_PHYSICAL_SECOND = (7.0, 8.0)


# Satellite population
ACTIVE_SATELLITE_COUNT = 5
TEST_SATELLITE_COUNT = 1
NUMBER_OF_PARTICLES = ACTIVE_SATELLITE_COUNT + TEST_SATELLITE_COUNT


# Satellite properties
# The radius is a visible research marker radius, not a real spacecraft radius.
PARTICLE_RADIUS = 1.2
MINIMUM_INITIAL_SPACING = 55.0
MAX_INITIAL_POSITION_ATTEMPTS = 50000
POSITION_UNCERTAINTY_SIGMA_KM = 0.0

# Speeds are scaled km/model-year and are sampled uniformly when satellites are
# initialized. These are educational exposure-time speeds, not orbital
# propagation velocities.
PARTICLE_GROUPS = [
    {
        "name": "Active satellite",
        "key": "active",
        "count": ACTIVE_SATELLITE_COUNT,
        "color": "tab:blue",
        "legend_label": "Blue = Active satellites",
        "identity_prefix": "S",
        "speed_range_km_per_model_year": (7.55, 7.85),
    },
    {
        "name": "Test satellite",
        "key": "test",
        "count": TEST_SATELLITE_COUNT,
        "color": "tab:orange",
        "legend_label": "Orange = Test satellite",
        "identity_prefix": "T",
        "speed_range_km_per_model_year": (7.20, 8.00),
    },
]


# Simulation timing
# MODEL_DT and MODEL_DURATION are expressed in MODEL_TIME_UNIT, then converted
# to canonical internal model years. Model years are scaled educational exposure
# time, not real orbital propagation time.
MODEL_TIME_UNIT = "years"
MODEL_DT = 0.01
MODEL_DURATION = 1000.0
TIME_BASIS_DAYS_PER_YEAR = 365.25
TOTAL_FRAMES = 1000
FPS = 30
SIMULATION_SPEED_MULTIPLIER = 10
SUPPORTED_SPEED_MULTIPLIERS = [1, 2, 3, 4, 5, 10, 50, 100, 500, 1000]
DISPLAY_TIME_UNIT = "years"
MAX_PHYSICS_SUBSTEP_MODEL_TIME = 0.5
PARTICLE_COLLISION_COOLDOWN_MODEL_TIME = 0.0
MAX_MODEL_TIME_STEPS = 1_000_000
ALLOW_EXTREME_MODEL_TIME_WORKLOAD = False

# Performance and display settings.
SHOW_LABELS = True
SHOW_VELOCITY_ARROWS = False
INFORMATION_UPDATE_INTERVAL_FRAMES = 5
TRAJECTORY_RECORD_INTERVAL_FRAMES = 5
RECORD_OBSERVATION_HISTORY = False
MAX_RETAINED_PAIR_STEP_CLASSIFICATIONS = 100
PAIR_CHUNK_SIZE = 100000
PROGRESS_MODE = "coarse"
SUPPORTED_PROGRESS_MODES = ["silent", "coarse", "verbose"]


# Random seed
# Each run uses BASE_RANDOM_SEED + run_index, so run 0 is repeatable and later
# Monte Carlo runs can get different controlled seeds without global randomness.
BASE_RANDOM_SEED = 12345
BASE_OBSERVATION_RANDOM_SEED = 54321
FRAGMENTATION_RANDOM_SEED_OFFSET = 1_000_003


# Educational true-cascade and fragmentation settings.
FRAGMENTATION_MODE = "size-dependent-stochastic"
SUPPORTED_FRAGMENTATION_MODES = ["disabled", "fixed-count", "size-dependent-stochastic"]
CASCADE_ONSET_GENERATION = 3
FIXED_FRAGMENT_COUNT = 0
SIZE_DEPENDENT_FRAGMENT_BASE_COUNT = 2
SIZE_DEPENDENT_FRAGMENT_COUNT_PER_KM_RADIUS = 1.0
SIZE_DEPENDENT_FRAGMENT_COUNT_STD = 0.0
FRAGMENT_RADIUS_DISTRIBUTION = "uniform"
SUPPORTED_FRAGMENT_RADIUS_DISTRIBUTIONS = ["uniform", "constant"]
FRAGMENT_RADIUS_MIN_KM = 0.15
FRAGMENT_RADIUS_MAX_KM = 0.35
FRAGMENT_RADIUS_CONSTANT_KM = 0.25
FRAGMENT_VELOCITY_REFERENCE = "center-of-mass"
SUPPORTED_FRAGMENT_VELOCITY_REFERENCES = ["center-of-mass", "parent-i", "parent-j"]
FRAGMENT_VELOCITY_KICK_DISTRIBUTION = "normal"
SUPPORTED_FRAGMENT_VELOCITY_KICK_DISTRIBUTIONS = ["normal", "uniform"]
FRAGMENT_VELOCITY_KICK_SCALE_KM_PER_MODEL_YEAR = 1.0
FRAGMENT_RELEASE_OFFSET_FACTOR = 2.25
MAX_ACTIVE_OBJECTS = 1000
MAX_TRUE_COLLISION_EVENTS = 100
CASCADE_TERMINATION_POLICY = "stop"
SUPPORTED_CASCADE_TERMINATION_POLICIES = ["stop", "disable-fragmentation"]


# Output settings
OUTPUT_DIRECTORY = "outputs"
BOUNDARY_CROSSING_DATA_FILENAME = "boundary_crossings.csv"
PARTICLE_COLLISION_DATA_FILENAME = "particle_collisions.csv"
BOUNDARY_CROSSING_GRAPH_FILENAME = "boundary_crossings_over_time.png"
PARTICLE_COLLISION_GRAPH_FILENAME = "particle_collisions_over_time.png"
PARTICLE_COLLISION_RATE_GRAPH_FILENAME = "particle_collision_rate_over_time.png"
TRAJECTORY_GRAPH_FILENAME = "particle_trajectories.png"
SIMULATION_REPORT_FILENAME = "simulation_report_v1_5_15.md"
MONTE_CARLO_RESULTS_FILENAME = "monte_carlo_raw_results_v1_5_15.csv"
MONTE_CARLO_SUMMARY_FILENAME = "monte_carlo_population_summary_v1_5_15.csv"
MONTE_CARLO_COLLISION_PROBABILITY_GRAPH_FILENAME = (
    "monte_carlo_collision_probability_by_population.png"
)
MONTE_CARLO_MEAN_COLLISION_COUNT_GRAPH_FILENAME = (
    "monte_carlo_mean_collision_count_by_population.png"
)
MONTE_CARLO_CUMULATIVE_COLLISION_PROBABILITY_GRAPH_FILENAME = (
    "monte_carlo_cumulative_collision_probability_by_run.png"
)
BENCHMARK_RESULTS_FILENAME = "monte_carlo_benchmark_results_v1_5_15.csv"
PAIR_STEP_SUMMARY_FILENAME = "pair_step_summary_v1_5_15.csv"
PAIR_STEP_DIAGNOSTIC_FILENAME = "pair_step_diagnostics_v1_5_15.csv"
TRUE_COLLISION_EVENTS_FILENAME = "true_collision_events_v1_5_15.csv"
OBSERVED_CASCADE_EVENTS_FILENAME = "observed_cascade_events_v1_5_15.csv"
CASCADE_EVENT_TABLE_FILENAME = "cascade_true_events_v1_5_15.csv"
CASCADE_OBSERVED_EVENT_TABLE_FILENAME = "cascade_observed_events_v1_5_15.csv"
CASCADE_RUN_TABLE_FILENAME = "cascade_runs_v1_5_15.csv"
CASCADE_TIME_BIN_TABLE_FILENAME = "cascade_time_bins_v1_5_15.csv"
CASCADE_LEVEL_METRICS_FILENAME = "cascade_level_metrics_v1_5_15.csv"
EXPERIMENT_ANALYSIS_SUMMARY_FILENAME = "experiment_analysis_summary_v1_5_15.csv"
EXPERIMENT_ANALYSIS_ONSET_SUMMARY_FILENAME = "experiment_onset_summary_v1_5_15.csv"
EXPERIMENT_ANALYSIS_DIAGNOSTICS_FILENAME = "experiment_diagnostics_v1_5_15.csv"
CASCADE_CONFIG_MANIFEST_FILENAME = "cascade_experiment_manifest_v1_5_15.json"
PROFILE_RESULTS_FILENAME = "component_profile_results_v1_5_15.csv"
SAVE_PAIR_STEP_DIAGNOSTIC_CSV = False
USE_UNIQUE_EXPERIMENT_DIRECTORIES = True
EXPERIMENT_METADATA_FILENAME = "experiment_metadata.json"
CASCADE_TIME_BIN_WIDTH_MODEL_TIME = 1.0
CASCADE_ROLLING_WINDOW_MODEL_TIME = 5.0
CASCADE_PLOT_INDIVIDUAL_TRAJECTORIES = True
CASCADE_MAX_INDIVIDUAL_TRAJECTORIES = 5
CASCADE_ACTIVE_POPULATION_GRAPH_FILENAME = "cascade_active_population_v1_5_15.png"
CASCADE_DEBRIS_POPULATION_GRAPH_FILENAME = "cascade_debris_population_v1_5_15.png"
CASCADE_COLLISION_RATE_GRAPH_FILENAME = "cascade_collision_rate_v1_5_15.png"
CASCADE_CUMULATIVE_COLLISIONS_GRAPH_FILENAME = "cascade_cumulative_collisions_v1_5_15.png"
CASCADE_MAX_DEPTH_GRAPH_FILENAME = "cascade_max_depth_v1_5_15.png"
CASCADE_ROLLING_RATE_GRAPH_FILENAME = "cascade_rolling_collision_rate_v1_5_15.png"
CASCADE_DEBRIS_GROWTH_RATE_GRAPH_FILENAME = "cascade_debris_growth_rate_v1_5_15.png"
CASCADE_PRECISION_RECALL_GRAPH_FILENAME = "cascade_precision_recall_by_population_sigma_v1_5_15.png"
CASCADE_PROBABILITY_GRAPH_FILENAME = "cascade_probability_by_population_sigma_v1_5_15.png"
CASCADE_ONSET_SUMMARY_GRAPH_FILENAME = "cascade_onset_summary_by_population_sigma_v1_5_15.png"
CASCADE_ONSET_BIAS_GRAPH_FILENAME = "cascade_onset_bias_distribution_v1_5_15.png"
CASCADE_FALSE_MISSED_GRAPH_FILENAME = "cascade_false_missed_probability_v1_5_15.png"
EXPERIMENT_RUNTIME_DIAGNOSTIC_GRAPH_FILENAME = "experiment_runtime_diagnostics_v1_5_15.png"
EXPERIMENT_FAILURE_CAP_DIAGNOSTIC_GRAPH_FILENAME = "experiment_failure_cap_diagnostics_v1_5_15.png"


# Monte Carlo experiment settings.
# Version 1.5.15 runs one condition for every population x sigma x repetition.
MONTE_CARLO_RUN_COUNT = 100
POPULATION_SWEEP_VALUES = [6,10]
UNCERTAINTY_SIGMA_SWEEP_VALUES = [0.0,1.0,2.0,4.0]
EXPERIMENT_STAGE = "production"
SUPPORTED_EXPERIMENT_STAGES = ["smoke", "pilot", "production"]
EXPERIMENT_RESUME_WITHOUT_DUPLICATION = True
EXPERIMENT_RUNTIME_ESTIMATE_SECONDS_PER_PAIR_STEP = 2.0e-6
EXPERIMENT_OUTPUT_ESTIMATE_BYTES_PER_RUN = 1500
EXPERIMENT_OUTPUT_ESTIMATE_BYTES_PER_TIME_BIN = 500
EXPERIMENT_SURVIVAL_MIN_RUNS = 20


# Benchmark settings
# Benchmark mode is separate from scientific Monte Carlo sweep mode. It measures
# runtime for configured combinations and does not create normal graphs.
BENCHMARK_POPULATION_VALUES = [NUMBER_OF_PARTICLES]
BENCHMARK_MONTE_CARLO_RUN_COUNTS = [MONTE_CARLO_RUN_COUNT]
BENCHMARK_DURATION_MODEL_TIME_VALUES = [MODEL_DURATION]
BENCHMARK_FEASIBILITY_TARGET_POPULATION = 1000
BENCHMARK_FEASIBILITY_TARGET_RUN_COUNT = 10000


# Component profiling settings
# Profiling is opt-in through RUN_MODE = "profile"; normal modes do not attach
# a profiler or collect component timings.
PROFILE_POPULATION_VALUES = [25, 50, 100, 200]
PROFILE_DURATION_MODEL_TIME = 5.0
PROFILE_MINIMUM_INITIAL_SPACING = 2.0 * PARTICLE_RADIUS


# Display time conversions. The canonical internal model-time unit is years.
SUPPORTED_DISPLAY_TIME_UNITS = SUPPORTED_MODEL_TIME_UNITS


def validate_configuration() -> None:
    """Check experiment settings before any simulation objects are created."""
    _validate_run_mode()
    _validate_headless_backend()
    _validate_positive_number("BOX_WIDTH", BOX_WIDTH)
    _validate_positive_number("BOX_HEIGHT", BOX_HEIGHT)
    _validate_non_negative_integer("ACTIVE_SATELLITE_COUNT", ACTIVE_SATELLITE_COUNT)
    _validate_non_negative_integer("TEST_SATELLITE_COUNT", TEST_SATELLITE_COUNT)

    if NUMBER_OF_PARTICLES <= 0:
        raise ValueError("Total satellite count must be greater than 0.")

    _validate_positive_number("PARTICLE_RADIUS", PARTICLE_RADIUS)
    _validate_non_negative_number(
        "POSITION_UNCERTAINTY_SIGMA_KM",
        POSITION_UNCERTAINTY_SIGMA_KM,
    )
    _validate_speed_range(
        "REPRESENTATIVE_LEO_SPEED_RANGE_KM_PER_PHYSICAL_SECOND",
        REPRESENTATIVE_LEO_SPEED_RANGE_KM_PER_PHYSICAL_SECOND,
    )
    for particle_group in PARTICLE_GROUPS:
        _validate_speed_range(
            f"{particle_group['name']} speed_range_km_per_model_year",
            particle_group["speed_range_km_per_model_year"],
        )

    _validate_model_time_settings()
    _validate_non_negative_integer("BASE_RANDOM_SEED", BASE_RANDOM_SEED)
    _validate_non_negative_integer(
        "BASE_OBSERVATION_RANDOM_SEED",
        BASE_OBSERVATION_RANDOM_SEED,
    )
    _validate_non_negative_integer(
        "FRAGMENTATION_RANDOM_SEED_OFFSET",
        FRAGMENTATION_RANDOM_SEED_OFFSET,
    )
    _validate_boolean(
        "SAVE_PAIR_STEP_DIAGNOSTIC_CSV",
        SAVE_PAIR_STEP_DIAGNOSTIC_CSV,
    )
    _validate_boolean(
        "RECORD_OBSERVATION_HISTORY",
        RECORD_OBSERVATION_HISTORY,
    )
    _validate_boolean(
        "USE_UNIQUE_EXPERIMENT_DIRECTORIES",
        USE_UNIQUE_EXPERIMENT_DIRECTORIES,
    )
    _validate_non_negative_integer(
        "MAX_RETAINED_PAIR_STEP_CLASSIFICATIONS",
        MAX_RETAINED_PAIR_STEP_CLASSIFICATIONS,
    )
    _validate_positive_integer("PAIR_CHUNK_SIZE", PAIR_CHUNK_SIZE)
    _validate_progress_mode()
    _validate_cascade_output_settings()
    _validate_positive_integer("MONTE_CARLO_RUN_COUNT", MONTE_CARLO_RUN_COUNT)
    _validate_population_sweep_values()
    _validate_uncertainty_sigma_sweep_values()
    _validate_experiment_settings()
    _validate_benchmark_settings()
    _validate_profile_settings()
    _validate_fragmentation_settings()
    _validate_model_time_workload()
    Path(OUTPUT_DIRECTORY).mkdir(parents=True, exist_ok=True)


def get_model_dt_years() -> float:
    """Return the configured base step in canonical model years."""
    return model_time_to_years(MODEL_DT, MODEL_TIME_UNIT, TIME_BASIS_DAYS_PER_YEAR)


def get_effective_model_dt_years() -> float:
    """Return one normal simulation step in canonical model years."""
    return get_model_dt_years() * float(SIMULATION_SPEED_MULTIPLIER)


def get_model_duration_years() -> float:
    """Return configured headless duration in canonical model years."""
    return model_time_to_years(
        MODEL_DURATION,
        MODEL_TIME_UNIT,
        TIME_BASIS_DAYS_PER_YEAR,
    )


def get_max_physics_substep_years() -> float:
    """Return the physics substep cap in canonical model years."""
    return model_time_config_value_to_years(MAX_PHYSICS_SUBSTEP_MODEL_TIME)


def get_particle_collision_cooldown_years() -> float:
    """Return the collision cooldown interval in canonical model years."""
    return model_time_config_value_to_years(PARTICLE_COLLISION_COOLDOWN_MODEL_TIME)


def model_time_config_value_to_years(value: float) -> float:
    """Convert a config-style model-time value into canonical model years."""
    return model_time_to_years(value, MODEL_TIME_UNIT, TIME_BASIS_DAYS_PER_YEAR)


def display_time_config_value_to_years(value: float) -> float:
    """Convert a display-unit value into canonical model years."""
    return model_time_to_years(value, DISPLAY_TIME_UNIT, TIME_BASIS_DAYS_PER_YEAR)


def model_years_to_display_time(value_years: float) -> float:
    """Convert canonical model years to the configured display unit."""
    return model_years_to_unit(
        value_years,
        DISPLAY_TIME_UNIT,
        TIME_BASIS_DAYS_PER_YEAR,
    )


def get_run_seed(run_index: int) -> int:
    """Return the controlled seed for one run index."""
    _validate_non_negative_integer("run_index", run_index)
    return BASE_RANDOM_SEED + run_index


def get_observation_seed(run_index: int) -> int:
    """Return the controlled observation seed for one run index."""
    _validate_non_negative_integer("run_index", run_index)
    return BASE_OBSERVATION_RANDOM_SEED + run_index


def get_condition_observation_seed(
    population_index: int,
    sigma_index: int,
    repetition_index: int,
) -> int:
    """Return a controlled observation seed for one factorial condition."""
    _validate_non_negative_integer("population_index", population_index)
    _validate_non_negative_integer("sigma_index", sigma_index)
    _validate_non_negative_integer("repetition_index", repetition_index)
    return (
        BASE_OBSERVATION_RANDOM_SEED
        + repetition_index
        + 100_000 * sigma_index
        + 10_000_000 * population_index
    )


def get_fragmentation_seed(run_index: int) -> int:
    """Return the controlled true-fragmentation seed for one run index."""
    _validate_non_negative_integer("run_index", run_index)
    return BASE_RANDOM_SEED + FRAGMENTATION_RANDOM_SEED_OFFSET + run_index


def get_particle_groups_for_population(population_value: int) -> list[dict]:
    """Return particle groups adjusted to one total population count.

    The sweep value means total satellites in the observation window. The
    configured test-satellite count is preserved when the population is large
    enough, and the remaining satellites are assigned to the active group.
    """
    _validate_positive_integer("population_value", population_value)
    test_count = min(TEST_SATELLITE_COUNT, population_value)
    active_count = population_value - test_count
    count_by_key = {
        "active": active_count,
        "test": test_count,
    }

    particle_groups = []
    for particle_group in PARTICLE_GROUPS:
        copied_group = particle_group.copy()
        copied_group["count"] = count_by_key.get(particle_group["key"], 0)
        particle_groups.append(copied_group)

    return particle_groups


def _validate_run_mode() -> None:
    """Raise a clear error when RUN_MODE is not one of the supported choices."""
    if (
        not isinstance(RUN_MODE, str)
        or RUN_MODE.strip().lower() not in SUPPORTED_RUN_MODES
    ):
        supported_modes = ", ".join(SUPPORTED_RUN_MODES)
        raise ValueError(
            f"RUN_MODE must be one of: {supported_modes}. "
            f"Current value: {RUN_MODE!r}."
        )


def _validate_headless_backend() -> None:
    """Raise a clear error when HEADLESS_BACKEND is unsupported."""
    if (
        not isinstance(HEADLESS_BACKEND, str)
        or HEADLESS_BACKEND.strip().lower() not in SUPPORTED_HEADLESS_BACKENDS
    ):
        supported_backends = ", ".join(SUPPORTED_HEADLESS_BACKENDS)
        raise ValueError(
            f"HEADLESS_BACKEND must be one of: {supported_backends}. "
            f"Current value: {HEADLESS_BACKEND!r}."
        )


def _validate_progress_mode() -> None:
    """Raise a clear error when PROGRESS_MODE is unsupported."""
    if (
        not isinstance(PROGRESS_MODE, str)
        or PROGRESS_MODE.strip().lower() not in SUPPORTED_PROGRESS_MODES
    ):
        supported_modes = ", ".join(SUPPORTED_PROGRESS_MODES)
        raise ValueError(
            f"PROGRESS_MODE must be one of: {supported_modes}. "
            f"Current value: {PROGRESS_MODE!r}."
        )


def _validate_model_time_settings() -> None:
    """Require supported model-time units and positive canonical values."""
    if MODEL_TIME_UNIT not in SUPPORTED_MODEL_TIME_UNITS:
        supported_units = ", ".join(SUPPORTED_MODEL_TIME_UNITS)
        raise ValueError(
            f"MODEL_TIME_UNIT must be one of: {supported_units}. "
            f"Current value: {MODEL_TIME_UNIT!r}."
        )
    if DISPLAY_TIME_UNIT not in SUPPORTED_DISPLAY_TIME_UNITS:
        supported_units = ", ".join(SUPPORTED_DISPLAY_TIME_UNITS)
        raise ValueError(
            f"DISPLAY_TIME_UNIT must be one of: {supported_units}. "
            f"Current value: {DISPLAY_TIME_UNIT!r}."
        )
    _validate_positive_number("MODEL_DT", MODEL_DT)
    _validate_positive_number("MODEL_DURATION", MODEL_DURATION)
    _validate_positive_number("TIME_BASIS_DAYS_PER_YEAR", TIME_BASIS_DAYS_PER_YEAR)
    _validate_positive_number(
        "MAX_PHYSICS_SUBSTEP_MODEL_TIME",
        MAX_PHYSICS_SUBSTEP_MODEL_TIME,
    )
    _validate_non_negative_number(
        "PARTICLE_COLLISION_COOLDOWN_MODEL_TIME",
        PARTICLE_COLLISION_COOLDOWN_MODEL_TIME,
    )
    _validate_positive_integer("MAX_MODEL_TIME_STEPS", MAX_MODEL_TIME_STEPS)
    _validate_boolean(
        "ALLOW_EXTREME_MODEL_TIME_WORKLOAD",
        ALLOW_EXTREME_MODEL_TIME_WORKLOAD,
    )


def _validate_positive_number(name: str, value: float) -> None:
    """Require a numeric setting to be greater than zero."""
    if not isinstance(value, (int, float)) or value <= 0:
        raise ValueError(f"{name} must be greater than 0. Current value: {value!r}.")


def _validate_non_negative_number(name: str, value: float) -> None:
    """Require a numeric setting to be zero or larger."""
    if not isinstance(value, (int, float)) or value < 0:
        raise ValueError(
            f"{name} must be greater than or equal to 0. Current value: {value!r}."
        )


def _validate_non_negative_integer(name: str, value: int) -> None:
    """Require a count to be a whole number that is zero or larger."""
    if not isinstance(value, int) or value < 0:
        raise ValueError(
            f"{name} must be a whole number greater than or equal to 0. "
            f"Current value: {value!r}."
        )


def _validate_positive_integer(name: str, value: int) -> None:
    """Require a count to be a whole number greater than zero."""
    if not isinstance(value, int) or value <= 0:
        raise ValueError(
            f"{name} must be a whole number greater than 0. "
            f"Current value: {value!r}."
        )


def _validate_boolean(name: str, value: bool) -> None:
    """Require a setting to be a real boolean flag."""
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be True or False. Current value: {value!r}.")


def _validate_choice(name: str, value: str, supported_values: list[str]) -> None:
    """Require a string setting to be one of a fixed set."""
    if not isinstance(value, str) or value not in supported_values:
        supported = ", ".join(supported_values)
        raise ValueError(f"{name} must be one of: {supported}. Current value: {value!r}.")


def _validate_population_sweep_values() -> None:
    """Require at least one positive total-population value for sweeps."""
    if not isinstance(POPULATION_SWEEP_VALUES, (list, tuple)):
        raise ValueError("POPULATION_SWEEP_VALUES must be a list or tuple.")
    if not POPULATION_SWEEP_VALUES:
        raise ValueError("POPULATION_SWEEP_VALUES must contain at least one value.")

    for population_value in POPULATION_SWEEP_VALUES:
        _validate_positive_integer("POPULATION_SWEEP_VALUES item", population_value)


def _validate_uncertainty_sigma_sweep_values() -> None:
    """Require at least one nonnegative uncertainty value for experiments."""
    if not isinstance(UNCERTAINTY_SIGMA_SWEEP_VALUES, (list, tuple)):
        raise ValueError("UNCERTAINTY_SIGMA_SWEEP_VALUES must be a list or tuple.")
    if not UNCERTAINTY_SIGMA_SWEEP_VALUES:
        raise ValueError("UNCERTAINTY_SIGMA_SWEEP_VALUES must contain at least one value.")

    for sigma_km in UNCERTAINTY_SIGMA_SWEEP_VALUES:
        _validate_non_negative_number("UNCERTAINTY_SIGMA_SWEEP_VALUES item", sigma_km)


def _validate_experiment_settings() -> None:
    """Require valid staged experiment and estimate settings."""
    _validate_choice("EXPERIMENT_STAGE", EXPERIMENT_STAGE, SUPPORTED_EXPERIMENT_STAGES)
    _validate_boolean(
        "EXPERIMENT_RESUME_WITHOUT_DUPLICATION",
        EXPERIMENT_RESUME_WITHOUT_DUPLICATION,
    )
    _validate_non_negative_number(
        "EXPERIMENT_RUNTIME_ESTIMATE_SECONDS_PER_PAIR_STEP",
        EXPERIMENT_RUNTIME_ESTIMATE_SECONDS_PER_PAIR_STEP,
    )
    _validate_non_negative_integer(
        "EXPERIMENT_OUTPUT_ESTIMATE_BYTES_PER_RUN",
        EXPERIMENT_OUTPUT_ESTIMATE_BYTES_PER_RUN,
    )
    _validate_non_negative_integer(
        "EXPERIMENT_OUTPUT_ESTIMATE_BYTES_PER_TIME_BIN",
        EXPERIMENT_OUTPUT_ESTIMATE_BYTES_PER_TIME_BIN,
    )
    _validate_positive_integer("EXPERIMENT_SURVIVAL_MIN_RUNS", EXPERIMENT_SURVIVAL_MIN_RUNS)


def _validate_benchmark_settings() -> None:
    """Require valid benchmark combinations and feasibility targets."""
    _validate_positive_integer_list(
        "BENCHMARK_POPULATION_VALUES",
        BENCHMARK_POPULATION_VALUES,
    )
    _validate_positive_integer_list(
        "BENCHMARK_MONTE_CARLO_RUN_COUNTS",
        BENCHMARK_MONTE_CARLO_RUN_COUNTS,
    )
    _validate_positive_number_list(
        "BENCHMARK_DURATION_MODEL_TIME_VALUES",
        BENCHMARK_DURATION_MODEL_TIME_VALUES,
    )
    _validate_positive_integer(
        "BENCHMARK_FEASIBILITY_TARGET_POPULATION",
        BENCHMARK_FEASIBILITY_TARGET_POPULATION,
    )
    _validate_positive_integer(
        "BENCHMARK_FEASIBILITY_TARGET_RUN_COUNT",
        BENCHMARK_FEASIBILITY_TARGET_RUN_COUNT,
    )


def _validate_profile_settings() -> None:
    """Require valid opt-in component profiler settings."""
    _validate_positive_integer_list(
        "PROFILE_POPULATION_VALUES",
        PROFILE_POPULATION_VALUES,
    )
    _validate_positive_number("PROFILE_DURATION_MODEL_TIME", PROFILE_DURATION_MODEL_TIME)
    _validate_non_negative_number(
        "PROFILE_MINIMUM_INITIAL_SPACING",
        PROFILE_MINIMUM_INITIAL_SPACING,
    )


def _validate_fragmentation_settings() -> None:
    """Require valid configurable educational cascade settings."""
    _validate_choice("FRAGMENTATION_MODE", FRAGMENTATION_MODE, SUPPORTED_FRAGMENTATION_MODES)
    _validate_positive_integer("CASCADE_ONSET_GENERATION", CASCADE_ONSET_GENERATION)
    _validate_non_negative_integer("FIXED_FRAGMENT_COUNT", FIXED_FRAGMENT_COUNT)
    _validate_non_negative_integer(
        "SIZE_DEPENDENT_FRAGMENT_BASE_COUNT",
        SIZE_DEPENDENT_FRAGMENT_BASE_COUNT,
    )
    _validate_non_negative_number(
        "SIZE_DEPENDENT_FRAGMENT_COUNT_PER_KM_RADIUS",
        SIZE_DEPENDENT_FRAGMENT_COUNT_PER_KM_RADIUS,
    )
    _validate_non_negative_number(
        "SIZE_DEPENDENT_FRAGMENT_COUNT_STD",
        SIZE_DEPENDENT_FRAGMENT_COUNT_STD,
    )
    _validate_choice(
        "FRAGMENT_RADIUS_DISTRIBUTION",
        FRAGMENT_RADIUS_DISTRIBUTION,
        SUPPORTED_FRAGMENT_RADIUS_DISTRIBUTIONS,
    )
    _validate_positive_number("FRAGMENT_RADIUS_MIN_KM", FRAGMENT_RADIUS_MIN_KM)
    _validate_positive_number("FRAGMENT_RADIUS_MAX_KM", FRAGMENT_RADIUS_MAX_KM)
    _validate_positive_number("FRAGMENT_RADIUS_CONSTANT_KM", FRAGMENT_RADIUS_CONSTANT_KM)
    if FRAGMENT_RADIUS_MIN_KM > FRAGMENT_RADIUS_MAX_KM:
        raise ValueError("FRAGMENT_RADIUS_MIN_KM must be <= FRAGMENT_RADIUS_MAX_KM.")
    _validate_choice(
        "FRAGMENT_VELOCITY_REFERENCE",
        FRAGMENT_VELOCITY_REFERENCE,
        SUPPORTED_FRAGMENT_VELOCITY_REFERENCES,
    )
    _validate_choice(
        "FRAGMENT_VELOCITY_KICK_DISTRIBUTION",
        FRAGMENT_VELOCITY_KICK_DISTRIBUTION,
        SUPPORTED_FRAGMENT_VELOCITY_KICK_DISTRIBUTIONS,
    )
    _validate_non_negative_number(
        "FRAGMENT_VELOCITY_KICK_SCALE_KM_PER_MODEL_YEAR",
        FRAGMENT_VELOCITY_KICK_SCALE_KM_PER_MODEL_YEAR,
    )
    _validate_positive_number(
        "FRAGMENT_RELEASE_OFFSET_FACTOR",
        FRAGMENT_RELEASE_OFFSET_FACTOR,
    )
    _validate_positive_integer("MAX_ACTIVE_OBJECTS", MAX_ACTIVE_OBJECTS)
    _validate_positive_integer("MAX_TRUE_COLLISION_EVENTS", MAX_TRUE_COLLISION_EVENTS)
    _validate_choice(
        "CASCADE_TERMINATION_POLICY",
        CASCADE_TERMINATION_POLICY,
        SUPPORTED_CASCADE_TERMINATION_POLICIES,
    )


def _validate_cascade_output_settings() -> None:
    """Require valid saved-table and graph sampling settings."""
    _validate_positive_number(
        "CASCADE_TIME_BIN_WIDTH_MODEL_TIME",
        CASCADE_TIME_BIN_WIDTH_MODEL_TIME,
    )
    _validate_positive_number(
        "CASCADE_ROLLING_WINDOW_MODEL_TIME",
        CASCADE_ROLLING_WINDOW_MODEL_TIME,
    )
    _validate_boolean(
        "CASCADE_PLOT_INDIVIDUAL_TRAJECTORIES",
        CASCADE_PLOT_INDIVIDUAL_TRAJECTORIES,
    )
    _validate_non_negative_integer(
        "CASCADE_MAX_INDIVIDUAL_TRAJECTORIES",
        CASCADE_MAX_INDIVIDUAL_TRAJECTORIES,
    )


def _validate_model_time_workload() -> None:
    """Refuse accidental runs with an extreme number of time steps."""
    estimated_step_count = calculate_step_count(
        get_model_duration_years(),
        get_effective_model_dt_years(),
    )
    if (
        not ALLOW_EXTREME_MODEL_TIME_WORKLOAD
        and estimated_step_count > MAX_MODEL_TIME_STEPS
    ):
        raise ValueError(
            "Configured model-time workload is too large: "
            f"{estimated_step_count:,} estimated steps exceeds "
            f"MAX_MODEL_TIME_STEPS={MAX_MODEL_TIME_STEPS:,}. "
            "Set ALLOW_EXTREME_MODEL_TIME_WORKLOAD=True only after confirming "
            "this is intentional."
        )


def _validate_positive_integer_list(name: str, values: list[int]) -> None:
    """Require a non-empty list of positive whole numbers."""
    if not isinstance(values, (list, tuple)):
        raise ValueError(f"{name} must be a list or tuple.")
    if not values:
        raise ValueError(f"{name} must contain at least one value.")

    for value in values:
        _validate_positive_integer(f"{name} item", value)


def _validate_positive_number_list(name: str, values: list[float]) -> None:
    """Require a non-empty list of positive numbers."""
    if not isinstance(values, (list, tuple)):
        raise ValueError(f"{name} must be a list or tuple.")
    if not values:
        raise ValueError(f"{name} must contain at least one value.")

    for value in values:
        _validate_positive_number(f"{name} item", value)


def _validate_speed_range(name: str, speed_range: tuple[float, float]) -> None:
    """Require a two-value speed range with positive increasing values."""
    if not isinstance(speed_range, (tuple, list)) or len(speed_range) != 2:
        raise ValueError(f"{name} must contain exactly two speeds.")

    minimum_speed, maximum_speed = speed_range
    if not isinstance(minimum_speed, (int, float)) or not isinstance(
        maximum_speed,
        (int, float),
    ):
        raise ValueError(f"{name} speeds must be numbers.")
    if minimum_speed <= 0 or maximum_speed <= 0:
        raise ValueError(f"{name} speeds must be greater than 0.")
    if minimum_speed > maximum_speed:
        raise ValueError(
            f"{name} minimum speed must be less than or equal to maximum speed."
        )


def __getattr__(name: str):
    """Return deprecated seconds-based aliases with an explicit warning."""
    aliases = {
        "TIME_STEP": ("MODEL_DT", MODEL_DT),
        "HEADLESS_DURATION_SECONDS": ("MODEL_DURATION", MODEL_DURATION),
        "MAX_PHYSICS_SUBSTEP_SECONDS": (
            "MAX_PHYSICS_SUBSTEP_MODEL_TIME",
            MAX_PHYSICS_SUBSTEP_MODEL_TIME,
        ),
        "PARTICLE_COLLISION_COOLDOWN_SECONDS": (
            "PARTICLE_COLLISION_COOLDOWN_MODEL_TIME",
            PARTICLE_COLLISION_COOLDOWN_MODEL_TIME,
        ),
        "BENCHMARK_DURATION_SECONDS_VALUES": (
            "BENCHMARK_DURATION_MODEL_TIME_VALUES",
            BENCHMARK_DURATION_MODEL_TIME_VALUES,
        ),
        "PROFILE_DURATION_SECONDS": (
            "PROFILE_DURATION_MODEL_TIME",
            PROFILE_DURATION_MODEL_TIME,
        ),
        "REPRESENTATIVE_LEO_SPEED_RANGE_KM_S": (
            "REPRESENTATIVE_LEO_SPEED_RANGE_KM_PER_PHYSICAL_SECOND",
            REPRESENTATIVE_LEO_SPEED_RANGE_KM_PER_PHYSICAL_SECOND,
        ),
    }
    if name not in aliases:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    replacement_name, value = aliases[name]
    warn_deprecated_alias(name, replacement_name)
    return value





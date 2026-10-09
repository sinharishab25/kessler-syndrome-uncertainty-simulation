"""Particle-particle collision helpers."""

from dataclasses import dataclass

import numpy as np

from simulation.particle import Particle


COLLISION_DISTANCE_TOLERANCE = 1e-9
RELATIVE_VELOCITY_SQUARED_TOLERANCE = 1e-12


@dataclass(frozen=True)
class ParticleCollision:
    """A record of one detected satellite-satellite collision."""

    particle_a_id: int
    particle_b_id: int
    position: np.ndarray
    relative_speed: float
    time_offset_model_years: float

    @property
    def time_offset_seconds(self) -> float:
        """Deprecated compatibility alias for the old ambiguous field name."""
        from simulation.time_units import warn_deprecated_alias

        warn_deprecated_alias("time_offset_seconds", "time_offset_model_years")
        return self.time_offset_model_years


@dataclass(frozen=True)
class ClosestApproachResult:
    """Closest-approach metrics for one relative state over one interval."""

    time_offset_model_years: float
    minimum_distance: float
    collision: bool

    @property
    def time_offset_seconds(self) -> float:
        """Deprecated compatibility alias for the old ambiguous field name."""
        from simulation.time_units import warn_deprecated_alias

        warn_deprecated_alias("time_offset_seconds", "time_offset_model_years")
        return self.time_offset_model_years


@dataclass(frozen=True)
class PairStepClassification:
    """True-vs-observed collision classification for one pair during one step."""

    particle_a_id: int
    particle_b_id: int
    true_time_offset_model_years: float
    observed_time_offset_model_years: float
    true_minimum_distance: float
    observed_minimum_distance: float
    true_collision: bool
    observed_collision: bool
    classification: str

    @property
    def true_time_offset_seconds(self) -> float:
        """Deprecated compatibility alias for the old ambiguous field name."""
        from simulation.time_units import warn_deprecated_alias

        warn_deprecated_alias(
            "true_time_offset_seconds",
            "true_time_offset_model_years",
        )
        return self.true_time_offset_model_years

    @property
    def observed_time_offset_seconds(self) -> float:
        """Deprecated compatibility alias for the old ambiguous field name."""
        from simulation.time_units import warn_deprecated_alias

        warn_deprecated_alias(
            "observed_time_offset_seconds",
            "observed_time_offset_model_years",
        )
        return self.observed_time_offset_model_years


def detect_particle_collisions(
    particles: list[Particle],
    dt: float,
    box_width: float,
    box_height: float,
    profiler=None,
) -> list[ParticleCollision]:
    """Return pairs whose continuous closest approach overlaps during ``dt``.

    The simulation region is periodic, so pair separation uses the minimum-image
    vector on the rectangular wraparound domain before computing closest
    approach. That is consistent with the existing modulo boundary model and
    avoids false misses when two particles are near opposite plot edges.
    """
    collisions = []
    numeric_dt = float(dt)

    if profiler is None:
        particle_count = len(particles)
        for first_index in range(particle_count):
            particle_a = particles[first_index]
            for second_index in range(first_index + 1, particle_count):
                particle_b = particles[second_index]
                closest_approach = _calculate_closest_approach(
                    particle_a,
                    particle_b,
                    numeric_dt,
                    box_width,
                    box_height,
                )
                _append_collision_if_needed(
                    collisions,
                    particle_a,
                    particle_b,
                    closest_approach,
                )
    else:
        for particle_a, particle_b in iterate_profiled_particle_pairs(
            particles,
            profiler,
        ):
            with profiler.measure("true_closest_approach_calculation"):
                closest_approach = _calculate_closest_approach(
                    particle_a,
                    particle_b,
                    numeric_dt,
                    box_width,
                    box_height,
                )
            with profiler.measure("event_deduplication_and_processing"):
                _append_collision_if_needed(
                    collisions,
                    particle_a,
                    particle_b,
                    closest_approach,
                )

    if profiler is None:
        collisions.sort(key=lambda collision: collision.time_offset_model_years)
    else:
        with profiler.measure("event_deduplication_and_processing"):
            collisions.sort(key=lambda collision: collision.time_offset_model_years)
    return collisions


def iterate_profiled_particle_pairs(
    particles: list[Particle],
    profiler,
):
    """Yield unordered particle pairs while timing pair-index selection."""
    particle_count = len(particles)
    for first_index in range(particle_count):
        for second_index in range(first_index + 1, particle_count):
            with profiler.measure("pair_index_generation"):
                particle_a = particles[first_index]
                particle_b = particles[second_index]
            yield particle_a, particle_b


def classify_particle_pair_step(
    particle_a: Particle,
    particle_b: Particle,
    observed_position_a: np.ndarray,
    observed_position_b: np.ndarray,
    dt: float,
    box_width: float,
    box_height: float,
    profiler=None,
) -> PairStepClassification:
    """Classify one unordered pair using true and observed geometry."""
    relative_velocity = particle_b.velocity - particle_a.velocity
    collision_distance = particle_a.radius + particle_b.radius

    if profiler is None:
        true_result = _classify_true_closest_approach(
            particle_a,
            particle_b,
            relative_velocity,
            dt,
            box_width,
            box_height,
            collision_distance,
        )
        observed_result = _classify_observed_closest_approach(
            observed_position_a,
            observed_position_b,
            relative_velocity,
            dt,
            box_width,
            box_height,
            collision_distance,
        )
        classification = _classify_collision_pair(
            true_result.collision,
            observed_result.collision,
        )
    else:
        with profiler.measure("true_closest_approach_calculation"):
            true_result = _classify_true_closest_approach(
                particle_a,
                particle_b,
                relative_velocity,
                dt,
                box_width,
                box_height,
                collision_distance,
            )
        with profiler.measure(
            "observed_closest_approach_and_tp_fp_fn_tn_classification"
        ):
            observed_result = _classify_observed_closest_approach(
                observed_position_a,
                observed_position_b,
                relative_velocity,
                dt,
                box_width,
                box_height,
                collision_distance,
            )
            classification = _classify_collision_pair(
                true_result.collision,
                observed_result.collision,
            )

    return PairStepClassification(
        particle_a_id=particle_a.particle_id,
        particle_b_id=particle_b.particle_id,
        true_time_offset_model_years=true_result.time_offset_model_years,
        observed_time_offset_model_years=observed_result.time_offset_model_years,
        true_minimum_distance=true_result.minimum_distance,
        observed_minimum_distance=observed_result.minimum_distance,
        true_collision=true_result.collision,
        observed_collision=observed_result.collision,
        classification=classification,
    )


def _append_collision_if_needed(
    collisions: list[ParticleCollision],
    particle_a: Particle,
    particle_b: Particle,
    closest_approach: dict[str, object],
) -> None:
    """Append a true collision candidate when closest approach overlaps."""
    collision_distance = particle_a.radius + particle_b.radius
    if (
        closest_approach["distance"]
        <= collision_distance + COLLISION_DISTANCE_TOLERANCE
    ):
        relative_velocity = particle_b.velocity - particle_a.velocity
        collisions.append(
            ParticleCollision(
                particle_a_id=particle_a.particle_id,
                particle_b_id=particle_b.particle_id,
                position=closest_approach["position"],
                relative_speed=float(np.linalg.norm(relative_velocity)),
                time_offset_model_years=closest_approach["time_offset_model_years"],
            )
        )


def _classify_true_closest_approach(
    particle_a: Particle,
    particle_b: Particle,
    relative_velocity: np.ndarray,
    dt: float,
    box_width: float,
    box_height: float,
    collision_distance: float,
) -> ClosestApproachResult:
    """Calculate true closest approach for a pair-step classification."""
    true_displacement = _minimum_image_vector(
        particle_b.position - particle_a.position,
        box_width,
        box_height,
    )
    return _calculate_closest_approach_metrics(
        true_displacement,
        relative_velocity,
        dt,
        collision_distance,
    )


def _classify_observed_closest_approach(
    observed_position_a: np.ndarray,
    observed_position_b: np.ndarray,
    relative_velocity: np.ndarray,
    dt: float,
    box_width: float,
    box_height: float,
    collision_distance: float,
) -> ClosestApproachResult:
    """Calculate observed closest approach for a pair-step classification."""
    observed_displacement = _minimum_image_vector(
        observed_position_b - observed_position_a,
        box_width,
        box_height,
    )
    return _calculate_closest_approach_metrics(
        observed_displacement,
        relative_velocity,
        dt,
        collision_distance,
    )


def _calculate_closest_approach_metrics(
    displacement: np.ndarray,
    relative_velocity: np.ndarray,
    dt: float,
    collision_distance: float,
) -> ClosestApproachResult:
    """Return closest approach using the existing zero-velocity rule."""
    relative_velocity_squared = float(np.dot(relative_velocity, relative_velocity))

    if relative_velocity_squared > RELATIVE_VELOCITY_SQUARED_TOLERANCE:
        unclamped_time = -float(np.dot(displacement, relative_velocity)) / (
            relative_velocity_squared
        )
        closest_time = float(np.clip(unclamped_time, 0.0, dt))
    else:
        closest_time = 0.0

    minimum_displacement = displacement + relative_velocity * closest_time
    minimum_distance = float(np.linalg.norm(minimum_displacement))

    return ClosestApproachResult(
        time_offset_model_years=closest_time,
        minimum_distance=minimum_distance,
        collision=(
            minimum_distance <= collision_distance + COLLISION_DISTANCE_TOLERANCE
        ),
    )


def _classify_collision_pair(true_collision: bool, observed_collision: bool) -> str:
    """Return the confusion-matrix label for one pair-step."""
    if true_collision and observed_collision:
        return "TP"
    if not true_collision and observed_collision:
        return "FP"
    if true_collision and not observed_collision:
        return "FN"

    return "TN"


def _calculate_closest_approach(
    particle_a: Particle,
    particle_b: Particle,
    dt: float,
    box_width: float,
    box_height: float,
) -> dict[str, object]:
    """Calculate the closest relative approach for one unordered pair."""
    displacement = _minimum_image_vector(
        particle_b.position - particle_a.position,
        box_width,
        box_height,
    )
    relative_velocity = particle_b.velocity - particle_a.velocity
    closest_approach = _calculate_closest_approach_metrics(
        displacement,
        relative_velocity,
        dt,
        particle_a.radius + particle_b.radius,
    )
    closest_time = closest_approach.time_offset_model_years
    minimum_displacement = displacement + relative_velocity * closest_time
    collision_position = _wrap_position(
        particle_a.position
        + particle_a.velocity * closest_time
        + 0.5 * minimum_displacement,
        box_width,
        box_height,
    )

    return {
        "distance": closest_approach.minimum_distance,
        "position": collision_position,
        "time_offset_model_years": closest_time,
    }


def _minimum_image_vector(
    displacement: np.ndarray,
    box_width: float,
    box_height: float,
) -> np.ndarray:
    """Return the nearest periodic image of a 2D displacement vector."""
    box_size = np.array([box_width, box_height], dtype=float)
    return ((displacement + 0.5 * box_size) % box_size) - 0.5 * box_size


def _wrap_position(
    position: np.ndarray,
    box_width: float,
    box_height: float,
) -> np.ndarray:
    """Wrap a 2D position into the rectangular periodic domain."""
    return np.array(
        [
            position[0] % box_width,
            position[1] % box_height,
        ],
        dtype=float,
    )

"""Simulation controller for the Version 1.5.15 LEO observation-window model."""

from pathlib import Path
from contextlib import nullcontext
import csv
import hashlib
import math

import numpy as np

import config
from simulation.collisions import (
    PairStepClassification,
    ParticleCollision,
    classify_particle_pair_step,
    detect_particle_collisions,
    iterate_profiled_particle_pairs,
)
from simulation.cascade import ObservedCascadeEvent
from simulation.particle import Particle
from simulation.physics import (
    BoundaryCrossing,
    apply_wraparound_boundary_crossings,
    move_particle,
)


class Simulation:
    """Create satellites, advance simulation time, and collect run statistics."""

    def __init__(
        self,
        random_generator: np.random.Generator,
        run_seed: int,
        run_index: int = 0,
        record_position_history: bool = True,
        particle_groups: list[dict] | None = None,
        population_value: int | None = None,
        observation_seed: int | None = None,
        position_uncertainty_sigma_km: float | None = None,
        profiler=None,
        minimum_initial_spacing: float | None = None,
        record_observation_history: bool | None = None,
    ) -> None:
        self.box_width = config.BOX_WIDTH
        self.box_height = config.BOX_HEIGHT
        self.record_position_history = record_position_history
        self.record_observation_history = (
            config.RECORD_OBSERVATION_HISTORY
            if record_observation_history is None
            else bool(record_observation_history)
        )
        self.run_seed = run_seed
        self.observation_seed = (
            observation_seed
            if observation_seed is not None
            else config.get_observation_seed(run_index)
        )
        self.run_index = run_index
        self.random_generator = random_generator
        self.observation_random_generator = np.random.default_rng(self.observation_seed)
        self.position_uncertainty_sigma_km = self._validate_position_uncertainty_sigma(
            config.POSITION_UNCERTAINTY_SIGMA_KM
            if position_uncertainty_sigma_km is None
            else position_uncertainty_sigma_km
        )
        self.particle_groups = particle_groups or config.PARTICLE_GROUPS
        self.population_value = (
            population_value
            if population_value is not None
            else sum(particle_group["count"] for particle_group in self.particle_groups)
        )
        self.base_time_step = config.get_model_dt_years()
        self.speed_multiplier = self._validate_speed_multiplier(
            config.SIMULATION_SPEED_MULTIPLIER
        )
        self.time_step = self.base_time_step * self.speed_multiplier
        self.total_frames = config.TOTAL_FRAMES
        self.paused = False
        self.profiler = profiler
        self.minimum_initial_spacing = (
            config.MINIMUM_INITIAL_SPACING
            if minimum_initial_spacing is None
            else float(minimum_initial_spacing)
        )

        self._initialize_dynamic_state()

    def _initialize_dynamic_state(self) -> None:
        """Create a fresh satellite population and clear recorded histories."""
        self.current_frame = 0
        self.current_time = 0.0
        self.boundary_crossing_count = 0
        self.boundary_crossing_records = []
        self.boundary_crossing_history = []
        self.particle_collision_count = 0
        self.particle_collision_records = []
        self.active_particle_collision_pairs = set()
        self.last_particle_collision_times = {}
        self.pair_step_classification_counts = {
            "TP": 0,
            "FP": 0,
            "FN": 0,
            "TN": 0,
        }
        self.total_pair_step_comparisons = 0
        self.pair_step_classified_step_count = 0
        self.pair_step_classification_records = []
        self.latest_pair_step_classifications = []
        self.active_observed_warning_pairs = set()
        self.observed_warning_event_count = 0
        self.observed_cascade_events = []
        self.next_observed_cascade_event_id = 0
        self.observed_cascade_onset_time_model_years = None
        self.observed_cascade_onset_generation = None
        self.observed_generation_by_object_id = {}
        self.observed_parent_event_by_object_id = {}
        self.observed_event_generation_by_id = {}
        self.pair_step_diagnostic_file = None
        self.pair_step_diagnostic_writer = None
        self.pair_step_diagnostic_path = None
        self.position_history = []
        self.observed_position_history = []
        self.latest_observed_positions_by_particle_id = {}
        self.latest_observation_errors_by_particle_id = {}
        self.trajectory_record_interval_frames = max(
            1,
            int(config.TRAJECTORY_RECORD_INTERVAL_FRAMES),
        )
        self.particles = self._create_particles(self.random_generator)
        self._initialize_observed_identity_state()
        self.initial_particle_states = self._create_initial_particle_states()
        self.initialization_fingerprint = self._create_initialization_fingerprint()

        if self.record_position_history:
            self._record_position_snapshot()
        self._record_boundary_crossing_history(crossings_this_frame=0)

    def _initialize_observed_identity_state(self) -> None:
        """Initialize observed ancestry from object identities only."""
        self.observed_generation_by_object_id = {
            particle.particle_id: 0 for particle in self.particles
        }
        self.observed_parent_event_by_object_id = {
            particle.particle_id: -1 for particle in self.particles
        }

    def reset(self) -> None:
        """Reset positions, velocities, identities, and collected run data."""
        self.paused = False
        self.close_pair_step_diagnostic_data()
        self.random_generator = np.random.default_rng(self.run_seed)
        self.observation_random_generator = np.random.default_rng(self.observation_seed)
        self._initialize_dynamic_state()

    def pause(self) -> None:
        """Pause simulation time while keeping the animation window responsive."""
        self.paused = True

    def resume(self) -> None:
        """Resume simulation time after a pause."""
        self.paused = False

    def set_speed_multiplier(self, speed_multiplier: float) -> None:
        """Update the simulation speed multiplier used by future frames."""
        self.speed_multiplier = self._validate_speed_multiplier(speed_multiplier)
        self.time_step = self.base_time_step * self.speed_multiplier

    def _validate_speed_multiplier(self, speed_multiplier: float) -> float:
        """Return a supported simulation speed multiplier."""
        numeric_multiplier = float(speed_multiplier)
        if numeric_multiplier not in config.SUPPORTED_SPEED_MULTIPLIERS:
            supported_values = ", ".join(
                str(value) for value in config.SUPPORTED_SPEED_MULTIPLIERS
            )
            raise ValueError(
                f"Unsupported speed multiplier {speed_multiplier}. "
                f"Choose one of: {supported_values}."
            )

        return numeric_multiplier

    def _validate_position_uncertainty_sigma(self, sigma_km: float) -> float:
        """Return a usable observation-noise standard deviation in kilometers."""
        numeric_sigma = float(sigma_km)
        if numeric_sigma < 0.0:
            raise ValueError("POSITION_UNCERTAINTY_SIGMA_KM must be >= 0.")

        return numeric_sigma

    def _create_particles(
        self,
        random_generator: np.random.Generator,
    ) -> list[Particle]:
        """Create grouped satellites with random positions, identities, and speeds."""
        particles = []
        particle_id = 0

        for particle_group in self.particle_groups:
            for group_index in range(particle_group["count"]):
                position = self._generate_non_overlapping_position(
                    particles,
                    random_generator,
                )
                velocity = self._generate_random_velocity_for_group(
                    particle_group,
                    random_generator,
                )

                particles.append(
                    Particle(
                        particle_id=particle_id,
                        identity=self._format_particle_identity(
                            particle_group,
                            group_index,
                        ),
                        position=position,
                        velocity=velocity,
                        radius=config.PARTICLE_RADIUS,
                        color=particle_group["color"],
                        category=particle_group["name"],
                        category_key=particle_group["key"],
                    )
                )
                particle_id += 1

        return particles

    def _format_particle_identity(self, particle_group: dict, group_index: int) -> str:
        """Return a simple sequential identity for the particle group."""
        return f"{particle_group['identity_prefix']}-{group_index + 1:03d}"

    def _generate_non_overlapping_position(
        self,
        existing_particles: list[Particle],
        random_generator: np.random.Generator,
    ) -> np.ndarray:
        """Generate a starting position that keeps large empty distances visible."""
        minimum_x = config.PARTICLE_RADIUS
        maximum_x = self.box_width - config.PARTICLE_RADIUS
        minimum_y = config.PARTICLE_RADIUS
        maximum_y = self.box_height - config.PARTICLE_RADIUS
        minimum_distance = max(
            2.0 * config.PARTICLE_RADIUS,
            self.minimum_initial_spacing,
        )

        for _attempt_number in range(config.MAX_INITIAL_POSITION_ATTEMPTS):
            candidate_position = np.array(
                [
                    random_generator.uniform(minimum_x, maximum_x),
                    random_generator.uniform(minimum_y, maximum_y),
                ],
                dtype=float,
            )

            if self._has_enough_spacing(
                candidate_position,
                existing_particles,
                minimum_distance,
            ):
                return candidate_position

        raise RuntimeError(
            "Could not place all satellites without overlap. "
            "Try reducing satellite counts or MINIMUM_INITIAL_SPACING."
        )

    def _has_enough_spacing(
        self,
        candidate_position: np.ndarray,
        existing_particles: list[Particle],
        minimum_distance: float,
    ) -> bool:
        """Return True if a candidate position is far enough from all satellites."""
        for particle in existing_particles:
            separation = np.linalg.norm(candidate_position - particle.position)
            if separation < minimum_distance:
                return False

        return True

    def _generate_random_velocity_for_group(
        self,
        particle_group: dict,
        random_generator: np.random.Generator,
    ) -> np.ndarray:
        """Create a velocity vector from a category-specific model speed range."""
        minimum_speed, maximum_speed = particle_group["speed_range_km_per_model_year"]
        speed = random_generator.uniform(minimum_speed, maximum_speed)
        angle = random_generator.uniform(0.0, 2.0 * np.pi)
        return speed * np.array([np.cos(angle), np.sin(angle)], dtype=float)

    def step(self, model_dt_years: float | None = None) -> list[BoundaryCrossing]:
        """Advance the simulation by one configured time step.

        Visual mode calls this once per animation update. Headless mode calls it
        in a tight numerical loop, so this method deliberately contains no
        plotting or UI work.
        """
        if self.paused:
            return []

        step_dt = self.time_step if model_dt_years is None else float(model_dt_years)
        if step_dt <= 0.0:
            raise ValueError("model_dt_years must be greater than 0.")

        crossings_this_frame = []
        particle_collisions_this_frame = []
        substeps = self._get_substep_count(step_dt)
        substep_dt = step_dt / substeps
        with self._measure_profile_component("uncertainty_sampling"):
            self._record_observed_position_snapshot()
        self._record_pair_step_classifications(step_dt)

        for substep_index in range(substeps):
            substep_start_time = self.current_time + substep_dt * substep_index
            particle_collisions = detect_particle_collisions(
                self.particles,
                substep_dt,
                self.box_width,
                self.box_height,
                profiler=self._get_profiler(),
            )
            with self._measure_profile_component(
                "event_deduplication_and_processing"
            ):
                new_particle_collisions = self._filter_new_particle_collisions(
                    particle_collisions,
                    substep_start_time,
                )
                particle_collisions_this_frame.extend(new_particle_collisions)

            with self._measure_profile_component(
                "physics_movement_and_wraparound"
            ):
                for particle in self.particles:
                    move_particle(particle, substep_dt)
                    boundary_crossings = apply_wraparound_boundary_crossings(
                        particle,
                        self.box_width,
                        self.box_height,
                    )
                    crossings_this_frame.extend(boundary_crossings)

        with self._measure_profile_component("event_deduplication_and_processing"):
            self.current_frame += 1
            self.current_time += step_dt
            self.boundary_crossing_count += len(crossings_this_frame)
            self.particle_collision_count += len(particle_collisions_this_frame)

            self._record_boundary_crossings(crossings_this_frame)
            self._record_particle_collisions(particle_collisions_this_frame)
        if self.record_position_history and self._should_record_position_snapshot():
            self._record_position_snapshot()
        with self._measure_profile_component("event_deduplication_and_processing"):
            self._record_boundary_crossing_history(len(crossings_this_frame))

        return crossings_this_frame

    def _filter_new_particle_collisions(
        self,
        particle_collisions: list[ParticleCollision],
        step_start_time: float,
    ) -> list[tuple[float, ParticleCollision]]:
        """Return only collision pairs that newly entered collision state."""
        overlapping_pairs = {
            self._get_particle_pair_key(collision)
            for collision in particle_collisions
        }
        separated_pairs = self.active_particle_collision_pairs - overlapping_pairs
        self.active_particle_collision_pairs -= separated_pairs

        new_collisions = []
        for collision in particle_collisions:
            pair_key = self._get_particle_pair_key(collision)
            if pair_key in self.active_particle_collision_pairs:
                continue
            self.active_particle_collision_pairs.add(pair_key)
            collision_time = step_start_time + collision.time_offset_model_years
            if self._is_particle_collision_in_cooldown(pair_key, collision_time):
                continue

            self.last_particle_collision_times[pair_key] = collision_time
            new_collisions.append((collision_time, collision))

        return new_collisions

    def _get_particle_pair_key(self, collision: ParticleCollision) -> tuple[int, int]:
        """Return a stable key for a particle collision pair."""
        return self._get_pair_key_from_ids(
            collision.particle_a_id,
            collision.particle_b_id,
        )

    def _get_pair_key_from_ids(
        self,
        particle_a_id: int,
        particle_b_id: int,
    ) -> tuple[int, int]:
        """Return a stable unordered key for two particle ids."""
        return tuple(sorted((particle_a_id, particle_b_id)))

    def _is_particle_collision_in_cooldown(
        self,
        pair_key: tuple[int, int],
        collision_time: float,
    ) -> bool:
        """Return True when a separated pair is still inside its cooldown window."""
        cooldown_model_years = max(0.0, config.get_particle_collision_cooldown_years())
        if (
            cooldown_model_years <= 0.0
            or pair_key not in self.last_particle_collision_times
        ):
            return False

        return (
            collision_time - self.last_particle_collision_times[pair_key]
            < cooldown_model_years
        )

    def _get_substep_count(self, time_step: float) -> int:
        """Return enough physics substeps to keep high multipliers stable."""
        return max(1, math.ceil(time_step / config.get_max_physics_substep_years()))

    def run_without_animation(self) -> None:
        """Run the legacy frame-count loop without displaying Matplotlib."""
        while self.current_frame < self.total_frames:
            self.step()

        self.save_boundary_crossing_data()
        self.save_particle_collision_data()
        self.save_pair_step_summary_data()

    def get_display_time(self) -> float:
        """Return elapsed simulation time converted to the configured display unit."""
        return config.model_years_to_display_time(self.current_time)

    def get_display_time_unit(self) -> str:
        """Return the configured unit used for displayed simulation time."""
        return config.DISPLAY_TIME_UNIT

    def get_average_particle_speed(self) -> float:
        """Return the average magnitude of satellite velocity vectors in km/s."""
        if not self.particles:
            return 0.0

        speeds = [particle.get_speed() for particle in self.particles]
        return float(np.mean(speeds))

    def get_particle_density(self) -> float:
        """Return number of satellites per square kilometer in the region."""
        return len(self.particles) / self.get_region_area()

    def get_region_area(self) -> float:
        """Return the represented rectangular area in square kilometers."""
        return self.box_width * self.box_height

    def get_test_particles(self) -> list[Particle]:
        """Return the editable orange test satellites."""
        return [particle for particle in self.particles if particle.is_test()]

    def reset_particle_to_initial(self, particle_id: int) -> None:
        """Restore one satellite to its original editable values."""
        particle = self._get_particle_by_id(particle_id)
        initial_state = self.initial_particle_states[particle_id]
        particle.position = initial_state["position"].copy()
        particle.velocity = initial_state["velocity"].copy()
        particle.radius = initial_state["radius"]
        self.clear_particle_collision_pair_state(particle_id)

    def clear_particle_collision_pair_state(self, particle_id: int) -> None:
        """Clear anti-spam collision state for pairs involving one edited particle."""
        self.active_particle_collision_pairs = {
            pair
            for pair in self.active_particle_collision_pairs
            if particle_id not in pair
        }
        self.last_particle_collision_times = {
            pair: collision_time
            for pair, collision_time in self.last_particle_collision_times.items()
            if particle_id not in pair
        }

    def _get_particle_by_id(self, particle_id: int) -> Particle:
        """Return the satellite with the requested particle id."""
        for particle in self.particles:
            if particle.particle_id == particle_id:
                return particle

        raise ValueError(f"No satellite found with particle id {particle_id}.")

    def _create_initial_particle_states(self) -> dict[int, dict]:
        """Store editable baseline values for each satellite."""
        return {
            particle.particle_id: {
                "position": particle.position.copy(),
                "velocity": particle.velocity.copy(),
                "radius": particle.radius,
            }
            for particle in self.particles
        }

    def _create_initialization_fingerprint(self) -> str:
        """Return a short hash of initial identities, positions, and velocities."""
        fingerprint_text_parts = []
        for particle in self.particles:
            fingerprint_text_parts.append(
                "|".join(
                    [
                        particle.identity,
                        f"{particle.position[0]:.12f}",
                        f"{particle.position[1]:.12f}",
                        f"{particle.velocity[0]:.12f}",
                        f"{particle.velocity[1]:.12f}",
                    ]
                )
            )

        fingerprint_text = "\n".join(fingerprint_text_parts)
        return hashlib.sha256(fingerprint_text.encode("utf-8")).hexdigest()[:16]

    def _get_display_time_scale(self) -> float:
        """Return one display unit expressed in canonical model years."""
        return config.display_time_config_value_to_years(1.0)

    def _record_boundary_crossings(self, crossings: list[BoundaryCrossing]) -> None:
        """Store detailed records for any boundary crossings in this frame."""
        particles_by_id = {particle.particle_id: particle for particle in self.particles}

        for crossing in crossings:
            particle = particles_by_id[crossing.particle_id]
            self.boundary_crossing_records.append(
                {
                    "frame": self.current_frame,
                    "simulation_time_model_years": self.current_time,
                    "satellite_identity": particle.identity,
                    "satellite_category": particle.category,
                    "speed_km_per_model_year": particle.get_speed(),
                    "boundary": crossing.boundary,
                    "total_boundary_crossings": self.boundary_crossing_count,
                }
            )

    def _record_particle_collisions(
        self,
        timed_collisions: list[tuple[float, ParticleCollision]],
    ) -> None:
        """Store detailed records for any particle collisions in this frame."""
        particles_by_id = {particle.particle_id: particle for particle in self.particles}

        for collision_time, collision in timed_collisions:
            particle_a = particles_by_id[collision.particle_a_id]
            particle_b = particles_by_id[collision.particle_b_id]
            self.particle_collision_records.append(
                {
                    "simulation_time_model_years": collision_time,
                    "satellite_a": particle_a.identity,
                    "satellite_b": particle_b.identity,
                    "x_position": float(collision.position[0]),
                    "y_position": float(collision.position[1]),
                    "relative_speed_km_per_model_year": collision.relative_speed,
                }
            )

    def _record_position_snapshot(self) -> None:
        """Store satellite positions for later trajectory plotting."""
        snapshot = {
            "frame": self.current_frame,
            "time": self.current_time,
            "positions": np.array([particle.position.copy() for particle in self.particles]),
            "identities": [particle.identity for particle in self.particles],
            "colors": [particle.color for particle in self.particles],
        }
        self.position_history.append(snapshot)

    def _record_pair_step_classifications(self, dt_model_years: float) -> None:
        """Count one true-vs-observed classification for each pair this step."""
        retained_classifications = []
        processed_pair_count = 0
        expected_pairs = self.get_pairs_per_step()
        profiler = self._get_profiler()

        if profiler is None:
            particle_count = len(self.particles)
            for first_index in range(particle_count):
                particle_a = self.particles[first_index]
                for second_index in range(first_index + 1, particle_count):
                    particle_b = self.particles[second_index]
                    classification = classify_particle_pair_step(
                        particle_a,
                        particle_b,
                        self.latest_observed_positions_by_particle_id[
                            particle_a.particle_id
                        ],
                        self.latest_observed_positions_by_particle_id[
                            particle_b.particle_id
                        ],
                        dt_model_years,
                        self.box_width,
                        self.box_height,
                    )
                    processed_pair_count += 1
                    self._process_pair_step_classification(
                        classification,
                        retained_classifications,
                    )
        else:
            for particle_a, particle_b in iterate_profiled_particle_pairs(
                self.particles,
                profiler,
            ):
                classification = classify_particle_pair_step(
                    particle_a,
                    particle_b,
                    self.latest_observed_positions_by_particle_id[
                        particle_a.particle_id
                    ],
                    self.latest_observed_positions_by_particle_id[
                        particle_b.particle_id
                    ],
                    dt_model_years,
                    self.box_width,
                    self.box_height,
                    profiler=profiler,
                )
                processed_pair_count += 1
                self._process_pair_step_classification(
                    classification,
                    retained_classifications,
                )

        if processed_pair_count != expected_pairs:
            raise RuntimeError(
                "Pair-step classification count mismatch: "
                f"expected {expected_pairs}, got {processed_pair_count}."
            )

        self.latest_pair_step_classifications = retained_classifications
        self.pair_step_classified_step_count += 1
        with self._measure_profile_component("event_deduplication_and_processing"):
            self._validate_pair_step_summary_counts()

    def _process_pair_step_classification(
        self,
        classification: PairStepClassification,
        retained_classifications: list[PairStepClassification],
    ) -> None:
        """Update run counts, stream diagnostics, and retain bounded examples."""
        with self._measure_profile_component("event_deduplication_and_processing"):
            self._record_pair_step_classification_count(classification)
            self._record_observed_warning_event_state(classification)
            if self._should_retain_pair_step_classification(
                classification,
                retained_classifications,
            ):
                retained_classifications.append(classification)
        self._write_pair_step_diagnostic_record_if_enabled(classification)

    def _should_retain_pair_step_classification(
        self,
        classification: PairStepClassification,
        retained_classifications: list[PairStepClassification],
    ) -> bool:
        """Keep bounded non-TN examples without storing every pair-step."""
        if classification.classification == "TN":
            return False

        return (
            len(retained_classifications)
            < config.MAX_RETAINED_PAIR_STEP_CLASSIFICATIONS
        )

    def _record_pair_step_classification_count(
        self,
        classification: PairStepClassification,
    ) -> None:
        """Increment the run-level confusion-matrix count for one pair-step."""
        if classification.classification not in self.pair_step_classification_counts:
            raise RuntimeError(
                f"Unknown pair-step classification: {classification.classification!r}."
            )

        self.pair_step_classification_counts[classification.classification] += 1
        self.total_pair_step_comparisons += 1

    def _record_observed_warning_event_state(
        self,
        classification: PairStepClassification,
    ) -> None:
        """Deduplicate prolonged observed collision warnings across steps."""
        pair_key = self._get_pair_key_from_ids(
            classification.particle_a_id,
            classification.particle_b_id,
        )
        if classification.observed_collision:
            if pair_key not in self.active_observed_warning_pairs:
                self.active_observed_warning_pairs.add(pair_key)
                self.observed_warning_event_count += 1
                self._record_observed_cascade_event(classification)
            return

        self.active_observed_warning_pairs.discard(pair_key)

    def _record_observed_cascade_event(
        self,
        classification: PairStepClassification,
    ) -> None:
        """Record one deduplicated observed event using observed identity ancestry."""
        self._ensure_observed_cascade_state()
        parent_i = min(classification.particle_a_id, classification.particle_b_id)
        parent_j = max(classification.particle_a_id, classification.particle_b_id)
        self._ensure_observed_object_id(parent_i)
        self._ensure_observed_object_id(parent_j)
        event_id = self.next_observed_cascade_event_id
        self.next_observed_cascade_event_id += 1
        generation = 1 + max(
            self.observed_generation_by_object_id[parent_i],
            self.observed_generation_by_object_id[parent_j],
        )
        event = ObservedCascadeEvent(
            event_id=event_id,
            model_time_years=(
                self.current_time + classification.observed_time_offset_model_years
            ),
            parent_i=parent_i,
            parent_j=parent_j,
            parent_event_i=self.observed_parent_event_by_object_id[parent_i],
            parent_event_j=self.observed_parent_event_by_object_id[parent_j],
            generation=generation,
        )
        self.observed_cascade_events.append(event)
        self.observed_event_generation_by_id[event_id] = generation
        self.observed_generation_by_object_id[parent_i] = generation
        self.observed_generation_by_object_id[parent_j] = generation
        self.observed_parent_event_by_object_id[parent_i] = event_id
        self.observed_parent_event_by_object_id[parent_j] = event_id
        self._record_observed_cascade_onset(event)

    def _ensure_observed_object_id(self, object_id: int) -> None:
        """Initialize an observed identity if it has not appeared before."""
        if object_id not in self.observed_generation_by_object_id:
            self.observed_generation_by_object_id[object_id] = 0
            self.observed_parent_event_by_object_id[object_id] = -1

    def _record_observed_cascade_onset(self, event: ObservedCascadeEvent) -> None:
        """Store first observed cascade onset, preserving None when absent."""
        if self.observed_cascade_onset_time_model_years is not None:
            return
        if event.generation >= config.CASCADE_ONSET_GENERATION:
            self.observed_cascade_onset_time_model_years = event.model_time_years
            self.observed_cascade_onset_generation = event.generation

    def _ensure_observed_cascade_state(self) -> None:
        """Backfill observed-cascade state for manually constructed test fixtures."""
        if not hasattr(self, "observed_cascade_events"):
            self.observed_cascade_events = []
        if not hasattr(self, "next_observed_cascade_event_id"):
            self.next_observed_cascade_event_id = len(self.observed_cascade_events)
        if not hasattr(self, "observed_cascade_onset_time_model_years"):
            self.observed_cascade_onset_time_model_years = None
        if not hasattr(self, "observed_cascade_onset_generation"):
            self.observed_cascade_onset_generation = None
        if not hasattr(self, "observed_generation_by_object_id"):
            self.observed_generation_by_object_id = {}
        if not hasattr(self, "observed_parent_event_by_object_id"):
            self.observed_parent_event_by_object_id = {}
        if not hasattr(self, "observed_event_generation_by_id"):
            self.observed_event_generation_by_id = {}

        for particle in getattr(self, "particles", []):
            self.observed_generation_by_object_id.setdefault(particle.particle_id, 0)
            self.observed_parent_event_by_object_id.setdefault(
                particle.particle_id,
                -1,
            )

    def _write_pair_step_diagnostic_record_if_enabled(
        self,
        classification: PairStepClassification,
    ) -> None:
        """Stream optional pair-step diagnostics without retaining all rows."""
        if not config.SAVE_PAIR_STEP_DIAGNOSTIC_CSV:
            return

        with self._measure_profile_component("csv_writing"):
            writer = self._get_pair_step_diagnostic_writer()
            writer.writerow(self._format_pair_step_classification_record(classification))

    def _get_pair_step_diagnostic_writer(self) -> csv.DictWriter:
        """Return a lazily opened diagnostic CSV writer."""
        if self.pair_step_diagnostic_writer is not None:
            return self.pair_step_diagnostic_writer

        output_directory = Path(config.OUTPUT_DIRECTORY)
        output_directory.mkdir(parents=True, exist_ok=True)
        self.pair_step_diagnostic_path = (
            output_directory / config.PAIR_STEP_DIAGNOSTIC_FILENAME
        )
        self.pair_step_diagnostic_file = self.pair_step_diagnostic_path.open(
            "w",
            newline="",
            encoding="utf-8",
        )
        fieldnames = list(
            self._format_pair_step_classification_record(
                self._create_empty_pair_step_classification(),
            ).keys()
        )
        self.pair_step_diagnostic_writer = csv.DictWriter(
            self.pair_step_diagnostic_file,
            fieldnames=fieldnames,
        )
        self.pair_step_diagnostic_writer.writeheader()
        return self.pair_step_diagnostic_writer

    def _create_empty_pair_step_classification(self) -> PairStepClassification:
        """Create a placeholder used only to derive diagnostic CSV field order."""
        return PairStepClassification(
            particle_a_id=0,
            particle_b_id=0,
            true_time_offset_model_years=0.0,
            observed_time_offset_model_years=0.0,
            true_minimum_distance=0.0,
            observed_minimum_distance=0.0,
            true_collision=False,
            observed_collision=False,
            classification="TN",
        )

    def _format_pair_step_classification_record(
        self,
        classification: PairStepClassification,
    ) -> dict[str, object]:
        """Convert one pair-step classification to an internal record."""
        return {
            "frame": self.current_frame,
            "simulation_time_model_years": self.current_time,
            "particle_a_id": classification.particle_a_id,
            "particle_b_id": classification.particle_b_id,
            "true_t_ca_model_years": classification.true_time_offset_model_years,
            "observed_t_ca_model_years": classification.observed_time_offset_model_years,
            "true_d_min_km": classification.true_minimum_distance,
            "observed_d_min_km": classification.observed_minimum_distance,
            "true_collision": classification.true_collision,
            "observed_collision": classification.observed_collision,
            "classification": classification.classification,
        }

    def get_pairs_per_step(self) -> int:
        """Return the number of eligible unordered particle pairs in one step."""
        particle_count = len(self.particles)
        return particle_count * (particle_count - 1) // 2

    def get_pair_step_summary(self) -> dict[str, object]:
        """Return one lightweight run-level classification summary."""
        self._ensure_observed_cascade_state()
        self._validate_pair_step_summary_counts()
        counts = self.pair_step_classification_counts.copy()

        return {
            "run_id": self.run_index,
            "physical_seed": self.run_seed,
            "observation_seed": self.observation_seed,
            "population": self.population_value,
            "sigma_km": self.position_uncertainty_sigma_km,
            "duration_model_years": self.current_time,
            "dt_model_years": self.time_step,
            "pairs_per_step": self.get_pairs_per_step(),
            "total_pair_step_comparisons": self.total_pair_step_comparisons,
            "TP": counts["TP"],
            "FP": counts["FP"],
            "FN": counts["FN"],
            "TN": counts["TN"],
            "true_positive_rate": self._safe_rate(
                counts["TP"],
                counts["TP"] + counts["FN"],
            ),
            "false_negative_rate": self._safe_rate(
                counts["FN"],
                counts["TP"] + counts["FN"],
            ),
            "false_positive_rate": self._safe_rate(
                counts["FP"],
                counts["FP"] + counts["TN"],
            ),
            "precision": self._safe_rate(
                counts["TP"],
                counts["TP"] + counts["FP"],
            ),
            "true_deduplicated_collision_event_count": self.particle_collision_count,
            "observed_deduplicated_warning_event_count": (
                self.observed_warning_event_count
            ),
            "true_collision_event_count": 0,
            "true_cascade_onset_time_model_years": None,
            "true_cascade_onset_generation": None,
            "cascade_termination_reason": "",
            "observed_cascade_event_count": len(self.observed_cascade_events),
            "observed_cascade_onset_time_model_years": (
                self.observed_cascade_onset_time_model_years
            ),
            "observed_cascade_onset_generation": (
                self.observed_cascade_onset_generation
            ),
        }

    def save_pair_step_summary_data(self) -> Path:
        """Write one lightweight pair-step classification summary row."""
        output_directory = Path(config.OUTPUT_DIRECTORY)
        output_directory.mkdir(parents=True, exist_ok=True)
        output_path = output_directory / config.PAIR_STEP_SUMMARY_FILENAME
        summary = self.get_pair_step_summary()
        fieldnames = [
            "run_id",
            "physical_seed",
            "observation_seed",
            "population",
            "sigma_km",
            "duration_model_years",
            "dt_model_years",
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
            "true_deduplicated_collision_event_count",
            "observed_deduplicated_warning_event_count",
            "true_collision_event_count",
            "true_cascade_onset_time_model_years",
            "true_cascade_onset_generation",
            "cascade_termination_reason",
            "observed_cascade_event_count",
            "observed_cascade_onset_time_model_years",
            "observed_cascade_onset_generation",
        ]

        with output_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(
                {
                    key: self._format_optional_csv_value(summary[key])
                    for key in fieldnames
                }
            )

        self.close_pair_step_diagnostic_data()
        return output_path

    def close_pair_step_diagnostic_data(self) -> None:
        """Close the optional streamed diagnostic CSV if it was opened."""
        if self.pair_step_diagnostic_file is not None:
            self.pair_step_diagnostic_file.close()
            self.pair_step_diagnostic_file = None
            self.pair_step_diagnostic_writer = None

    def _validate_pair_step_summary_counts(self) -> None:
        """Validate the run-level pair-step count identities."""
        counts = self.pair_step_classification_counts
        if any(count < 0 for count in counts.values()):
            raise RuntimeError("Pair-step classification counts cannot be negative.")

        counted_total = sum(counts.values())
        if counted_total != self.total_pair_step_comparisons:
            raise RuntimeError(
                "Pair-step classification total mismatch: "
                f"counts sum to {counted_total}, tracked total is "
                f"{self.total_pair_step_comparisons}."
            )

        expected_total = self.get_pairs_per_step() * self.pair_step_classified_step_count
        if self.total_pair_step_comparisons != expected_total:
            raise RuntimeError(
                "Pair-step comparison total mismatch: "
                f"expected {expected_total}, got {self.total_pair_step_comparisons}."
            )

    def _safe_rate(self, numerator: int, denominator: int) -> float | None:
        """Return a rate, or None when the requested denominator is zero."""
        if denominator == 0:
            return None

        return numerator / denominator

    def _format_optional_csv_value(self, value: object) -> object:
        """Write missing rate values as blank CSV cells."""
        if value is None:
            return ""

        return value

    def _record_observed_position_snapshot(self) -> None:
        """Store noisy observation positions without changing true particle state."""
        true_positions = np.array(
            [particle.position.copy() for particle in self.particles],
            dtype=float,
        )
        observation_errors = self.observation_random_generator.normal(
            loc=0.0,
            scale=self.position_uncertainty_sigma_km,
            size=true_positions.shape,
        )
        observed_positions = true_positions + observation_errors
        particle_ids = [particle.particle_id for particle in self.particles]

        self.latest_observed_positions_by_particle_id = {
            particle_id: observed_positions[index].copy()
            for index, particle_id in enumerate(particle_ids)
        }
        self.latest_observation_errors_by_particle_id = {
            particle_id: observation_errors[index].copy()
            for index, particle_id in enumerate(particle_ids)
        }
        if self.record_observation_history:
            self.observed_position_history.append(
                {
                    "frame": self.current_frame,
                    "time": self.current_time,
                    "true_positions": true_positions,
                    "observed_positions": observed_positions,
                    "observation_errors": observation_errors,
                    "particle_ids": particle_ids,
                    "identities": [particle.identity for particle in self.particles],
                }
            )

    def _should_record_position_snapshot(self) -> bool:
        """Return True when this frame should be kept for trajectory output."""
        return self.current_frame % self.trajectory_record_interval_frames == 0

    def _record_boundary_crossing_history(self, crossings_this_frame: int) -> None:
        """Store per-frame boundary crossing totals for graphing."""
        self.boundary_crossing_history.append(
            {
                "frame": self.current_frame,
                "time": self.current_time,
                "crossings_this_frame": crossings_this_frame,
                "total_boundary_crossings": self.boundary_crossing_count,
            }
        )

    def save_boundary_crossing_data(self) -> Path:
        """Write boundary crossing records to a CSV file."""
        output_directory = Path(config.OUTPUT_DIRECTORY)
        output_directory.mkdir(parents=True, exist_ok=True)
        output_path = output_directory / config.BOUNDARY_CROSSING_DATA_FILENAME

        fieldnames = [
            "frame",
            "simulation_time_model_years",
            "satellite_identity",
            "satellite_category",
            "speed_km_per_model_year",
            "boundary",
            "total_boundary_crossings",
        ]

        with output_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.boundary_crossing_records)

        return output_path

    def _get_profiler(self):
        """Return the optional component profiler for this simulation."""
        return getattr(self, "profiler", None)

    def _measure_profile_component(self, component: str):
        """Return a no-op context when profiling is disabled."""
        profiler = self._get_profiler()
        if profiler is None:
            return nullcontext()

        return profiler.measure(component)

    def save_particle_collision_data(self) -> Path:
        """Write particle collision records to a CSV file."""
        output_directory = Path(config.OUTPUT_DIRECTORY)
        output_directory.mkdir(parents=True, exist_ok=True)
        output_path = output_directory / config.PARTICLE_COLLISION_DATA_FILENAME

        fieldnames = [
            "simulation_time_model_years",
            "satellite_a",
            "satellite_b",
            "x_position",
            "y_position",
            "relative_speed_km_per_model_year",
        ]

        with output_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.particle_collision_records)

        return output_path








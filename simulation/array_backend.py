"""NumPy array backend for Version 1.5.8 headless simulations."""

from pathlib import Path
import csv
import hashlib
import math

import numpy as np

import config
from simulation.cascade import ObservedCascadeEvent, TrueCollisionEvent
from simulation.collisions import (
    COLLISION_DISTANCE_TOLERANCE,
    RELATIVE_VELOCITY_SQUARED_TOLERANCE,
    PairStepClassification,
    ParticleCollision,
)
from simulation.particle import Particle
from simulation.physics import BoundaryCrossing


class ArraySimulation:
    """Headless simulation backend storing particle state in NumPy arrays."""

    def __init__(
        self,
        random_generator: np.random.Generator,
        run_seed: int,
        run_index: int = 0,
        record_position_history: bool = False,
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
        self.record_position_history = bool(record_position_history)
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
        self.fragmentation_seed = config.get_fragmentation_seed(run_index)
        self.fragmentation_random_generator = np.random.default_rng(
            self.fragmentation_seed
        )
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
        self.pair_chunk_size = int(config.PAIR_CHUNK_SIZE)
        self.minimum_initial_spacing = (
            config.MINIMUM_INITIAL_SPACING
            if minimum_initial_spacing is None
            else float(minimum_initial_spacing)
        )

        self._initialize_dynamic_state()

    @classmethod
    def from_particles(
        cls,
        particles: list[Particle],
        time_step: float,
        observation_seed: int = 0,
        position_uncertainty_sigma_km: float = 0.0,
        box_width: float = 100.0,
        box_height: float = 100.0,
    ) -> "ArraySimulation":
        """Create a deterministic manual array simulation for tests."""
        simulation = cls.__new__(cls)
        simulation.box_width = box_width
        simulation.box_height = box_height
        simulation.record_position_history = False
        simulation.record_observation_history = True
        simulation.run_seed = 0
        simulation.observation_seed = observation_seed
        simulation.run_index = 0
        simulation.random_generator = np.random.default_rng(0)
        simulation.observation_random_generator = np.random.default_rng(observation_seed)
        simulation.fragmentation_seed = config.get_fragmentation_seed(0)
        simulation.fragmentation_random_generator = np.random.default_rng(
            simulation.fragmentation_seed
        )
        simulation.position_uncertainty_sigma_km = position_uncertainty_sigma_km
        simulation.population_value = len(particles)
        simulation.base_time_step = time_step
        simulation.speed_multiplier = 1.0
        simulation.time_step = time_step
        simulation.total_frames = 0
        simulation.paused = False
        simulation.profiler = None
        simulation.pair_chunk_size = int(config.PAIR_CHUNK_SIZE)
        simulation.minimum_initial_spacing = 0.0
        simulation._initialize_empty_records()
        simulation._load_particle_arrays(particles)
        simulation.initial_particle_states = {}
        simulation.initialization_fingerprint = (
            simulation._create_initialization_fingerprint()
        )
        return simulation

    def _initialize_dynamic_state(self) -> None:
        """Create array-backed particle state and clear collected records."""
        self._initialize_empty_records()
        particles = self._create_initial_particles(self.random_generator)
        self._load_particle_arrays(particles)
        self.initial_particle_states = {}
        self.initialization_fingerprint = self._create_initialization_fingerprint()

        if self.record_position_history:
            self._record_position_snapshot()
        self._record_boundary_crossing_history(crossings_this_frame=0)

    def _initialize_empty_records(self) -> None:
        """Clear run records while leaving particle arrays to be loaded later."""
        self.current_frame = 0
        self.current_time = 0.0
        self.boundary_crossing_count = 0
        self.boundary_crossing_records = []
        self.boundary_crossing_history = []
        self.particle_collision_count = 0
        self.particle_collision_records = []
        self.true_collision_events = []
        self.next_true_collision_event_id = 0
        self.true_cascade_onset_time_model_years = None
        self.true_cascade_onset_generation = None
        self.cascade_terminated = False
        self.cascade_termination_reason = ""
        self.fragmentation_disabled_by_policy = False
        self.active_particle_collision_pairs = set()
        self.last_particle_collision_times = {}
        self.pair_step_classification_counts = {
            "TP": 0,
            "FP": 0,
            "FN": 0,
            "TN": 0,
        }
        self.total_pair_step_comparisons = 0
        self.expected_pair_step_comparisons = 0
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
        self.pending_observed_creator_event_id = None
        self.pending_observed_creator_frame = None
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

    def _create_initial_particles(
        self,
        random_generator: np.random.Generator,
    ) -> list[Particle]:
        """Create the same ordered initial particles as the object backend."""
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

    def _load_particle_arrays(self, particles: list[Particle]) -> None:
        """Load stable particle metadata and numeric state into arrays."""
        self.object_ids = np.array([particle.particle_id for particle in particles], dtype=int)
        self.positions = np.array([particle.position.copy() for particle in particles], dtype=float)
        self.velocities = np.array([particle.velocity.copy() for particle in particles], dtype=float)
        self.radii = np.array([particle.radius for particle in particles], dtype=float)
        self.active_mask = np.ones(len(particles), dtype=bool)
        self.object_types = np.array([particle.category_key for particle in particles])
        self.generations = np.zeros(len(particles), dtype=int)
        self.parent_event_ids = np.full(len(particles), -1, dtype=int)
        self.identities = np.array([particle.identity for particle in particles])
        self.categories = np.array([particle.category for particle in particles])
        self.colors = np.array([particle.color for particle in particles])
        self.next_object_id = (
            int(self.object_ids.max()) + 1 if len(self.object_ids) > 0 else 0
        )
        self.latest_observed_positions = np.empty_like(self.positions)
        self.latest_observation_errors = np.zeros_like(self.positions)
        self._initialize_observed_identity_state()

    def _initialize_observed_identity_state(self) -> None:
        """Initialize observed ancestry from visible object identities only."""
        self.observed_generation_by_object_id = {
            int(object_id): 0 for object_id in self.object_ids
        }
        self.observed_parent_event_by_object_id = {
            int(object_id): -1 for object_id in self.object_ids
        }

    def _format_particle_identity(self, particle_group: dict, group_index: int) -> str:
        """Return a simple sequential identity for one group."""
        return f"{particle_group['identity_prefix']}-{group_index + 1:03d}"

    def _generate_non_overlapping_position(
        self,
        existing_particles: list[Particle],
        random_generator: np.random.Generator,
    ) -> np.ndarray:
        """Generate positions with the same draw order as the object backend."""
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
        """Return True when a candidate matches object-backend spacing rules."""
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
        """Create a velocity vector with the same draw order as Simulation."""
        minimum_speed, maximum_speed = particle_group["speed_range_km_per_model_year"]
        speed = random_generator.uniform(minimum_speed, maximum_speed)
        angle = random_generator.uniform(0.0, 2.0 * np.pi)
        return speed * np.array([np.cos(angle), np.sin(angle)], dtype=float)

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

    def step(self, model_dt_years: float | None = None) -> list[BoundaryCrossing]:
        """Advance active rows one step with vectorized physics."""
        if self.paused or self.cascade_terminated:
            return []

        step_dt = self.time_step if model_dt_years is None else float(model_dt_years)
        if step_dt <= 0.0:
            raise ValueError("model_dt_years must be greater than 0.")

        crossings_this_frame = []
        particle_collisions_this_frame = []
        substeps = self._get_substep_count(step_dt)
        substep_dt = step_dt / substeps
        self._record_observed_position_snapshot()
        self._record_pair_step_classifications(step_dt)

        for substep_index in range(substeps):
            substep_start_time = self.current_time + substep_dt * substep_index
            particle_collisions = self._detect_particle_collisions(substep_dt)
            if self._is_true_cascade_mode_enabled():
                new_particle_collisions = self._accept_true_cascade_collisions(
                    particle_collisions,
                    substep_start_time,
                )
            else:
                new_particle_collisions = self._filter_new_particle_collisions(
                    particle_collisions,
                    substep_start_time,
                )
            particle_collisions_this_frame.extend(new_particle_collisions)
            crossings_this_frame.extend(self._move_active_rows_and_wrap(substep_dt))

        self.current_frame += 1
        self.current_time += step_dt
        self.boundary_crossing_count += len(crossings_this_frame)
        self.particle_collision_count += len(particle_collisions_this_frame)
        self._record_boundary_crossings(crossings_this_frame)
        self._record_particle_collisions(particle_collisions_this_frame)
        if self.record_position_history and self._should_record_position_snapshot():
            self._record_position_snapshot()
        self._record_boundary_crossing_history(len(crossings_this_frame))

        return crossings_this_frame

    def _detect_particle_collisions(self, dt: float) -> list[ParticleCollision]:
        """Chunked/vectorized continuous detection over active rows only."""
        active_rows = self._get_active_rows()
        if len(active_rows) < 2:
            return []

        collisions = []
        for pair_a, pair_b in self._iter_active_pair_row_chunks(active_rows):
            true_result = self._calculate_pair_closest_approach(
                self.positions[pair_b] - self.positions[pair_a],
                self.velocities[pair_b] - self.velocities[pair_a],
                dt,
            )
            collision_distances = self.radii[pair_a] + self.radii[pair_b]
            collision_mask = (
                true_result["minimum_distances"]
                <= collision_distances + COLLISION_DISTANCE_TOLERANCE
            )
            if not np.any(collision_mask):
                continue

            collision_pair_a = pair_a[collision_mask]
            collision_pair_b = pair_b[collision_mask]
            collision_times = true_result["times"][collision_mask]
            collision_displacements = true_result["minimum_displacements"][
                collision_mask
            ]
            collision_positions = self._wrap_positions(
                self.positions[collision_pair_a]
                + self.velocities[collision_pair_a] * collision_times[:, np.newaxis]
                + 0.5 * collision_displacements
            )
            relative_velocities = (
                self.velocities[collision_pair_b]
                - self.velocities[collision_pair_a]
            )
            relative_speeds = np.linalg.norm(relative_velocities, axis=1)

            for index in range(len(collision_pair_a)):
                collisions.append(
                    ParticleCollision(
                        particle_a_id=int(self.object_ids[collision_pair_a[index]]),
                        particle_b_id=int(self.object_ids[collision_pair_b[index]]),
                        position=collision_positions[index].copy(),
                        relative_speed=float(relative_speeds[index]),
                        time_offset_model_years=float(collision_times[index]),
                    )
                )

        collisions.sort(
            key=lambda collision: (
                collision.time_offset_model_years,
                min(collision.particle_a_id, collision.particle_b_id),
                max(collision.particle_a_id, collision.particle_b_id),
            )
        )
        return collisions

    def _record_pair_step_classifications(self, dt_model_years: float) -> None:
        """Vectorize pair-step classification over active rows only."""
        active_rows = self._get_active_rows()
        retained_classifications = []
        if len(active_rows) < 2:
            self.expected_pair_step_comparisons += 0
            self.latest_pair_step_classifications = retained_classifications
            self.pair_step_classified_step_count += 1
            self._validate_pair_step_summary_counts()
            return

        observed_warning_pairs = set()
        observed_warning_candidates = []

        for pair_a, pair_b in self._iter_active_pair_row_chunks(active_rows):
            relative_velocities = self.velocities[pair_b] - self.velocities[pair_a]
            collision_distances = self.radii[pair_a] + self.radii[pair_b]
            true_result = self._calculate_pair_closest_approach(
                self.positions[pair_b] - self.positions[pair_a],
                relative_velocities,
                dt_model_years,
            )
            observed_result = self._calculate_pair_closest_approach(
                self.latest_observed_positions[pair_b]
                - self.latest_observed_positions[pair_a],
                relative_velocities,
                dt_model_years,
            )
            true_collision = (
                true_result["minimum_distances"]
                <= collision_distances + COLLISION_DISTANCE_TOLERANCE
            )
            observed_collision = (
                observed_result["minimum_distances"]
                <= collision_distances + COLLISION_DISTANCE_TOLERANCE
            )

            classifications = np.full(len(pair_a), "TN", dtype="<U2")
            classifications[true_collision & observed_collision] = "TP"
            classifications[~true_collision & observed_collision] = "FP"
            classifications[true_collision & ~observed_collision] = "FN"

            for label in ("TP", "FP", "FN", "TN"):
                count = int(np.count_nonzero(classifications == label))
                self.pair_step_classification_counts[label] += count
            self.total_pair_step_comparisons += len(pair_a)
            self.expected_pair_step_comparisons += len(pair_a)
            observed_warning_pairs.update(
                self._get_observed_warning_pairs(pair_a, pair_b, observed_collision)
            )
            observed_warning_candidates.extend(
                self._get_observed_warning_candidates(
                    pair_a,
                    pair_b,
                    observed_result,
                    observed_collision,
                )
            )

            self._materialize_pair_step_classifications_if_needed(
                pair_a,
                pair_b,
                true_result,
                observed_result,
                true_collision,
                observed_collision,
                classifications,
                retained_classifications,
            )

        self._update_observed_warning_event_state(
            observed_warning_pairs,
            observed_warning_candidates,
        )

        self.latest_pair_step_classifications = retained_classifications
        self.pair_step_classified_step_count += 1
        self._validate_pair_step_summary_counts()

    def _create_pair_step_classification(
        self,
        row_a: int,
        row_b: int,
        true_result: dict[str, np.ndarray],
        observed_result: dict[str, np.ndarray],
        true_collision: bool,
        observed_collision: bool,
        classification: str,
        pair_index: int,
    ) -> PairStepClassification:
        """Create a row object for retained examples or streamed diagnostics."""
        return PairStepClassification(
            particle_a_id=int(self.object_ids[row_a]),
            particle_b_id=int(self.object_ids[row_b]),
            true_time_offset_model_years=float(true_result["times"][pair_index]),
            observed_time_offset_model_years=float(observed_result["times"][pair_index]),
            true_minimum_distance=float(true_result["minimum_distances"][pair_index]),
            observed_minimum_distance=float(
                observed_result["minimum_distances"][pair_index]
            ),
            true_collision=true_collision,
            observed_collision=observed_collision,
            classification=classification,
        )

    def _calculate_pair_closest_approach(
        self,
        displacements: np.ndarray,
        relative_velocities: np.ndarray,
        dt: float,
    ) -> dict[str, np.ndarray]:
        """Vectorized closest-approach metrics using the current formulas."""
        minimum_image_displacements = self._minimum_image_vectors(displacements)
        relative_velocity_squared = np.einsum(
            "ij,ij->i",
            relative_velocities,
            relative_velocities,
        )
        times = np.zeros(len(minimum_image_displacements), dtype=float)
        moving_mask = relative_velocity_squared > RELATIVE_VELOCITY_SQUARED_TOLERANCE
        if np.any(moving_mask):
            unclamped_times = -(
                np.einsum(
                    "ij,ij->i",
                    minimum_image_displacements[moving_mask],
                    relative_velocities[moving_mask],
                )
                / relative_velocity_squared[moving_mask]
            )
            times[moving_mask] = np.clip(unclamped_times, 0.0, dt)

        minimum_displacements = (
            minimum_image_displacements + relative_velocities * times[:, np.newaxis]
        )
        minimum_distances = np.linalg.norm(minimum_displacements, axis=1)
        return {
            "times": times,
            "minimum_displacements": minimum_displacements,
            "minimum_distances": minimum_distances,
        }

    def _move_active_rows_and_wrap(self, dt: float) -> list[BoundaryCrossing]:
        """Vectorize active-row movement and wraparound boundary detection."""
        active_rows = self._get_active_rows()
        self.positions[active_rows] += self.velocities[active_rows] * dt

        x_positions = self.positions[active_rows, 0]
        y_positions = self.positions[active_rows, 1]
        left_mask = x_positions < 0.0
        right_mask = x_positions > self.box_width
        bottom_mask = y_positions < 0.0
        top_mask = y_positions > self.box_height
        crossings = []

        for local_index, row in enumerate(active_rows):
            if left_mask[local_index]:
                crossings.append(BoundaryCrossing(int(self.object_ids[row]), "left"))
            elif right_mask[local_index]:
                crossings.append(BoundaryCrossing(int(self.object_ids[row]), "right"))

            if bottom_mask[local_index]:
                crossings.append(BoundaryCrossing(int(self.object_ids[row]), "bottom"))
            elif top_mask[local_index]:
                crossings.append(BoundaryCrossing(int(self.object_ids[row]), "top"))

        x_crossed = left_mask | right_mask
        y_crossed = bottom_mask | top_mask
        self.positions[active_rows[x_crossed], 0] %= self.box_width
        self.positions[active_rows[y_crossed], 1] %= self.box_height
        return crossings

    def _record_observed_position_snapshot(self) -> None:
        """Sample observation errors for active rows only in stable row order."""
        active_rows = self._get_active_rows()
        self._register_newly_observed_object_ids(active_rows)
        observation_errors = self.observation_random_generator.normal(
            loc=0.0,
            scale=self.position_uncertainty_sigma_km,
            size=(len(active_rows), 2),
        )
        self.latest_observation_errors[:] = 0.0
        self.latest_observed_positions[:] = self.positions
        self.latest_observation_errors[active_rows] = observation_errors
        self.latest_observed_positions[active_rows] = (
            self.positions[active_rows] + observation_errors
        )
        self.latest_observed_positions_by_particle_id = {
            int(self.object_ids[row]): self.latest_observed_positions[row].copy()
            for row in active_rows
        }
        self.latest_observation_errors_by_particle_id = {
            int(self.object_ids[row]): self.latest_observation_errors[row].copy()
            for row in active_rows
        }
        if self.record_observation_history:
            self.observed_position_history.append(
                {
                    "frame": self.current_frame,
                    "time": self.current_time,
                    "true_positions": self.positions[active_rows].copy(),
                    "observed_positions": (
                        self.latest_observed_positions[active_rows].copy()
                    ),
                    "observation_errors": observation_errors.copy(),
                    "particle_ids": [int(self.object_ids[row]) for row in active_rows],
                    "identities": [str(self.identities[row]) for row in active_rows],
                }
            )

    def _register_newly_observed_object_ids(self, active_rows: np.ndarray) -> None:
        """Assign observed ancestry to newly visible IDs without true ancestry."""
        inherited_event_id = None
        if (
            self.pending_observed_creator_event_id is not None
            and self.current_frame == self.pending_observed_creator_frame + 1
        ):
            inherited_event_id = self.pending_observed_creator_event_id

        registered_new_id = False
        for row in active_rows:
            object_id = int(self.object_ids[row])
            if object_id in self.observed_generation_by_object_id:
                continue
            registered_new_id = True
            if inherited_event_id is None:
                self.observed_generation_by_object_id[object_id] = 0
                self.observed_parent_event_by_object_id[object_id] = -1
            else:
                self.observed_generation_by_object_id[object_id] = (
                    self.observed_event_generation_by_id[inherited_event_id]
                )
                self.observed_parent_event_by_object_id[object_id] = inherited_event_id

        if registered_new_id or (
            self.pending_observed_creator_frame is not None
            and self.current_frame > self.pending_observed_creator_frame + 1
        ):
            self.pending_observed_creator_event_id = None
            self.pending_observed_creator_frame = None

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

    def _accept_true_cascade_collisions(
        self,
        particle_collisions: list[ParticleCollision],
        step_start_time: float,
    ) -> list[tuple[float, ParticleCollision]]:
        """Accept deterministic destructive true-cascade collisions."""
        accepted_collisions = []
        used_object_ids = set()

        for collision in particle_collisions:
            if len(self.true_collision_events) >= config.MAX_TRUE_COLLISION_EVENTS:
                self._apply_cascade_cap("max_true_collision_events")
                break

            parent_i = min(collision.particle_a_id, collision.particle_b_id)
            parent_j = max(collision.particle_a_id, collision.particle_b_id)
            if parent_i in used_object_ids or parent_j in used_object_ids:
                continue

            row_i = self._get_row_for_object_id(parent_i)
            row_j = self._get_row_for_object_id(parent_j)
            if not self.active_mask[row_i] or not self.active_mask[row_j]:
                continue

            used_object_ids.update((parent_i, parent_j))
            collision_time = step_start_time + collision.time_offset_model_years
            generation = 1 + max(int(self.generations[row_i]), int(self.generations[row_j]))
            requested_fragments = self._calculate_fragment_count(row_i, row_j)
            fragments_to_create = self._apply_population_cap(requested_fragments)

            self.active_mask[[row_i, row_j]] = False
            self._clear_collision_pair_state_for_objects((parent_i, parent_j))
            event_id = self.next_true_collision_event_id
            self.next_true_collision_event_id += 1

            if fragments_to_create > 0:
                self._create_fragments(
                    event_id,
                    generation,
                    collision.position,
                    row_i,
                    row_j,
                    fragments_to_create,
                )

            event = TrueCollisionEvent(
                event_id=event_id,
                model_time_years=collision_time,
                parent_i=parent_i,
                parent_j=parent_j,
                parent_event_i=int(self.parent_event_ids[row_i]),
                parent_event_j=int(self.parent_event_ids[row_j]),
                generation=generation,
                fragments_created=fragments_to_create,
                population_after=self.get_active_particle_count(),
            )
            self.true_collision_events.append(event)
            self._record_true_cascade_onset(event)
            accepted_collisions.append((collision_time, collision))

            if fragments_to_create < requested_fragments:
                self._apply_cascade_cap("population_cap")
            if len(self.true_collision_events) >= config.MAX_TRUE_COLLISION_EVENTS:
                self._apply_cascade_cap("max_true_collision_events")

        return accepted_collisions

    def _is_true_cascade_mode_enabled(self) -> bool:
        """Return True when true collisions should remove parents and fragment."""
        return config.FRAGMENTATION_MODE != "disabled"

    def _calculate_fragment_count(self, row_i: int, row_j: int) -> int:
        """Return requested educational fragment count for one event."""
        if self.fragmentation_disabled_by_policy:
            return 0
        if config.FRAGMENTATION_MODE == "fixed-count":
            return int(config.FIXED_FRAGMENT_COUNT)
        if config.FRAGMENTATION_MODE == "size-dependent-stochastic":
            mean_count = (
                config.SIZE_DEPENDENT_FRAGMENT_BASE_COUNT
                + (self.radii[row_i] + self.radii[row_j])
                * config.SIZE_DEPENDENT_FRAGMENT_COUNT_PER_KM_RADIUS
            )
            if config.SIZE_DEPENDENT_FRAGMENT_COUNT_STD > 0.0:
                sampled_count = self.fragmentation_random_generator.normal(
                    loc=mean_count,
                    scale=config.SIZE_DEPENDENT_FRAGMENT_COUNT_STD,
                )
            else:
                sampled_count = mean_count
            return max(0, int(round(sampled_count)))
        return 0

    def _apply_population_cap(self, requested_fragments: int) -> int:
        """Limit fragments so active population does not exceed the configured cap."""
        active_after_parent_removal = self.get_active_particle_count() - 2
        available_slots = max(0, int(config.MAX_ACTIVE_OBJECTS) - active_after_parent_removal)
        return min(int(requested_fragments), available_slots)

    def _create_fragments(
        self,
        event_id: int,
        generation: int,
        collision_position: np.ndarray,
        row_i: int,
        row_j: int,
        fragment_count: int,
    ) -> None:
        """Append deterministic debris fragments around the collision position."""
        radii = self._sample_fragment_radii(fragment_count)
        positions = self._calculate_fragment_release_positions(
            collision_position,
            radii,
        )
        reference_velocity = self._calculate_fragment_reference_velocity(row_i, row_j)
        velocities = reference_velocity + self._sample_fragment_velocity_kicks(
            fragment_count
        )

        object_ids = np.arange(
            self.next_object_id,
            self.next_object_id + fragment_count,
            dtype=int,
        )
        self.next_object_id += fragment_count
        self.object_ids = np.concatenate([self.object_ids, object_ids])
        self.positions = np.vstack([self.positions, positions])
        self.velocities = np.vstack([self.velocities, velocities])
        self.radii = np.concatenate([self.radii, radii])
        self.active_mask = np.concatenate(
            [self.active_mask, np.ones(fragment_count, dtype=bool)]
        )
        self.object_types = np.concatenate(
            [self.object_types, np.array(["debris"] * fragment_count)]
        )
        self.generations = np.concatenate(
            [self.generations, np.full(fragment_count, generation, dtype=int)]
        )
        self.parent_event_ids = np.concatenate(
            [self.parent_event_ids, np.full(fragment_count, event_id, dtype=int)]
        )
        identities = np.array([f"D-{object_id:06d}" for object_id in object_ids])
        self.identities = np.concatenate([self.identities, identities])
        self.categories = np.concatenate(
            [self.categories, np.array(["Debris fragment"] * fragment_count)]
        )
        self.colors = np.concatenate(
            [self.colors, np.array(["tab:red"] * fragment_count)]
        )
        self.latest_observed_positions = np.vstack(
            [self.latest_observed_positions, positions.copy()]
        )
        self.latest_observation_errors = np.vstack(
            [self.latest_observation_errors, np.zeros((fragment_count, 2))]
        )

    def _sample_fragment_radii(self, fragment_count: int) -> np.ndarray:
        """Sample educational fragment radii from the configured distribution."""
        if config.FRAGMENT_RADIUS_DISTRIBUTION == "constant":
            return np.full(fragment_count, config.FRAGMENT_RADIUS_CONSTANT_KM, dtype=float)
        return self.fragmentation_random_generator.uniform(
            config.FRAGMENT_RADIUS_MIN_KM,
            config.FRAGMENT_RADIUS_MAX_KM,
            size=fragment_count,
        )

    def _calculate_fragment_release_positions(
        self,
        collision_position: np.ndarray,
        radii: np.ndarray,
    ) -> np.ndarray:
        """Place sibling fragments on a deterministic ring to avoid instant overlap."""
        fragment_count = len(radii)
        if fragment_count == 0:
            return np.empty((0, 2), dtype=float)
        largest_radius = float(np.max(radii))
        release_radius = max(
            config.FRAGMENT_RELEASE_OFFSET_FACTOR * largest_radius * fragment_count,
            config.FRAGMENT_RELEASE_OFFSET_FACTOR * largest_radius,
        )
        phase = self.fragmentation_random_generator.uniform(0.0, 2.0 * np.pi)
        angles = phase + np.arange(fragment_count) * (2.0 * np.pi / fragment_count)
        offsets = np.column_stack([np.cos(angles), np.sin(angles)]) * release_radius
        return self._wrap_positions(collision_position[np.newaxis, :] + offsets)

    def _calculate_fragment_reference_velocity(
        self,
        row_i: int,
        row_j: int,
    ) -> np.ndarray:
        """Return the configured parent-reference velocity for new fragments."""
        if config.FRAGMENT_VELOCITY_REFERENCE == "parent-i":
            return self.velocities[row_i].copy()
        if config.FRAGMENT_VELOCITY_REFERENCE == "parent-j":
            return self.velocities[row_j].copy()
        return 0.5 * (self.velocities[row_i] + self.velocities[row_j])

    def _sample_fragment_velocity_kicks(self, fragment_count: int) -> np.ndarray:
        """Sample deterministic isotropic velocity perturbations."""
        if fragment_count == 0:
            return np.empty((0, 2), dtype=float)
        scale = float(config.FRAGMENT_VELOCITY_KICK_SCALE_KM_PER_MODEL_YEAR)
        angles = self.fragmentation_random_generator.uniform(
            0.0,
            2.0 * np.pi,
            size=fragment_count,
        )
        if config.FRAGMENT_VELOCITY_KICK_DISTRIBUTION == "uniform":
            magnitudes = self.fragmentation_random_generator.uniform(
                0.0,
                scale,
                size=fragment_count,
            )
        else:
            magnitudes = np.abs(
                self.fragmentation_random_generator.normal(
                    loc=0.0,
                    scale=scale,
                    size=fragment_count,
                )
            )
        return np.column_stack([np.cos(angles), np.sin(angles)]) * magnitudes[:, None]

    def _record_true_cascade_onset(self, event: TrueCollisionEvent) -> None:
        """Store first true cascade onset, preserving None when no onset occurs."""
        if self.true_cascade_onset_time_model_years is not None:
            return
        if event.generation >= config.CASCADE_ONSET_GENERATION:
            self.true_cascade_onset_time_model_years = event.model_time_years
            self.true_cascade_onset_generation = event.generation

    def _apply_cascade_cap(self, reason: str) -> None:
        """Apply configured termination policy after an event/population cap."""
        if self.cascade_termination_reason == "":
            self.cascade_termination_reason = reason
        if config.CASCADE_TERMINATION_POLICY == "stop":
            self.cascade_terminated = True
        else:
            self.fragmentation_disabled_by_policy = True

    def _clear_collision_pair_state_for_objects(self, object_ids: tuple[int, ...]) -> None:
        """Remove active/cooldown pair state for deactivated objects."""
        removed_ids = set(object_ids)
        self.active_particle_collision_pairs = {
            pair for pair in self.active_particle_collision_pairs
            if pair[0] not in removed_ids and pair[1] not in removed_ids
        }
        self.last_particle_collision_times = {
            pair: collision_time
            for pair, collision_time in self.last_particle_collision_times.items()
            if pair[0] not in removed_ids and pair[1] not in removed_ids
        }

    def _materialize_pair_step_classifications_if_needed(
        self,
        pair_a: np.ndarray,
        pair_b: np.ndarray,
        true_result: dict[str, np.ndarray],
        observed_result: dict[str, np.ndarray],
        true_collision: np.ndarray,
        observed_collision: np.ndarray,
        classifications: np.ndarray,
        retained_classifications: list[PairStepClassification],
    ) -> None:
        """Stream diagnostics or retain bounded non-TN examples for one chunk."""
        if config.SAVE_PAIR_STEP_DIAGNOSTIC_CSV:
            indices_to_materialize = range(len(classifications))
        elif config.MAX_RETAINED_PAIR_STEP_CLASSIFICATIONS > 0:
            remaining_slots = (
                config.MAX_RETAINED_PAIR_STEP_CLASSIFICATIONS
                - len(retained_classifications)
            )
            if remaining_slots <= 0:
                indices_to_materialize = []
            else:
                indices_to_materialize = np.nonzero(classifications != "TN")[0][
                    :remaining_slots
                ]
        else:
            indices_to_materialize = []

        for index in indices_to_materialize:
            label = classifications[index]
            classification = self._create_pair_step_classification(
                pair_a[index],
                pair_b[index],
                true_result,
                observed_result,
                bool(true_collision[index]),
                bool(observed_collision[index]),
                str(label),
                index,
            )
            if label != "TN":
                retained_classifications.append(classification)
            self._write_pair_step_diagnostic_record_if_enabled(classification)

    def _get_observed_warning_pairs(
        self,
        pair_a: np.ndarray,
        pair_b: np.ndarray,
        observed_collision: np.ndarray,
    ) -> set[tuple[int, int]]:
        """Return observed-collision pair keys from one chunk."""
        return {
            self._get_pair_key_from_ids(
                int(self.object_ids[pair_a[index]]),
                int(self.object_ids[pair_b[index]]),
            )
            for index in np.nonzero(observed_collision)[0]
        }

    def _get_observed_warning_candidates(
        self,
        pair_a: np.ndarray,
        pair_b: np.ndarray,
        observed_result: dict[str, np.ndarray],
        observed_collision: np.ndarray,
    ) -> list[dict[str, object]]:
        """Return observed-collision candidates with observed closest-approach times."""
        candidates = []
        for index in np.nonzero(observed_collision)[0]:
            parent_i = int(self.object_ids[pair_a[index]])
            parent_j = int(self.object_ids[pair_b[index]])
            candidates.append(
                {
                    "pair_key": self._get_pair_key_from_ids(parent_i, parent_j),
                    "parent_i": min(parent_i, parent_j),
                    "parent_j": max(parent_i, parent_j),
                    "time_offset_model_years": float(observed_result["times"][index]),
                }
            )
        return candidates

    def _update_observed_warning_event_state(
        self,
        observed_pairs: set[tuple[int, int]],
        observed_candidates: list[dict[str, object]],
    ) -> None:
        """Deduplicate prolonged observed collision warnings once per step."""
        new_pairs = observed_pairs - self.active_observed_warning_pairs
        self.observed_warning_event_count += len(new_pairs)
        self._record_observed_cascade_events(
            [
                candidate
                for candidate in observed_candidates
                if candidate["pair_key"] in new_pairs
            ]
        )
        self.active_observed_warning_pairs = observed_pairs

    def _record_observed_cascade_events(
        self,
        observed_candidates: list[dict[str, object]],
    ) -> None:
        """Record deduplicated observed events using observed identity ancestry."""
        used_object_ids = set()
        observed_candidates.sort(
            key=lambda candidate: (
                candidate["time_offset_model_years"],
                candidate["parent_i"],
                candidate["parent_j"],
            )
        )
        for candidate in observed_candidates:
            parent_i = int(candidate["parent_i"])
            parent_j = int(candidate["parent_j"])
            if parent_i in used_object_ids or parent_j in used_object_ids:
                continue
            used_object_ids.update((parent_i, parent_j))
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
                    self.current_time + float(candidate["time_offset_model_years"])
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
            self.pending_observed_creator_event_id = event_id
            self.pending_observed_creator_frame = self.current_frame
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

    def _record_boundary_crossings(self, crossings: list[BoundaryCrossing]) -> None:
        """Store detailed records for any boundary crossings this frame."""
        for crossing in crossings:
            row = self._get_row_for_object_id(crossing.particle_id)
            self.boundary_crossing_records.append(
                {
                    "frame": self.current_frame,
                    "simulation_time_model_years": self.current_time,
                    "satellite_identity": str(self.identities[row]),
                    "satellite_category": str(self.categories[row]),
                    "speed_km_per_model_year": self._get_speed_for_row(row),
                    "boundary": crossing.boundary,
                    "total_boundary_crossings": self.boundary_crossing_count,
                }
            )

    def _record_particle_collisions(
        self,
        timed_collisions: list[tuple[float, ParticleCollision]],
    ) -> None:
        """Store detailed records for any particle collisions in this frame."""
        for collision_time, collision in timed_collisions:
            row_a = self._get_row_for_object_id(collision.particle_a_id)
            row_b = self._get_row_for_object_id(collision.particle_b_id)
            self.particle_collision_records.append(
                {
                    "simulation_time_model_years": collision_time,
                    "satellite_a": str(self.identities[row_a]),
                    "satellite_b": str(self.identities[row_b]),
                    "x_position": float(collision.position[0]),
                    "y_position": float(collision.position[1]),
                    "relative_speed_km_per_model_year": collision.relative_speed,
                }
            )

    def _record_position_snapshot(self) -> None:
        """Store active positions for visual-style trajectory plotting."""
        active_rows = self._get_active_rows()
        self.position_history.append(
            {
                "frame": self.current_frame,
                "time": self.current_time,
                "positions": self.positions[active_rows].copy(),
                "identities": [str(self.identities[row]) for row in active_rows],
                "colors": [str(self.colors[row]) for row in active_rows],
            }
        )

    def _write_pair_step_diagnostic_record_if_enabled(
        self,
        classification: PairStepClassification,
    ) -> None:
        """Stream optional pair-step diagnostics without retaining all rows."""
        if not config.SAVE_PAIR_STEP_DIAGNOSTIC_CSV:
            return
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
        """Convert one pair-step classification to a CSV record."""
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

    def get_pair_step_summary(self) -> dict[str, object]:
        """Return one lightweight run-level classification summary."""
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
            "precision": self._safe_rate(counts["TP"], counts["TP"] + counts["FP"]),
            "true_deduplicated_collision_event_count": self.particle_collision_count,
            "observed_deduplicated_warning_event_count": (
                self.observed_warning_event_count
            ),
            "true_collision_event_count": len(self.true_collision_events),
            "true_cascade_onset_time_model_years": (
                self.true_cascade_onset_time_model_years
            ),
            "true_cascade_onset_generation": self.true_cascade_onset_generation,
            "cascade_termination_reason": self.cascade_termination_reason,
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

    def save_observed_cascade_events_data(self) -> Path:
        """Write deduplicated observed cascade events to CSV."""
        output_directory = Path(config.OUTPUT_DIRECTORY)
        output_directory.mkdir(parents=True, exist_ok=True)
        output_path = output_directory / config.OBSERVED_CASCADE_EVENTS_FILENAME
        fieldnames = [
            "event_id",
            "model_time_years",
            "parent_i",
            "parent_j",
            "parent_event_i",
            "parent_event_j",
            "generation",
            "event_layer",
        ]
        with output_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            for event in self.observed_cascade_events:
                writer.writerow(
                    {
                        "event_id": event.event_id,
                        "model_time_years": event.model_time_years,
                        "parent_i": event.parent_i,
                        "parent_j": event.parent_j,
                        "parent_event_i": event.parent_event_i,
                        "parent_event_j": event.parent_event_j,
                        "generation": event.generation,
                        "event_layer": event.event_layer,
                    }
                )
        return output_path

    def save_true_collision_events_data(self) -> Path:
        """Write accepted true cascade collision events to CSV."""
        output_directory = Path(config.OUTPUT_DIRECTORY)
        output_directory.mkdir(parents=True, exist_ok=True)
        output_path = output_directory / config.TRUE_COLLISION_EVENTS_FILENAME
        fieldnames = [
            "event_id",
            "model_time_years",
            "parent_i",
            "parent_j",
            "parent_event_i",
            "parent_event_j",
            "generation",
            "fragments_created",
            "population_after",
            "event_layer",
        ]
        with output_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            for event in self.true_collision_events:
                writer.writerow(
                    {
                        "event_id": event.event_id,
                        "model_time_years": event.model_time_years,
                        "parent_i": event.parent_i,
                        "parent_j": event.parent_j,
                        "parent_event_i": event.parent_event_i,
                        "parent_event_j": event.parent_event_j,
                        "generation": event.generation,
                        "fragments_created": event.fragments_created,
                        "population_after": event.population_after,
                        "event_layer": event.event_layer,
                    }
                )
        return output_path

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

    def close_pair_step_diagnostic_data(self) -> None:
        """Close the optional streamed diagnostic CSV if it was opened."""
        if self.pair_step_diagnostic_file is not None:
            self.pair_step_diagnostic_file.close()
            self.pair_step_diagnostic_file = None
            self.pair_step_diagnostic_writer = None

    def get_active_particle_count(self) -> int:
        """Return the number of active rows."""
        return int(np.count_nonzero(self.active_mask))

    def get_pairs_per_step(self) -> int:
        """Return active unordered pairs for one step."""
        active_count = self.get_active_particle_count()
        return active_count * (active_count - 1) // 2

    def get_average_particle_speed(self) -> float:
        """Return the average speed over active rows."""
        active_rows = self._get_active_rows()
        if len(active_rows) == 0:
            return 0.0
        return float(np.mean(np.linalg.norm(self.velocities[active_rows], axis=1)))

    def get_particle_density(self) -> float:
        """Return active rows per square kilometer."""
        return self.get_active_particle_count() / self.get_region_area()

    def get_region_area(self) -> float:
        """Return the represented rectangular area."""
        return self.box_width * self.box_height

    def get_display_time(self) -> float:
        """Return elapsed simulation time converted to display units."""
        return config.model_years_to_display_time(self.current_time)

    def get_display_time_unit(self) -> str:
        """Return the configured display time unit."""
        return config.DISPLAY_TIME_UNIT

    @property
    def particles(self) -> list[Particle]:
        """Adapt active array rows to Particle objects for reference-only callers."""
        return self.to_particles()

    def to_particles(self) -> list[Particle]:
        """Return Particle objects for active rows in stable row order."""
        particles = []
        for row in self._get_active_rows():
            particles.append(
                Particle(
                    particle_id=int(self.object_ids[row]),
                    identity=str(self.identities[row]),
                    position=self.positions[row].copy(),
                    velocity=self.velocities[row].copy(),
                    radius=float(self.radii[row]),
                    color=str(self.colors[row]),
                    category=str(self.categories[row]),
                    category_key=str(self.object_types[row]),
                )
            )
        return particles

    def _get_active_rows(self) -> np.ndarray:
        """Return active row indices in stable row order."""
        return np.nonzero(self.active_mask)[0]

    def _iter_active_pair_row_chunks(
        self,
        active_rows: np.ndarray,
    ):
        """Yield active unordered pair row chunks in stable pair order."""
        chunk_size = max(1, int(self.pair_chunk_size))
        active_count = len(active_rows)
        first_local_index = 0

        while first_local_index < active_count - 1:
            row_pair_count = active_count - first_local_index - 1
            if row_pair_count > chunk_size:
                second_local_start = first_local_index + 1
                while second_local_start < active_count:
                    second_local_stop = min(
                        active_count,
                        second_local_start + chunk_size,
                    )
                    pair_b = active_rows[second_local_start:second_local_stop]
                    pair_a = np.full(len(pair_b), active_rows[first_local_index], dtype=int)
                    yield pair_a, pair_b
                    second_local_start = second_local_stop
                first_local_index += 1
                continue

            chunk_first_start = first_local_index
            chunk_first_stop = first_local_index
            chunk_pair_count = 0
            while chunk_first_stop < active_count - 1:
                next_row_pair_count = active_count - chunk_first_stop - 1
                if (
                    chunk_pair_count > 0
                    and chunk_pair_count + next_row_pair_count > chunk_size
                ):
                    break
                chunk_pair_count += next_row_pair_count
                chunk_first_stop += 1

            local_first_indices = np.arange(chunk_first_start, chunk_first_stop)
            pair_lengths = active_count - local_first_indices - 1
            pair_a = np.repeat(active_rows[local_first_indices], pair_lengths)
            pair_b = np.concatenate(
                [
                    active_rows[local_index + 1 : active_count]
                    for local_index in local_first_indices
                ]
            )
            yield pair_a, pair_b
            first_local_index = chunk_first_stop

    def _minimum_image_vectors(self, displacements: np.ndarray) -> np.ndarray:
        """Return nearest periodic images of 2D displacement vectors."""
        box_size = np.array([self.box_width, self.box_height], dtype=float)
        return ((displacements + 0.5 * box_size) % box_size) - 0.5 * box_size

    def _wrap_positions(self, positions: np.ndarray) -> np.ndarray:
        """Wrap positions into the rectangular periodic domain."""
        wrapped = positions.copy()
        wrapped[:, 0] %= self.box_width
        wrapped[:, 1] %= self.box_height
        return wrapped

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
        """Return True when a separated pair is still inside cooldown."""
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

    def _create_initialization_fingerprint(self) -> str:
        """Return a short hash of active initial identities, positions, velocities."""
        fingerprint_text_parts = []
        for row in self._get_active_rows():
            fingerprint_text_parts.append(
                "|".join(
                    [
                        str(self.identities[row]),
                        f"{self.positions[row, 0]:.12f}",
                        f"{self.positions[row, 1]:.12f}",
                        f"{self.velocities[row, 0]:.12f}",
                        f"{self.velocities[row, 1]:.12f}",
                    ]
                )
            )
        fingerprint_text = "\n".join(fingerprint_text_parts)
        return hashlib.sha256(fingerprint_text.encode("utf-8")).hexdigest()[:16]

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

    def _should_record_position_snapshot(self) -> bool:
        """Return True when this frame should be kept for trajectory output."""
        return self.current_frame % self.trajectory_record_interval_frames == 0

    def _validate_pair_step_summary_counts(self) -> None:
        """Validate run-level pair-step count identities."""
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
        expected_total = self.expected_pair_step_comparisons
        if self.total_pair_step_comparisons != expected_total:
            raise RuntimeError(
                "Pair-step comparison total mismatch: "
                f"expected {expected_total}, got {self.total_pair_step_comparisons}."
            )

    def _get_row_for_object_id(self, object_id: int) -> int:
        """Return the row for a stable object ID."""
        matches = np.nonzero(self.object_ids == object_id)[0]
        if len(matches) != 1:
            raise ValueError(f"No object row found for object ID {object_id}.")
        return int(matches[0])

    def _get_speed_for_row(self, row: int) -> float:
        """Return speed magnitude for one row."""
        return float(np.linalg.norm(self.velocities[row]))

    def _safe_rate(self, numerator: int, denominator: int) -> float | None:
        """Return a rate, or None when denominator is zero."""
        if denominator == 0:
            return None
        return numerator / denominator

    def _format_optional_csv_value(self, value: object) -> object:
        """Write missing rate values as blank CSV cells."""
        if value is None:
            return ""
        return value

    def _get_display_time_scale(self) -> float:
        """Return one display unit expressed in canonical model years."""
        return config.display_time_config_value_to_years(1.0)



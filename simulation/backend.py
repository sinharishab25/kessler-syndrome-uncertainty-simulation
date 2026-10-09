"""Backend selection helpers for headless simulations."""

import numpy as np

import config
from simulation.array_backend import ArraySimulation
from simulation.simulation import Simulation


def create_headless_simulation(
    random_generator: np.random.Generator,
    run_seed: int,
    run_index: int = 0,
    particle_groups: list[dict] | None = None,
    population_value: int | None = None,
    observation_seed: int | None = None,
    position_uncertainty_sigma_km: float | None = None,
    minimum_initial_spacing: float | None = None,
):
    """Create the configured non-visual backend for headless execution."""
    backend = config.HEADLESS_BACKEND.strip().lower()
    simulation_class = ArraySimulation if backend == "array" else Simulation
    return simulation_class(
        random_generator=random_generator,
        run_seed=run_seed,
        run_index=run_index,
        record_position_history=False,
        particle_groups=particle_groups,
        population_value=population_value,
        observation_seed=observation_seed,
        position_uncertainty_sigma_km=position_uncertainty_sigma_km,
        minimum_initial_spacing=minimum_initial_spacing,
    )

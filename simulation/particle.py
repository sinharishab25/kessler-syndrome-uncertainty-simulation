"""Satellite particle state for the Version 1.5.5 simulation."""

from dataclasses import dataclass

import numpy as np


@dataclass
class Particle:
    """A moving satellite marker with persistent identity and category state.

    The particle stores its own identity, position, velocity, radius, and visual
    category. Boundary physics remains in physics.py so this state object stays
    lightweight and reusable for later Kessler experiments.
    """

    particle_id: int
    identity: str
    position: np.ndarray
    velocity: np.ndarray
    radius: float
    color: str
    category: str
    category_key: str

    def move(self, dt: float) -> None:
        """Advance the satellite marker by dt simulated seconds."""
        self.position = self.position + self.velocity * dt

    def get_speed(self) -> float:
        """Return the current satellite speed in kilometers per second."""
        return float(np.linalg.norm(self.velocity))

    def is_test(self) -> bool:
        """Return True when this marker is an editable test satellite."""
        return self.category_key == "test"







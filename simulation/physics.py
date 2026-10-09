"""Physics helpers for satellite movement and boundary crossings."""

from dataclasses import dataclass

from simulation.particle import Particle


@dataclass(frozen=True)
class BoundaryCrossing:
    """A record of one satellite crossing the simulation boundary."""

    particle_id: int
    boundary: str


def move_particle(particle: Particle, dt: float) -> None:
    """Move a particle forward by dt simulated seconds."""
    particle.move(dt)


def apply_wraparound_boundary_crossings(
    particle: Particle,
    box_width: float,
    box_height: float,
) -> list[BoundaryCrossing]:
    """Wrap satellites that leave the rectangular region to the opposite side."""
    crossings = []

    if particle.position[0] < 0.0:
        particle.position[0] = particle.position[0] % box_width
        crossings.append(BoundaryCrossing(particle.particle_id, "left"))
    elif particle.position[0] > box_width:
        particle.position[0] = particle.position[0] % box_width
        crossings.append(BoundaryCrossing(particle.particle_id, "right"))

    if particle.position[1] < 0.0:
        particle.position[1] = particle.position[1] % box_height
        crossings.append(BoundaryCrossing(particle.particle_id, "bottom"))
    elif particle.position[1] > box_height:
        particle.position[1] = particle.position[1] % box_height
        crossings.append(BoundaryCrossing(particle.particle_id, "top"))

    return crossings

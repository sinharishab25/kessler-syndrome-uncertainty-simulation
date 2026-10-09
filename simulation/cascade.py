"""Cascade event records for educational true and observed runs."""

from dataclasses import dataclass


@dataclass(frozen=True)
class TrueCollisionEvent:
    """One accepted true collision event in the cascade graph."""

    event_id: int
    model_time_years: float
    parent_i: int
    parent_j: int
    parent_event_i: int
    parent_event_j: int
    generation: int
    fragments_created: int
    population_after: int
    event_layer: str = "true"


@dataclass(frozen=True)
class ObservedCascadeEvent:
    """One deduplicated observed collision event in the observation layer."""

    event_id: int
    model_time_years: float
    parent_i: int
    parent_j: int
    parent_event_i: int
    parent_event_j: int
    generation: int
    event_layer: str = "observed"

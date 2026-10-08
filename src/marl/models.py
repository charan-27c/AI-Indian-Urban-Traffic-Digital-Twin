# src/marl/models.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class JunctionObservation:
    junction_id: str

    vehicle_count: float = 0.0
    queue_length: float = 0.0
    waiting_time: float = 0.0
    average_speed: float = 0.0
    pressure: float = 0.0

    predicted_pressure: float = 0.0
    predicted_queue_length: float = 0.0

    emergency_vehicle_count: float = 0.0
    emergency_distance: float = 0.0
    emergency_route_relevance: float = 0.0

    incident_severity: float = 0.0
    blocked_movement_count: float = 0.0

    current_phase: float = 0.0
    phase_elapsed: float = 0.0

    neighbor_pressure: float = 0.0

    def to_vector(self) -> Tuple[float, ...]:
        return (
            self.vehicle_count,
            self.queue_length,
            self.waiting_time,
            self.average_speed,
            self.pressure,
            self.predicted_pressure,
            self.predicted_queue_length,
            self.emergency_vehicle_count,
            self.emergency_distance,
            self.emergency_route_relevance,
            self.incident_severity,
            self.blocked_movement_count,
            self.current_phase,
            self.phase_elapsed,
            self.neighbor_pressure,
        )


@dataclass(frozen=True)
class JunctionAction:
    junction_id: str

    movement_id: str | None = None

    green_extension: float = 0.0

    emergency_priority: bool = False

    request_all_red: bool = False


@dataclass(frozen=True)
class MARLReward:
    junction_id: str

    throughput_reward: float = 0.0
    queue_penalty: float = 0.0
    waiting_penalty: float = 0.0
    emergency_reward: float = 0.0
    coordination_reward: float = 0.0
    safety_penalty: float = 0.0

    @property
    def total(self) -> float:
        return (
            self.throughput_reward
            + self.queue_penalty
            + self.waiting_penalty
            + self.emergency_reward
            + self.coordination_reward
            + self.safety_penalty
        )
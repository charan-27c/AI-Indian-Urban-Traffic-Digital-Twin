from __future__ import annotations

from typing import Iterable, Mapping

from .models import JunctionObservation


class MARLObservationBuilder:
    """
    Converts dynamic traffic-state information into
    observations for junction agents.

    No junction IDs are hardcoded.
    """

    def __init__(
        self,
        emergency_distance_scale: float = 1000.0,
    ) -> None:
        self.emergency_distance_scale = float(
            emergency_distance_scale
        )

    @staticmethod
    def _value(
        obj,
        name: str,
        default: float = 0.0,
    ) -> float:
        value = getattr(obj, name, default)

        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    def build(
        self,
        junction,
        *,
        predicted_pressure: float = 0.0,
        predicted_queue_length: float = 0.0,
        emergency_vehicle_count: float = 0.0,
        emergency_distance: float = 0.0,
        emergency_route_relevance: float = 0.0,
        incident_severity: float = 0.0,
        blocked_movement_count: float = 0.0,
        current_phase: float = 0.0,
        phase_elapsed: float = 0.0,
        neighbor_pressure: float = 0.0,
    ) -> JunctionObservation:

        junction_id = str(
            getattr(junction, "junction_id", "")
        )

        if not junction_id:
            raise ValueError(
                "junction_id cannot be empty."
            )

        return JunctionObservation(
            junction_id=junction_id,
            vehicle_count=max(
                0.0,
                self._value(
                    junction,
                    "vehicle_count",
                ),
            ),
            queue_length=max(
                0.0,
                self._value(
                    junction,
                    "queue_length",
                ),
            ),
            waiting_time=max(
                0.0,
                self._value(
                    junction,
                    "waiting_time",
                ),
            ),
            average_speed=max(
                0.0,
                self._value(
                    junction,
                    "average_speed",
                ),
            ),
            pressure=max(
                0.0,
                self._value(
                    junction,
                    "pressure",
                ),
            ),
            predicted_pressure=max(
                0.0,
                float(predicted_pressure),
            ),
            predicted_queue_length=max(
                0.0,
                float(predicted_queue_length),
            ),
            emergency_vehicle_count=max(
                0.0,
                float(emergency_vehicle_count),
            ),
            emergency_distance=max(
                0.0,
                float(emergency_distance),
            ),
            emergency_route_relevance=max(
                0.0,
                min(
                    1.0,
                    float(emergency_route_relevance),
                ),
            ),
            incident_severity=max(
                0.0,
                min(
                    1.0,
                    float(incident_severity),
                ),
            ),
            blocked_movement_count=max(
                0.0,
                float(blocked_movement_count),
            ),
            current_phase=max(
                0.0,
                float(current_phase),
            ),
            phase_elapsed=max(
                0.0,
                float(phase_elapsed),
            ),
            neighbor_pressure=max(
                0.0,
                float(neighbor_pressure),
            ),
        )

    def build_all(
        self,
        junctions: Iterable,
        **kwargs,
    ) -> dict[str, JunctionObservation]:

        observations = {}

        for junction in junctions:
            observation = self.build(
                junction,
                **kwargs,
            )

            observations[
                observation.junction_id
            ] = observation

        return observations
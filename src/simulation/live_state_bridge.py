from __future__ import annotations

from typing import Any, Mapping

from traffic_state.models import VehicleState


class LiveSUMOVehicleStateAdapter:
    """
    Converts live SUMO/TraCI vehicle telemetry into the
    project's canonical VehicleState model.

    No junction, lane, edge, or vehicle-class IDs are hardcoded.
    """

    EMERGENCY_CLASSES = {
        "ambulance",
        "fire_engine",
        "police_vehicle",
        "emergency",
    }

    def __init__(
        self,
        queue_speed_threshold: float = 1.0,
    ) -> None:
        self.queue_speed_threshold = float(
            queue_speed_threshold
        )

    # ========================================================
    # SAFE FIELD ACCESS
    # ========================================================

    @staticmethod
    def _read(
        data: Mapping[str, Any],
        key: str,
        default: Any = None,
    ) -> Any:
        return data.get(key, default)

    # ========================================================
    # VEHICLE CLASS
    # ========================================================

    def _vehicle_class(
        self,
        data: Mapping[str, Any],
    ) -> str:
        value = (
            self._read(
                data,
                "vehicle_class",
                None,
            )
        )

        if value is None:
            value = (
                self._read(
                    data,
                    "vClass",
                    None,
                )
            )

        if value is None:
            value = (
                self._read(
                    data,
                    "type",
                    None,
                )
            )

        if value is None:
            value = "unknown"

        return str(value)

    # ========================================================
    # CONVERT ONE VEHICLE
    # ========================================================

    def convert(
        self,
        vehicle_id: str,
        data: Mapping[str, Any],
    ) -> VehicleState:

        if not vehicle_id:
            raise ValueError(
                "vehicle_id cannot be empty."
            )

        lane_id = str(
            self._read(
                data,
                "lane_id",
                "",
            )
        )

        edge_id = str(
            self._read(
                data,
                "edge_id",
                "",
            )
        )

        speed = float(
            self._read(
                data,
                "speed",
                0.0,
            )
        )

        waiting_time = float(
            self._read(
                data,
                "waiting_time",
                0.0,
            )
        )

        vehicle_class = (
            self._vehicle_class(data)
        )

        stopped = (
            speed <= self.queue_speed_threshold
        )

        return VehicleState(
            vehicle_id=str(vehicle_id),
            vehicle_class=vehicle_class,
            lane_id=lane_id,
            edge_id=edge_id,
            speed=max(0.0, speed),
            waiting_time=max(
                0.0,
                waiting_time,
            ),
            stopped=stopped,
        )

    # ========================================================
    # CONVERT SNAPSHOT
    # ========================================================

    def convert_snapshot(
        self,
        snapshot: Mapping[
            str,
            Mapping[str, Any]
        ],
    ) -> list[VehicleState]:

        vehicles = []

        for vehicle_id, data in snapshot.items():

            try:
                vehicles.append(
                    self.convert(
                        vehicle_id,
                        data,
                    )
                )
            except (
                TypeError,
                ValueError,
                KeyError,
            ):
                continue

        return vehicles
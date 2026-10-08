from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class VehicleObservation:
    """
    One tracked-vehicle observation at a single timestamp.
    """

    vehicle_id: str

    timestamp: float

    x: float
    y: float

    speed: float = 0.0

    lane_id: Optional[str] = None
    movement_id: Optional[str] = None
    junction_id: Optional[str] = None

    heading: Optional[float] = None


@dataclass(frozen=True)
class TrajectoryFeatures:
    """
    Derived features for one tracked vehicle.
    """

    vehicle_id: str

    speed: float
    acceleration: float

    displacement: float

    stationary_duration: float

    heading_change: float

    wrong_way_score: float

    sudden_braking_score: float

    lane_change: bool

import math


class TrajectoryFeatureExtractor:
    """
    Converts consecutive vehicle observations into
    trajectory-level features.

    No fixed junction, lane, or movement is assumed.
    """

    def extract(
        self,
        previous: VehicleObservation,
        current: VehicleObservation,
    ) -> TrajectoryFeatures:

        if previous.vehicle_id != current.vehicle_id:
            raise ValueError(
                "Observations must belong to the same vehicle."
            )

        dt = current.timestamp - previous.timestamp

        if dt <= 0:
            raise ValueError(
                "Current timestamp must be greater than "
                "previous timestamp."
            )

        dx = current.x - previous.x
        dy = current.y - previous.y

        displacement = math.sqrt(
            dx * dx + dy * dy
        )

        acceleration = (
            current.speed - previous.speed
        ) / dt

        if current.speed <= 1.0:
            stationary_duration = dt
        else:
            stationary_duration = 0.0

        heading_change = 0.0

        if (
            previous.heading is not None
            and current.heading is not None
        ):
            heading_change = abs(
                current.heading
                - previous.heading
            )

            # Normalize angle difference to [0, 180].
            heading_change %= 360.0

            if heading_change > 180.0:
                heading_change = (
                    360.0 - heading_change
                )

        lane_change = (
            previous.lane_id is not None
            and current.lane_id is not None
            and previous.lane_id
            != current.lane_id
        )

        return TrajectoryFeatures(
            vehicle_id=current.vehicle_id,

            speed=current.speed,

            acceleration=acceleration,

            displacement=displacement,

            stationary_duration=(
                stationary_duration
            ),

            heading_change=heading_change,

            wrong_way_score=0.0,

            sudden_braking_score=0.0,

            lane_change=lane_change,
        )
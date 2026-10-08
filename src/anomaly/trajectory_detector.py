from typing import Optional

from .models import (
    AnomalyEvent,
    AnomalyType,
)

from .trajectory_features import (
    TrajectoryFeatures,
)


class TrajectoryAnomalyDetector:
    """
    Detects anomalies from tracked vehicle trajectories.
    """

    def __init__(
        self,
        stationary_speed_threshold: float = 1.0,
        sudden_braking_threshold: float = -5.0,
        stationary_duration_threshold: float = 30.0,
    ) -> None:

        self.stationary_speed_threshold = (
            stationary_speed_threshold
        )

        self.sudden_braking_threshold = (
            sudden_braking_threshold
        )

        self.stationary_duration_threshold = (
            stationary_duration_threshold
        )

    def detect(
        self,
        features: TrajectoryFeatures,
        timestamp: float,
        junction_id: Optional[str] = None,
        movement_id: Optional[str] = None,
        lane_id: Optional[str] = None,
    ) -> list[AnomalyEvent]:

        events = []

        # ====================================================
        # STOPPED VEHICLE
        # ====================================================

        if (
            features.speed
            <= self.stationary_speed_threshold
            and
            features.stationary_duration
            >= self.stationary_duration_threshold
        ):

            events.append(
                AnomalyEvent(
                    anomaly_type=(
                        AnomalyType.STOPPED_VEHICLE
                    ),
                    confidence=0.90,
                    timestamp=timestamp,
                    junction_id=junction_id,
                    movement_id=movement_id,
                    lane_id=lane_id,
                    severity=0.60,
                    description=(
                        "Vehicle has remained stationary "
                        "for an abnormal duration."
                    ),
                )
            )

        # ====================================================
        # SUDDEN BRAKING
        # ====================================================

        if (
            features.acceleration
            <= self.sudden_braking_threshold
        ):

            events.append(
                AnomalyEvent(
                    anomaly_type=(
                        AnomalyType.SUDDEN_CONGESTION
                    ),
                    confidence=0.70,
                    timestamp=timestamp,
                    junction_id=junction_id,
                    movement_id=movement_id,
                    lane_id=lane_id,
                    severity=0.50,
                    description=(
                        "Sudden vehicle deceleration detected."
                    ),
                )
            )

        # ====================================================
        # WRONG WAY
        # ====================================================

        if features.wrong_way_score >= 0.8:

            events.append(
                AnomalyEvent(
                    anomaly_type=(
                        AnomalyType.WRONG_WAY
                    ),
                    confidence=features.wrong_way_score,
                    timestamp=timestamp,
                    junction_id=junction_id,
                    movement_id=movement_id,
                    lane_id=lane_id,
                    severity=0.90,
                    description=(
                        "Vehicle trajectory is inconsistent "
                        "with the permitted movement direction."
                    ),
                )
            )

        return events
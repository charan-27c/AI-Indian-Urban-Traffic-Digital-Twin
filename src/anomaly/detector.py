from .models import AnomalyEvent, AnomalyType
from .feature_engine import AnomalyFeatures


class TrafficAnomalyDetector:

    def __init__(
        self,
        stopped_speed_threshold: float = 1.0,
        congestion_speed_threshold: float = 3.0,
        queue_growth_threshold: float = 5.0,
        speed_drop_threshold: float = 5.0,
    ):

        self.stopped_speed_threshold = (
            stopped_speed_threshold
        )

        self.congestion_speed_threshold = (
            congestion_speed_threshold
        )

        self.queue_growth_threshold = (
            queue_growth_threshold
        )

        self.speed_drop_threshold = (
            speed_drop_threshold
        )

    def detect(
        self,
        features: AnomalyFeatures,
        timestamp: float,
        junction_id=None,
        movement_id=None,
        lane_id=None,
    ):

        events = []

        # ---------------------------------------------
        # SUDDEN CONGESTION
        # ---------------------------------------------

        if (
            features.speed_drop
            >= self.speed_drop_threshold
            and
            features.queue_growth
            >= self.queue_growth_threshold
        ):

            events.append(
                AnomalyEvent(
                    anomaly_type=AnomalyType.SUDDEN_CONGESTION,
                    confidence=0.8,
                    severity=0.7,
                    timestamp=timestamp,
                    junction_id=junction_id,
                    movement_id=movement_id,
                    lane_id=lane_id,
                    description=(
                        "Sudden traffic deterioration detected."
                    ),
                )
            )

        # ---------------------------------------------
        # BLOCKAGE
        # ---------------------------------------------

        if (
            features.queue_length
            > 10
            and
            features.average_speed
            <= self.congestion_speed_threshold
            and
            features.occupancy
            >= 0.7
        ):

            events.append(
                AnomalyEvent(
                    anomaly_type=AnomalyType.BLOCKAGE,
                    confidence=0.85,
                    severity=0.8,
                    timestamp=timestamp,
                    junction_id=junction_id,
                    movement_id=movement_id,
                    lane_id=lane_id,
                    description=(
                        "Possible road blockage detected."
                    ),
                )
            )

        # ---------------------------------------------
        # STOPPED VEHICLES
        # ---------------------------------------------

        if (
            features.stopped_vehicle_count > 0
            and
            features.stationary_duration >= 30
        ):

            events.append(
                AnomalyEvent(
                    anomaly_type=AnomalyType.STOPPED_VEHICLE,
                    confidence=0.9,
                    severity=0.6,
                    timestamp=timestamp,
                    junction_id=junction_id,
                    movement_id=movement_id,
                    lane_id=lane_id,
                    description=(
                        "Vehicle stopped for an abnormal duration."
                    ),
                )
            )

        return events
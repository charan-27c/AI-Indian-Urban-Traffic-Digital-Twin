from dataclasses import dataclass
from enum import Enum
from typing import Optional


class AnomalyType(str, Enum):
    NONE = "none"
    ACCIDENT = "accident"
    BLOCKAGE = "blockage"
    FLOODING = "flooding"
    STOPPED_VEHICLE = "stopped_vehicle"
    SUDDEN_CONGESTION = "sudden_congestion"
    WRONG_WAY = "wrong_way"
    PEDESTRIAN_INTRUSION = "pedestrian_intrusion"


@dataclass(frozen=True)
class AnomalyEvent:
    anomaly_type: AnomalyType

    confidence: float

    timestamp: float

    junction_id: Optional[str] = None
    movement_id: Optional[str] = None
    lane_id: Optional[str] = None

    severity: float = 0.0

    description: str = ""

    def __post_init__(self):
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                "confidence must be between 0 and 1."
            )

        if not 0.0 <= self.severity <= 1.0:
            raise ValueError(
                "severity must be between 0 and 1."
            )
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class EnvironmentalAnomalyType(str, Enum):
    """
    Environmental and road-condition anomalies.
    """

    FLOODING = "flooding"
    ROAD_BLOCKAGE = "road_blockage"
    ACCIDENT = "accident"
    FIRE = "fire"
    SMOKE = "smoke"


@dataclass(frozen=True)
class EnvironmentalObservation:
    """
    Observation produced by the perception layer.

    This is deliberately independent of any fixed junction.
    """

    timestamp: float
    camera_id: str

    anomaly_type: EnvironmentalAnomalyType

    confidence: float

    x: Optional[float] = None
    y: Optional[float] = None

    severity: float = 0.0

    affected_area: float = 0.0


@dataclass(frozen=True)
class EnvironmentalAnomaly:
    """
    Confirmed environmental/road anomaly.
    """

    anomaly_type: EnvironmentalAnomalyType

    camera_id: str

    first_seen: float
    last_seen: float

    confidence: float
    severity: float

    affected_area: float

    active: bool = True
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class IncidentType(str, Enum):
    """
    Unified incident categories.
    """

    ACCIDENT = "accident"
    FLOODING = "flooding"
    ROAD_BLOCKAGE = "road_blockage"
    FIRE = "fire"
    SMOKE = "smoke"

    STOPPED_VEHICLE = "stopped_vehicle"
    WRONG_WAY = "wrong_way"
    SUDDEN_BRAKING = "sudden_braking"
    LANE_CHANGE = "lane_change"


class IncidentStatus(str, Enum):
    ACTIVE = "active"
    RESOLVED = "resolved"


@dataclass(frozen=True)
class Incident:
    """
    Unified traffic incident.

    The incident is independent of a fixed J1/J2/J3/J4
    configuration.
    """

    incident_id: str

    incident_type: IncidentType

    camera_id: Optional[str]

    junction_id: Optional[str]

    timestamp: float

    confidence: float

    severity: float

    affected_area: float = 0.0

    status: IncidentStatus = IncidentStatus.ACTIVE
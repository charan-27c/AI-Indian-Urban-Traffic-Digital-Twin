from dataclasses import dataclass, field
from typing import Dict


# ============================================================
# VEHICLE STATE
# ============================================================

@dataclass
class VehicleState:

    vehicle_id: str
    vehicle_class: str

    lane_id: str
    edge_id: str

    speed: float
    waiting_time: float

    stopped: bool = False


# ============================================================
# MOVEMENT TRAFFIC STATE
# ============================================================

@dataclass
class MovementTrafficState:

    movement_id: str

    junction_id: str

    vehicle_count: int = 0
    queue_length: int = 0

    average_speed: float = 0.0

    waiting_time: float = 0.0
    delay: float = 0.0

    pressure: float = 0.0
    available: bool = True
    blocked: bool = False
    effective_pressure: float = 0.0

    lanes: set[str] = field(
        default_factory=set
    )


# ============================================================
# LANE TRAFFIC STATE
# ============================================================

@dataclass
class LaneTrafficState:

    lane_id: str
    edge_id: str = ""

    vehicle_count: int = 0
    queue_length: int = 0

    average_speed: float = 0.0

    waiting_time: float = 0.0
    delay: float = 0.0

    pressure: float = 0.0

    available: bool = True
    blocked: bool = False
    effective_pressure: float = 0.0


# ============================================================
# APPROACH TRAFFIC STATE
# ============================================================

@dataclass
class ApproachTrafficState:

    approach_id: str

    vehicle_count: int = 0
    queue_length: int = 0

    average_speed: float = 0.0

    density: float = 0.0
    occupancy: float = 0.0

    waiting_time: float = 0.0
    delay: float = 0.0

    flow_rate: float = 0.0
    pressure: float = 0.0

    lanes: set[
        str
    ] = field(
        default_factory=set
    )

    movements: set[
        str
    ] = field(
        default_factory=set
    )


# ============================================================
# JUNCTION TRAFFIC STATE
# ============================================================

@dataclass
class JunctionTrafficState:

    junction_id: str

    vehicle_count: int = 0
    queue_length: int = 0

    average_speed: float = 0.0

    density: float = 0.0
    occupancy: float = 0.0

    waiting_time: float = 0.0
    delay: float = 0.0

    flow_rate: float = 0.0
    pressure: float = 0.0

    lanes: Dict[
        str,
        LaneTrafficState
    ] = field(
        default_factory=dict
    )

    approaches: Dict[
        str,
        ApproachTrafficState
    ] = field(
        default_factory=dict
    )

    movements: Dict[
        str,
        MovementTrafficState
    ] = field(
        default_factory=dict
    )


# ============================================================
# NETWORK TRAFFIC STATE
# ============================================================

@dataclass
class NetworkTrafficState:

    timestamp: float = 0.0

    vehicle_count: int = 0
    queue_length: int = 0

    average_speed: float = 0.0

    density: float = 0.0
    occupancy: float = 0.0

    waiting_time: float = 0.0
    delay: float = 0.0

    flow_rate: float = 0.0
    pressure: float = 0.0

    junctions: Dict[
        str,
        JunctionTrafficState
    ] = field(
        default_factory=dict
    )
from dataclasses import dataclass
from typing import Dict, Any


@dataclass(frozen=True)
class TrafficHistoryRecord:
    """
    One timestamped snapshot of network traffic.

    This is intentionally topology-independent.
    No J1/J2/J3/J4 assumptions are made.
    """

    timestamp: float

    vehicle_count: int
    queue_length: int

    average_speed: float

    density: float
    occupancy: float

    waiting_time: float
    delay: float

    flow_rate: float
    pressure: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "vehicle_count": self.vehicle_count,
            "queue_length": self.queue_length,
            "average_speed": self.average_speed,
            "density": self.density,
            "occupancy": self.occupancy,
            "waiting_time": self.waiting_time,
            "delay": self.delay,
            "flow_rate": self.flow_rate,
            "pressure": self.pressure,
        }
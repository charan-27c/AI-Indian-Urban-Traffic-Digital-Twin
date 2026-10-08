from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class IncidentImpact:
    """
    Estimated impact of an incident on the dynamic
    traffic topology.
    """

    incident_id: str

    junction_id: Optional[str]

    affected_edges: tuple[str, ...]
    affected_lanes: tuple[str, ...]
    affected_movements: tuple[str, ...]

    capacity_reduction: float

    blocked: bool

    severity: float

    spillback_risk: float

    def is_significant(self) -> bool:
        return (
            self.blocked
            or self.capacity_reduction >= 0.50
            or self.spillback_risk >= 0.70
            or self.severity >= 0.70
        )
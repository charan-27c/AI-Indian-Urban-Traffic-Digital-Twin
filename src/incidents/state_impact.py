from dataclasses import dataclass
from typing import Iterable, Optional, Set

from .topology_impact import IncidentTopologyImpactManager


@dataclass(frozen=True)
class IncidentStateImpact:
    blocked_lanes: frozenset[str]
    blocked_movements: frozenset[str]

    capacity_reduction: float = 0.0
    spillback_risk: float = 0.0
    severity: float = 0.0

    def affects_lane(self, lane_id: str) -> bool:
        return lane_id in self.blocked_lanes

    def affects_movement(self, movement_id: str) -> bool:
        return movement_id in self.blocked_movements


class IncidentStateImpactResolver:
    """
    Converts the runtime incident topology overlay into
    state-engine-friendly availability information.
    """

    def __init__(
        self,
        topology_impact_manager: IncidentTopologyImpactManager,
    ) -> None:
        self.manager = topology_impact_manager

    def resolve(
        self,
        lane_ids: Iterable[str] = (),
        movement_ids: Iterable[str] = (),
    ) -> IncidentStateImpact:

        lane_set: Set[str] = {
            lane_id
            for lane_id in lane_ids
            if self.manager.is_lane_blocked(lane_id)
        }

        movement_set: Set[str] = {
            movement_id
            for movement_id in movement_ids
            if self.manager.is_movement_blocked(
                movement_id
            )
        }

        return IncidentStateImpact(
            blocked_lanes=frozenset(lane_set),
            blocked_movements=frozenset(movement_set),
        )

    def lane_available(
        self,
        lane_id: str,
    ) -> bool:
        return self.manager.is_lane_available(
            lane_id
        )

    def movement_available(
        self,
        movement_id: str,
    ) -> bool:
        return self.manager.is_movement_available(
            movement_id
        )
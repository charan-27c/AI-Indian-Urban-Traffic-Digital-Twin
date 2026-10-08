from typing import Iterable, List

from .topology_impact import IncidentTopologyImpactManager


class RuntimeAvailabilityFilter:
    """
    Filters dynamically discovered topology resources using
    the current incident overlay.

    The underlying topology is never modified.
    """

    def __init__(
        self,
        impact_manager: IncidentTopologyImpactManager,
    ) -> None:
        self.impact_manager = impact_manager

    def available_lanes(
        self,
        lane_ids: Iterable[str],
    ) -> List[str]:
        return [
            lane_id
            for lane_id in lane_ids
            if self.impact_manager.is_lane_available(
                lane_id
            )
        ]

    def available_movements(
        self,
        movement_ids: Iterable[str],
    ) -> List[str]:
        return [
            movement_id
            for movement_id in movement_ids
            if self.impact_manager.is_movement_available(
                movement_id
            )
        ]

    def blocked_lanes(
        self,
        lane_ids: Iterable[str],
    ) -> List[str]:
        return [
            lane_id
            for lane_id in lane_ids
            if self.impact_manager.is_lane_blocked(
                lane_id
            )
        ]

    def blocked_movements(
        self,
        movement_ids: Iterable[str],
    ) -> List[str]:
        return [
            movement_id
            for movement_id in movement_ids
            if self.impact_manager.is_movement_blocked(
                movement_id
            )
        ]
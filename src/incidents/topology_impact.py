from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Set


@dataclass(frozen=True)
class TopologyImpact:
    """
    Runtime availability impact caused by an incident.

    The underlying discovered topology remains unchanged.
    This object represents temporary operational restrictions.
    """

    incident_id: str

    blocked_lanes: tuple[str, ...]
    blocked_movements: tuple[str, ...]

    affected_junction_id: Optional[str]

    active: bool = True


class IncidentTopologyImpactManager:
    """
    Maintains temporary topology restrictions caused by incidents.

    IMPORTANT:
        This does not modify the discovered topology.

    Base topology:
        lane/movement exists

    Runtime overlay:
        lane/movement may currently be unavailable
    """

    def __init__(self) -> None:

        self._impacts: Dict[
            str,
            TopologyImpact,
        ] = {}

        self._blocked_lanes: Set[str] = set()
        self._blocked_movements: Set[str] = set()

    # ========================================================
    # APPLY INCIDENT
    # ========================================================

    def apply(
        self,
        incident_id: str,
        blocked_lanes: Iterable[str] = (),
        blocked_movements: Iterable[str] = (),
        junction_id: Optional[str] = None,
    ) -> TopologyImpact:

        lanes = tuple(
            dict.fromkeys(
                blocked_lanes
            )
        )

        movements = tuple(
            dict.fromkeys(
                blocked_movements
            )
        )

        impact = TopologyImpact(
            incident_id=incident_id,
            blocked_lanes=lanes,
            blocked_movements=movements,
            affected_junction_id=junction_id,
            active=True,
        )

        # If the incident already existed, first remove
        # its previous contribution.
        self.remove(incident_id)

        self._impacts[
            incident_id
        ] = impact

        self._blocked_lanes.update(
            lanes
        )

        self._blocked_movements.update(
            movements
        )

        return impact

    # ========================================================
    # REMOVE INCIDENT
    # ========================================================

    def remove(
        self,
        incident_id: str,
    ) -> Optional[TopologyImpact]:

        previous = self._impacts.pop(
            incident_id,
            None,
        )

        if previous is None:
            return None

        # Rebuild the aggregate sets rather than blindly
        # removing resources. This is important when two
        # incidents affect the same lane/movement.
        self._rebuild_indexes()

        return previous

    # ========================================================
    # REBUILD
    # ========================================================

    def _rebuild_indexes(self) -> None:

        self._blocked_lanes.clear()
        self._blocked_movements.clear()

        for impact in self._impacts.values():

            if not impact.active:
                continue

            self._blocked_lanes.update(
                impact.blocked_lanes
            )

            self._blocked_movements.update(
                impact.blocked_movements
            )

    # ========================================================
    # LANE AVAILABILITY
    # ========================================================

    def is_lane_blocked(
        self,
        lane_id: str,
    ) -> bool:

        return lane_id in self._blocked_lanes

    def is_lane_available(
        self,
        lane_id: str,
    ) -> bool:

        return not self.is_lane_blocked(
            lane_id
        )

    # ========================================================
    # MOVEMENT AVAILABILITY
    # ========================================================

    def is_movement_blocked(
        self,
        movement_id: str,
    ) -> bool:

        return (
            movement_id
            in self._blocked_movements
        )

    def is_movement_available(
        self,
        movement_id: str,
    ) -> bool:

        return not self.is_movement_blocked(
            movement_id
        )

    # ========================================================
    # GET BLOCKED RESOURCES
    # ========================================================

    def blocked_lanes(self) -> Set[str]:

        return set(
            self._blocked_lanes
        )

    def blocked_movements(self) -> Set[str]:

        return set(
            self._blocked_movements
        )

    # ========================================================
    # ACTIVE IMPACTS
    # ========================================================

    def active_impacts(self):

        return [
            impact
            for impact in self._impacts.values()
            if impact.active
        ]

    # ========================================================
    # RESET
    # ========================================================

    def reset(self) -> None:

        self._impacts.clear()
        self._blocked_lanes.clear()
        self._blocked_movements.clear()
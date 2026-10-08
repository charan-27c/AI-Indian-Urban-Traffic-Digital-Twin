from typing import Iterable

from .incident import (
    Incident,
    IncidentType,
)

from .impact import IncidentImpact


class IncidentImpactAnalyzer:
    """
    Converts incidents into traffic-network impacts.

    The analyzer does NOT assume:
        J1/J2/J3/J4
        four approaches
        fixed lanes
        fixed movements

    Topology information is supplied at runtime.
    """

    def __init__(
        self,
        topology=None,
    ) -> None:

        self.topology = topology

    # ========================================================
    # ANALYZE
    # ========================================================

    def analyze(
        self,
        incident: Incident,
        affected_edges: Iterable[str] = (),
        affected_lanes: Iterable[str] = (),
        affected_movements: Iterable[str] = (),
    ) -> IncidentImpact:

        edges = tuple(
            affected_edges
        )

        lanes = tuple(
            affected_lanes
        )

        movements = tuple(
            affected_movements
        )

        capacity_reduction = (
            self._estimate_capacity_reduction(
                incident
            )
        )

        blocked = (
            incident.incident_type
            in {
                IncidentType.ACCIDENT,
                IncidentType.ROAD_BLOCKAGE,
                IncidentType.FLOODING,
            }
        )

        spillback_risk = (
            self._estimate_spillback_risk(
                incident
            )
        )

        return IncidentImpact(
            incident_id=incident.incident_id,

            junction_id=incident.junction_id,

            affected_edges=edges,

            affected_lanes=lanes,

            affected_movements=movements,

            capacity_reduction=capacity_reduction,

            blocked=blocked,

            severity=incident.severity,

            spillback_risk=spillback_risk,
        )

    # ========================================================
    # CAPACITY
    # ========================================================

    def _estimate_capacity_reduction(
        self,
        incident: Incident,
    ) -> float:

        if incident.incident_type == (
            IncidentType.ACCIDENT
        ):
            return min(
                1.0,
                0.60 + (
                    0.40
                    * incident.severity
                ),
            )

        if incident.incident_type == (
            IncidentType.ROAD_BLOCKAGE
        ):
            return min(
                1.0,
                0.70 + (
                    0.30
                    * incident.severity
                ),
            )

        if incident.incident_type == (
            IncidentType.FLOODING
        ):
            return min(
                1.0,
                0.50 + (
                    0.50
                    * incident.severity
                ),
            )

        if incident.incident_type == (
            IncidentType.STOPPED_VEHICLE
        ):
            return min(
                1.0,
                0.20 + (
                    0.30
                    * incident.severity
                ),
            )

        if incident.incident_type == (
            IncidentType.SUDDEN_BRAKING
        ):
            return min(
                1.0,
                0.10 + (
                    0.20
                    * incident.severity
                ),
            )

        return min(
            1.0,
            0.10 * incident.severity,
        )

    # ========================================================
    # SPILLBACK
    # ========================================================

    def _estimate_spillback_risk(
        self,
        incident: Incident,
    ) -> float:

        if incident.incident_type == (
            IncidentType.ROAD_BLOCKAGE
        ):
            return min(
                1.0,
                0.60
                + 0.40 * incident.severity,
            )

        if incident.incident_type == (
            IncidentType.ACCIDENT
        ):
            return min(
                1.0,
                0.50
                + 0.50 * incident.severity,
            )

        if incident.incident_type == (
            IncidentType.FLOODING
        ):
            return min(
                1.0,
                0.40
                + 0.60 * incident.severity,
            )

        return min(
            1.0,
            0.20 * incident.severity,
        )
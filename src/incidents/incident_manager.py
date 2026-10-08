from typing import Dict, Optional

from .incident import (
    Incident,
    IncidentStatus,
    IncidentType,
)


class IncidentManager:
    """
    Central incident registry.

    Responsibilities:

    1. Register incidents.
    2. Deduplicate repeated detections.
    3. Keep different incidents separate.
    4. Resolve incidents.
    5. Provide active incidents to the traffic system.

    No fixed junction or movement is assumed.
    """

    def __init__(self) -> None:

        self._incidents: Dict[
            str,
            Incident,
        ] = {}

        self._counter = 0

    # ========================================================
    # INCIDENT ID
    # ========================================================

    def _generate_id(self) -> str:

        self._counter += 1

        return f"INC-{self._counter:06d}"

    # ========================================================
    # REGISTER
    # ========================================================

    def register(
        self,
        incident_type: IncidentType,
        timestamp: float,
        confidence: float,
        severity: float,
        camera_id: Optional[str] = None,
        junction_id: Optional[str] = None,
        affected_area: float = 0.0,
    ) -> Incident:

        # ----------------------------------------------------
        # Deduplicate active incident
        # ----------------------------------------------------

        for incident in self._incidents.values():

            if (
                incident.status
                == IncidentStatus.ACTIVE
                and incident.incident_type
                == incident_type
                and incident.camera_id
                == camera_id
                and incident.junction_id
                == junction_id
            ):

                updated = Incident(
                    incident_id=incident.incident_id,

                    incident_type=incident.incident_type,

                    camera_id=incident.camera_id,

                    junction_id=incident.junction_id,

                    timestamp=timestamp,

                    confidence=max(
                        incident.confidence,
                        confidence,
                    ),

                    severity=max(
                        incident.severity,
                        severity,
                    ),

                    affected_area=max(
                        incident.affected_area,
                        affected_area,
                    ),

                    status=IncidentStatus.ACTIVE,
                )

                self._incidents[
                    incident.incident_id
                ] = updated

                return updated

        # ----------------------------------------------------
        # New incident
        # ----------------------------------------------------

        incident = Incident(
            incident_id=self._generate_id(),

            incident_type=incident_type,

            camera_id=camera_id,

            junction_id=junction_id,

            timestamp=timestamp,

            confidence=confidence,

            severity=severity,

            affected_area=affected_area,

            status=IncidentStatus.ACTIVE,
        )

        self._incidents[
            incident.incident_id
        ] = incident

        return incident

    # ========================================================
    # RESOLVE
    # ========================================================

    def resolve(
        self,
        incident_id: str,
        timestamp: float,
    ) -> Optional[Incident]:

        incident = self._incidents.get(
            incident_id
        )

        if incident is None:
            return None

        resolved = Incident(
            incident_id=incident.incident_id,

            incident_type=incident.incident_type,

            camera_id=incident.camera_id,

            junction_id=incident.junction_id,

            timestamp=timestamp,

            confidence=incident.confidence,

            severity=incident.severity,

            affected_area=incident.affected_area,

            status=IncidentStatus.RESOLVED,
        )

        self._incidents[
            incident_id
        ] = resolved

        return resolved

    # ========================================================
    # ACTIVE INCIDENTS
    # ========================================================

    def active_incidents(self):

        return [
            incident
            for incident in self._incidents.values()
            if incident.status
            == IncidentStatus.ACTIVE
        ]

    # ========================================================
    # ALL INCIDENTS
    # ========================================================

    def all_incidents(self):

        return list(
            self._incidents.values()
        )

    # ========================================================
    # GET
    # ========================================================

    def get(
        self,
        incident_id: str,
    ) -> Optional[Incident]:

        return self._incidents.get(
            incident_id
        )

    # ========================================================
    # RESET
    # ========================================================

    def reset(self) -> None:

        self._incidents.clear()

        self._counter = 0
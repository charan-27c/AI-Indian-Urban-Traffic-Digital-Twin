from __future__ import annotations

from dataclasses import replace
from typing import Iterable, Mapping, Optional

from .incident_aware_optimizer import MovementCandidate


class IncidentPredictionFusion:
    """
    Stage 3C

    Combines:
        - current movement traffic
        - prediction-adjusted pressure
        - incident impact
        - runtime lane/movement availability

    This module does NOT control SUMO signals.

    It only produces safe MovementCandidate objects.

    Safety:
        blocked movement -> blocked=True
        unavailable movement -> available=False

    The downstream IncidentAwareSignalOptimizer remains
    the final movement-selection layer.
    """

    def __init__(
        self,
        incident_pressure_weight: float = 1.0,
        severity_weight: float = 20.0,
        capacity_weight: float = 1.0,
        spillback_weight: float = 0.5,
    ) -> None:

        self.incident_pressure_weight = float(
            incident_pressure_weight
        )

        self.severity_weight = float(
            severity_weight
        )

        self.capacity_weight = float(
            capacity_weight
        )

        self.spillback_weight = float(
            spillback_weight
        )

    # ============================================================
    # INCIDENT PRESSURE
    # ============================================================

    def incident_pressure_adjustment(
        self,
        pressure: float,
        *,
        capacity_reduction: float = 0.0,
        spillback_risk: float = 0.0,
        severity: float = 0.0,
        blocked: bool = False,
    ) -> float:
        """
        Calculate incident-adjusted pressure.

        A blocked movement receives a large internal pressure
        value for diagnostic purposes, but it is ALSO marked
        blocked.

        Therefore pressure can never override safety filtering.
        """

        pressure = max(
            0.0,
            float(pressure),
        )

        capacity_reduction = max(
            0.0,
            min(
                1.0,
                float(capacity_reduction),
            ),
        )

        spillback_risk = max(
            0.0,
            min(
                1.0,
                float(spillback_risk),
            ),
        )

        severity = max(
            0.0,
            min(
                1.0,
                float(severity),
            ),
        )

        # --------------------------------------------------------
        # BLOCKED MOVEMENT
        # --------------------------------------------------------

        if blocked:
            return (
                pressure * 2.0
                + 100.0
            )

        # --------------------------------------------------------
        # NON-BLOCKING INCIDENT
        # --------------------------------------------------------

        adjusted = pressure

        adjusted *= (
            1.0
            + self.capacity_weight
            * capacity_reduction
        )

        adjusted *= (
            1.0
            + self.spillback_weight
            * spillback_risk
        )

        adjusted += (
            self.severity_weight
            * severity
        )

        return max(
            0.0,
            adjusted,
        )

    # ============================================================
    # IMPACT VALUE HELPER
    # ============================================================

    @staticmethod
    def _impact_value(
        impact,
        name: str,
        default=0.0,
    ):
        """
        Read an attribute safely from an incident impact object.
        """

        if impact is None:
            return default

        value = getattr(
            impact,
            name,
            default,
        )

        return value

    # ============================================================
    # SINGLE MOVEMENT
    # ============================================================

    def fuse_candidate(
        self,
        candidate: MovementCandidate,
        *,
        impact=None,
        available: Optional[bool] = None,
        blocked: Optional[bool] = None,
    ) -> MovementCandidate:
        """
        Fuse incident information into an existing candidate.

        Prediction has already been incorporated into
        candidate.effective_pressure by the prediction adapter.

        Therefore this method adjusts effective_pressure instead
        of replacing the prediction signal.

        Safety precedence:

            candidate.blocked
                    OR
            incident.blocked
                    OR
            runtime blocked
                    ↓
            final blocked

        A caller can never accidentally unblock an already
        blocked candidate by passing blocked=False.
        """

        # --------------------------------------------------------
        # INCIDENT BLOCK STATUS
        # --------------------------------------------------------

        impact_blocked = bool(
            self._impact_value(
                impact,
                "blocked",
                False,
            )
        )

        impact_available = not impact_blocked

        # --------------------------------------------------------
        # EXISTING CANDIDATE SAFETY
        # --------------------------------------------------------

        candidate_blocked = bool(
            candidate.blocked
        )

        # --------------------------------------------------------
        # FINAL BLOCK STATUS
        # --------------------------------------------------------

        explicit_blocked = (
            False
            if blocked is None
            else bool(blocked)
        )

        final_blocked = (
            candidate_blocked
            or impact_blocked
            or explicit_blocked
        )

        # --------------------------------------------------------
        # FINAL AVAILABILITY
        # --------------------------------------------------------

        if available is None:
            final_available = impact_available
        else:
            final_available = bool(
                available
            ) and impact_available

        # Existing candidate blocked state always wins.
        if candidate_blocked:
            final_available = False

        # Any final blocked state means unavailable.
        if final_blocked:
            final_available = False

        # --------------------------------------------------------
        # INCIDENT PARAMETERS
        # --------------------------------------------------------

        capacity_reduction = float(
            self._impact_value(
                impact,
                "capacity_reduction",
                0.0,
            )
        )

        spillback_risk = float(
            self._impact_value(
                impact,
                "spillback_risk",
                0.0,
            )
        )

        severity = float(
            self._impact_value(
                impact,
                "severity",
                0.0,
            )
        )

        # --------------------------------------------------------
        # BASE PRESSURE
        # --------------------------------------------------------

        if candidate.effective_pressure > 0.0:
            base_pressure = (
                candidate.effective_pressure
            )
        else:
            base_pressure = (
                candidate.pressure
            )

        # --------------------------------------------------------
        # INCIDENT-ADJUSTED PRESSURE
        # --------------------------------------------------------

        adjusted_pressure = (
            self.incident_pressure_adjustment(
                base_pressure,
                capacity_reduction=capacity_reduction,
                spillback_risk=spillback_risk,
                severity=severity,
                blocked=final_blocked,
            )
        )

        # --------------------------------------------------------
        # RETURN NEW IMMUTABLE CANDIDATE
        # --------------------------------------------------------

        return replace(
            candidate,
            effective_pressure=adjusted_pressure,
            available=final_available,
            blocked=final_blocked,
        )

    # ============================================================
    # MULTIPLE MOVEMENTS
    # ============================================================

    def fuse_candidates(
        self,
        candidates: Iterable[MovementCandidate],
        *,
        impacts: Optional[
            Mapping[str, object]
        ] = None,
        availability: Optional[
            Mapping[str, bool]
        ] = None,
        blocked: Optional[
            Mapping[str, bool]
        ] = None,
    ) -> list[MovementCandidate]:
        """
        Fuse incident information for all movements.

        All mappings are keyed by movement_id.
        """

        impacts = impacts or {}
        availability = availability or {}
        blocked = blocked or {}

        result: list[
            MovementCandidate
        ] = []

        for candidate in candidates:

            movement_id = (
                candidate.movement_id
            )

            fused = self.fuse_candidate(
                candidate,
                impact=impacts.get(
                    movement_id
                ),
                available=availability.get(
                    movement_id
                ),
                blocked=blocked.get(
                    movement_id
                ),
            )

            result.append(
                fused
            )

        return result

    # ============================================================
    # DIRECT OPTIMIZER PIPELINE
    # ============================================================

    def select(
        self,
        optimizer,
        candidates: Iterable[MovementCandidate],
        *,
        impacts: Optional[
            Mapping[str, object]
        ] = None,
        availability: Optional[
            Mapping[str, bool]
        ] = None,
        blocked: Optional[
            Mapping[str, bool]
        ] = None,
    ):
        """
        Fuse incidents and invoke the existing
        safety-aware optimizer.
        """

        fused_candidates = (
            self.fuse_candidates(
                candidates,
                impacts=impacts,
                availability=availability,
                blocked=blocked,
            )
        )

        return optimizer.select_movement(
            fused_candidates
        )
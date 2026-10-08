from dataclasses import dataclass
from typing import Iterable, Optional


@dataclass(frozen=True)
class MovementCandidate:
    """
    Runtime information about one traffic movement.

    This class is intentionally independent of SUMO.
    The TrafficStateEngine can populate it from
    MovementTrafficState.
    """

    movement_id: str

    pressure: float = 0.0

    effective_pressure: float = 0.0

    vehicle_count: int = 0

    queue_length: int = 0

    waiting_time: float = 0.0

    available: bool = True

    blocked: bool = False

    emergency: bool = False

    starvation_wait: float = 0.0


@dataclass(frozen=True)
class SignalDecision:
    """
    Decision produced by the signal optimizer.
    """

    movement_id: Optional[str]

    score: float

    reason: str

    safe: bool


class IncidentAwareSignalOptimizer:
    """
    Incident-aware movement-level signal optimizer.

    Safety hierarchy:

        1. Blocked movement rejection
        2. Available movement filtering
        3. Emergency priority
        4. Starvation protection
        5. Effective pressure
        6. Normal pressure

    The optimizer never selects a blocked movement.

    It does not directly control SUMO signals.
    It only decides WHICH movement should receive service.

    A separate signal executor/controller can apply
    the decision while enforcing yellow/all-red/min-green
    safety transitions.
    """

    def __init__(
        self,
        emergency_bonus: float = 1000.0,
        starvation_weight: float = 2.0,
        pressure_weight: float = 1.0,
        queue_weight: float = 1.0,
        waiting_weight: float = 0.5,
    ) -> None:

        self.emergency_bonus = emergency_bonus

        self.starvation_weight = starvation_weight

        self.pressure_weight = pressure_weight

        self.queue_weight = queue_weight

        self.waiting_weight = waiting_weight

    # ============================================================
    # SAFETY
    # ============================================================

    @staticmethod
    def is_safe_candidate(
        movement: MovementCandidate,
    ) -> bool:
        """
        A blocked movement is NEVER safe.

        Both flags are checked because different
        pipeline layers may populate either field.
        """

        if movement.blocked:
            return False

        if not movement.available:
            return False

        return True

    # ============================================================
    # SCORE
    # ============================================================

    def calculate_score(
        self,
        movement: MovementCandidate,
    ) -> float:
        """
        Calculate optimization score.

        effective_pressure is preferred because it can
        already include incident impact.

        Normal pressure is retained as fallback.
        """

        if not self.is_safe_candidate(movement):
            return float("-inf")

        pressure = movement.effective_pressure

        if pressure <= 0.0:
            pressure = movement.pressure

        score = 0.0

        # --------------------------------------------------------
        # Traffic pressure
        # --------------------------------------------------------

        score += (
            pressure
            * self.pressure_weight
        )

        # --------------------------------------------------------
        # Queue
        # --------------------------------------------------------

        score += (
            movement.queue_length
            * self.queue_weight
        )

        # --------------------------------------------------------
        # Waiting time
        # --------------------------------------------------------

        score += (
            movement.waiting_time
            * self.waiting_weight
        )

        # --------------------------------------------------------
        # Starvation protection
        # --------------------------------------------------------

        score += (
            movement.starvation_wait
            * self.starvation_weight
        )

        # --------------------------------------------------------
        # Emergency priority
        # --------------------------------------------------------

        if movement.emergency:
            score += self.emergency_bonus

        return score

    # ============================================================
    # AVAILABLE MOVEMENTS
    # ============================================================

    def get_available_movements(
        self,
        movements: Iterable[MovementCandidate],
    ) -> list[MovementCandidate]:
        """
        Return only movements that are safe to serve.
        """

        return [
            movement
            for movement in movements
            if self.is_safe_candidate(movement)
        ]

    # ============================================================
    # BEST MOVEMENT
    # ============================================================

    def select_movement(
        self,
        movements: Iterable[MovementCandidate],
    ) -> SignalDecision:
        """
        Select the safest highest-priority movement.
        """

        movement_list = list(movements)

        if not movement_list:
            return SignalDecision(
                movement_id=None,
                score=0.0,
                reason="no_movements",
                safe=False,
            )

        # --------------------------------------------------------
        # HARD SAFETY FILTER
        # --------------------------------------------------------

        available = self.get_available_movements(
            movement_list
        )

        if not available:
            return SignalDecision(
                movement_id=None,
                score=0.0,
                reason="all_movements_blocked_or_unavailable",
                safe=False,
            )

        # --------------------------------------------------------
        # SCORE
        # --------------------------------------------------------

        scored = []

        for movement in available:

            score = self.calculate_score(
                movement
            )

            scored.append(
                (
                    score,
                    movement,
                )
            )

        # --------------------------------------------------------
        # BEST SAFE MOVEMENT
        # --------------------------------------------------------

        best_score, best_movement = max(
            scored,
            key=lambda item: (
                item[0],
                item[1].movement_id,
            ),
        )

        # --------------------------------------------------------
        # REASON
        # --------------------------------------------------------

        if best_movement.emergency:
            reason = "emergency_priority"

        elif best_movement.starvation_wait > 0.0:
            reason = "starvation_protection"

        elif (
            best_movement.effective_pressure
            > best_movement.pressure
        ):
            reason = "incident_adjusted_pressure"

        else:
            reason = "highest_safe_pressure"

        return SignalDecision(
            movement_id=best_movement.movement_id,
            score=best_score,
            reason=reason,
            safe=True,
        )

    # ============================================================
    # CURRENT MOVEMENT SAFETY
    # ============================================================

    def should_switch_from_current(
        self,
        current_movement: Optional[MovementCandidate],
        candidates: Iterable[MovementCandidate],
        minimum_green_completed: bool,
        switch_threshold: float = 10.0,
    ) -> bool:
        """
        Decide whether the current movement should be replaced.

        Minimum-green safety ALWAYS wins.

        A blocked current movement may be replaced immediately
        by the external safety controller if its transition
        rules permit it.
        """

        if current_movement is None:
            return True

        # --------------------------------------------------------
        # BLOCKED CURRENT MOVEMENT
        # --------------------------------------------------------

        if not self.is_safe_candidate(
            current_movement
        ):
            return True

        # --------------------------------------------------------
        # MINIMUM GREEN SAFETY
        # --------------------------------------------------------

        if not minimum_green_completed:
            return False

        decision = self.select_movement(
            candidates
        )

        if not decision.safe:
            return False

        if decision.movement_id == (
            current_movement.movement_id
        ):
            return False

        current_score = self.calculate_score(
            current_movement
        )

        score_difference = (
            decision.score
            - current_score
        )

        return (
            score_difference
            >= switch_threshold
        )
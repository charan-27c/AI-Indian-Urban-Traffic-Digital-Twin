from dataclasses import dataclass
from enum import Enum
from typing import Optional


class SignalState(str, Enum):
    RED = "red"
    GREEN = "green"
    YELLOW = "yellow"
    ALL_RED = "all_red"


@dataclass(frozen=True)
class SignalTransition:
    junction_id: Optional[str]
    movement_id: Optional[str]
    state: SignalState
    timestamp: float
    duration: float
    reason: str


class SafeSignalExecutor:
    """
    Safety-first signal state machine.

    State sequence:

        ALL_RED
           ↓
         GREEN
           ↓
        YELLOW
           ↓
        ALL_RED
           ↓
         GREEN

    Safety rules:

    1. Initial ALL_RED clearance is mandatory.
    2. Only one junction can be active.
    3. Only one movement can be GREEN.
    4. Minimum green time cannot be violated.
    5. GREEN -> YELLOW -> ALL_RED is mandatory.
    6. Yellow clearance must complete.
    7. All-red clearance must complete.
    8. Blocked movements cannot become GREEN.
    """

    def __init__(
        self,
        min_green_time: float = 20.0,
        max_green_time: float = 90.0,
        yellow_time: float = 3.0,
        all_red_time: float = 3.0,
    ):
        if min_green_time <= 0:
            raise ValueError(
                "min_green_time must be > 0"
            )

        if max_green_time < min_green_time:
            raise ValueError(
                "max_green_time must be >= min_green_time"
            )

        if yellow_time < 0:
            raise ValueError(
                "yellow_time must be >= 0"
            )

        if all_red_time < 0:
            raise ValueError(
                "all_red_time must be >= 0"
            )

        self.min_green_time = min_green_time
        self.max_green_time = max_green_time
        self.yellow_time = yellow_time
        self.all_red_time = all_red_time

        self.active_junction_id: Optional[str] = None
        self.active_movement_id: Optional[str] = None

        self.state = SignalState.ALL_RED
        self.state_started_at = 0.0

        self._transitions: list[SignalTransition] = []

    # ================================================================
    # STATE INFORMATION
    # ================================================================

    def elapsed(self, now: float) -> float:
        return max(
            0.0,
            now - self.state_started_at,
        )

    def green_elapsed(self, now: float) -> float:
        if self.state != SignalState.GREEN:
            return 0.0

        return self.elapsed(now)

    def is_green(self) -> bool:
        return self.state == SignalState.GREEN

    def is_all_red(self) -> bool:
        return self.state == SignalState.ALL_RED

    def current_target(
        self,
    ) -> tuple[Optional[str], Optional[str]]:
        return (
            self.active_junction_id,
            self.active_movement_id,
        )

    # ================================================================
    # GREEN TIMING
    # ================================================================

    def minimum_green_completed(
        self,
        now: float,
    ) -> bool:
        return (
            self.state == SignalState.GREEN
            and self.green_elapsed(now)
            >= self.min_green_time
        )

    def maximum_green_reached(
        self,
        now: float,
    ) -> bool:
        return (
            self.state == SignalState.GREEN
            and self.green_elapsed(now)
            >= self.max_green_time
        )

    def can_switch(
        self,
        now: float,
    ) -> bool:
        if self.state != SignalState.GREEN:
            return False

        return self.minimum_green_completed(now)

    # ================================================================
    # MOVEMENT SAFETY
    # ================================================================

    @staticmethod
    def can_activate_movement(
        junction_id: str,
        movement_id: str,
        *,
        movement_available: bool = True,
        movement_blocked: bool = False,
    ) -> bool:

        if not junction_id:
            return False

        if not movement_id:
            return False

        if not movement_available:
            return False

        if movement_blocked:
            return False

        return True

    # ================================================================
    # TRANSITION LOGGING
    # ================================================================

    def _record_transition(
        self,
        *,
        junction_id: Optional[str],
        movement_id: Optional[str],
        state: SignalState,
        timestamp: float,
        duration: float,
        reason: str,
    ) -> SignalTransition:

        transition = SignalTransition(
            junction_id=junction_id,
            movement_id=movement_id,
            state=state,
            timestamp=timestamp,
            duration=duration,
            reason=reason,
        )

        self._transitions.append(transition)

        return transition

    def transitions(
        self,
    ) -> tuple[SignalTransition, ...]:
        return tuple(self._transitions)

    # ================================================================
    # GREEN
    # ================================================================

    def activate_green(
        self,
        junction_id: str,
        movement_id: str,
        now: float,
        *,
        movement_available: bool = True,
        movement_blocked: bool = False,
        reason: str = "optimizer_selection",
    ) -> SignalTransition:

        if not self.can_activate_movement(
            junction_id,
            movement_id,
            movement_available=movement_available,
            movement_blocked=movement_blocked,
        ):
            raise ValueError(
                f"Unsafe movement activation: "
                f"junction={junction_id}, "
                f"movement={movement_id}"
            )

        if self.state != SignalState.ALL_RED:
            raise RuntimeError(
                "GREEN activation requires ALL_RED state"
            )

        self.active_junction_id = junction_id
        self.active_movement_id = movement_id

        self.state = SignalState.GREEN
        self.state_started_at = now

        return self._record_transition(
            junction_id=junction_id,
            movement_id=movement_id,
            state=SignalState.GREEN,
            timestamp=now,
            duration=self.max_green_time,
            reason=reason,
        )

    # ================================================================
    # YELLOW
    # ================================================================

    def enter_yellow(
        self,
        now: float,
        reason: str = "movement_switch",
    ) -> SignalTransition:

        if self.state != SignalState.GREEN:
            raise RuntimeError(
                "YELLOW transition requires GREEN state"
            )

        if not self.minimum_green_completed(now):
            raise RuntimeError(
                "Cannot enter YELLOW before minimum green time"
            )

        transition = self._record_transition(
            junction_id=self.active_junction_id,
            movement_id=self.active_movement_id,
            state=SignalState.YELLOW,
            timestamp=now,
            duration=self.yellow_time,
            reason=reason,
        )

        self.state = SignalState.YELLOW
        self.state_started_at = now

        return transition

    # ================================================================
    # ALL RED
    # ================================================================

    def enter_all_red(
        self,
        now: float,
        reason: str = "junction_handover",
    ) -> SignalTransition:

        transition = self._record_transition(
            junction_id=None,
            movement_id=None,
            state=SignalState.ALL_RED,
            timestamp=now,
            duration=self.all_red_time,
            reason=reason,
        )

        self.state = SignalState.ALL_RED
        self.state_started_at = now

        self.active_junction_id = None
        self.active_movement_id = None

        return transition

    def all_red_completed(
        self,
        now: float,
    ) -> bool:

        if self.state != SignalState.ALL_RED:
            return False

        return (
            self.elapsed(now)
            >= self.all_red_time
        )

    # ================================================================
    # YELLOW COMPLETION
    # ================================================================

    def complete_yellow(
        self,
        now: float,
        reason: str = "yellow_complete",
    ) -> SignalTransition:

        if self.state != SignalState.YELLOW:
            raise RuntimeError(
                "Yellow completion requires YELLOW state"
            )

        if self.elapsed(now) < self.yellow_time:
            raise RuntimeError(
                "Yellow clearance time has not completed"
            )

        return self.enter_all_red(
            now=now,
            reason=reason,
        )

    # ================================================================
    # MAIN REQUEST
    # ================================================================

    def request_switch(
        self,
        junction_id: str,
        movement_id: str,
        now: float,
        *,
        movement_available: bool = True,
        movement_blocked: bool = False,
        reason: str = "optimizer_selection",
    ) -> Optional[SignalTransition]:

        if not self.can_activate_movement(
            junction_id,
            movement_id,
            movement_available=movement_available,
            movement_blocked=movement_blocked,
        ):
            return None

        # ------------------------------------------------------------
        # ALL RED -> GREEN
        # ------------------------------------------------------------

        if self.state == SignalState.ALL_RED:

            if not self.all_red_completed(now):
                return None

            return self.activate_green(
                junction_id,
                movement_id,
                now,
                movement_available=movement_available,
                movement_blocked=movement_blocked,
                reason=reason,
            )

        # ------------------------------------------------------------
        # SAME MOVEMENT
        # ------------------------------------------------------------

        if (
            self.state == SignalState.GREEN
            and self.active_junction_id == junction_id
            and self.active_movement_id == movement_id
        ):
            return None

        # ------------------------------------------------------------
        # GREEN -> YELLOW
        #
        # Important:
        # The state transition happens, but request_switch()
        # returns None for this intermediate step.
        # ------------------------------------------------------------

        if self.state == SignalState.GREEN:

            if not self.minimum_green_completed(now):
                return None

            self.enter_yellow(
                now=now,
                reason=reason,
            )

            return None

        # ------------------------------------------------------------
        # YELLOW -> ALL RED
        # ------------------------------------------------------------

        if self.state == SignalState.YELLOW:

            if self.elapsed(now) < self.yellow_time:
                return None

            self.enter_all_red(
                now=now,
                reason="yellow_clearance_complete",
            )

            return None

        return None

    # ================================================================
    # RESET
    # ================================================================

    def reset(self) -> None:

        self.active_junction_id = None
        self.active_movement_id = None

        self.state = SignalState.ALL_RED
        self.state_started_at = 0.0

        self._transitions.clear()
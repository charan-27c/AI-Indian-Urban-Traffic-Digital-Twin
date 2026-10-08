from dataclasses import dataclass
from typing import Iterable, Optional

from .incident_aware_optimizer import (
    IncidentAwareSignalOptimizer,
    MovementCandidate,
    SignalDecision,
)
from .signal_executor import (
    SafeSignalExecutor,
    SignalState,
)
from .sumo_movement_mapper import (
    SUMOMovementMapper,
)
from .sumo_signal_adapter import (
    SUMOSignalAdapter,
)


@dataclass(frozen=True)
class ControlResult:
    movement_id: Optional[str]
    tls_id: Optional[str]
    signal_indices: tuple[int, ...]
    decision: Optional[SignalDecision]
    executor_state: SignalState
    applied_signal: Optional[str]
    changed: bool
    reason: str


class SignalControlPipeline:
    """
    Connects:

        movement pressure
              ↓
        AI optimizer
              ↓
        movement → SUMO mapping
              ↓
        safety executor
              ↓
        SUMO signal adapter

    The pipeline does not know fixed junction names.
    """

    def __init__(
        self,
        optimizer: IncidentAwareSignalOptimizer,
        executor: SafeSignalExecutor,
        movement_mapper: SUMOMovementMapper,
        sumo_adapter: SUMOSignalAdapter,
    ):
        self.optimizer = optimizer
        self.executor = executor
        self.movement_mapper = movement_mapper
        self.sumo_adapter = sumo_adapter

    def select_and_control(
        self,
        movements: Iterable[MovementCandidate],
        now: float,
    ) -> ControlResult:

        movements = tuple(movements)

        decision = self.optimizer.select_movement(
            movements
        )

        if decision is None or decision.movement_id is None:

            return ControlResult(
                movement_id=None,
                tls_id=None,
                signal_indices=(),
                decision=decision,
                executor_state=self.executor.state,
                applied_signal=None,
                changed=False,
                reason=(
                    "no_safe_movement"
                    if decision is not None
                    else "no_decision"
             ),
            )

        movement_id = decision.movement_id

        mapping = (
            self.movement_mapper
            .get_mapping(movement_id)
        )

        if mapping is None:
            return ControlResult(
                movement_id=movement_id,
                tls_id=None,
                signal_indices=(),
                decision=decision,
                executor_state=self.executor.state,
                applied_signal=None,
                changed=False,
                reason="movement_not_mapped_to_sumo",
            )

        movement = next(
            (
                item
                for item in movements
                if item.movement_id == movement_id
            ),
            None,
        )

        if movement is None:
            return ControlResult(
                movement_id=movement_id,
                tls_id=mapping.tls_id,
                signal_indices=mapping.signal_indices,
                decision=decision,
                executor_state=self.executor.state,
                applied_signal=None,
                changed=False,
                reason="movement_disappeared",
            )

        # ------------------------------------------------------------
        # SAFETY EXECUTOR
        # ------------------------------------------------------------

        previous_state = self.executor.state

        self.executor.request_switch(
            junction_id=mapping.tls_id,
            movement_id=movement_id,
            now=now,
            movement_available=movement.available,
            movement_blocked=movement.blocked,
            reason=decision.reason,
        )

        current_state = self.executor.state

        # ------------------------------------------------------------
        # Apply only the state currently permitted by executor.
        # ------------------------------------------------------------

        if current_state == SignalState.GREEN:

            signal = self.sumo_adapter.apply_state(
                mapping.tls_id,
                SignalState.GREEN,
                mapping.signal_indices,
            )

            return ControlResult(
                movement_id=movement_id,
                tls_id=mapping.tls_id,
                signal_indices=mapping.signal_indices,
                decision=decision,
                executor_state=current_state,
                applied_signal=signal,
                changed=(
                    previous_state != current_state
                   
                ),
                reason=decision.reason,
            )

        if current_state == SignalState.YELLOW:

            signal = self.sumo_adapter.apply_state(
                mapping.tls_id,
                SignalState.YELLOW,
                mapping.signal_indices,
            )

            return ControlResult(
                movement_id=movement_id,
                tls_id=mapping.tls_id,
                signal_indices=mapping.signal_indices,
                decision=decision,
                executor_state=current_state,
                applied_signal=signal,
                changed=True,
                reason="yellow_clearance",
            )

        if current_state == SignalState.ALL_RED:

            signal = self.sumo_adapter.apply_state(
                mapping.tls_id,
                SignalState.ALL_RED,
            )

            return ControlResult(
                movement_id=movement_id,
                tls_id=mapping.tls_id,
                signal_indices=mapping.signal_indices,
                decision=decision,
                executor_state=current_state,
                applied_signal=signal,
                changed=(
                    previous_state
                    != SignalState.ALL_RED
                ),
                reason="all_red_clearance",
            )

        return ControlResult(
            movement_id=movement_id,
            tls_id=mapping.tls_id,
            signal_indices=mapping.signal_indices,
            decision=decision,
            executor_state=current_state,
            applied_signal=None,
            changed=False,
            reason="state_not_applied",
        )
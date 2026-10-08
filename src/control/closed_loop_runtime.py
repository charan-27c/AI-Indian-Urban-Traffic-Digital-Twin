from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class RuntimeCycleResult:
    timestamp: float
    state: object
    prediction_due: bool
    prediction: object | None
    history_count: int


class ClosedLoopTrafficRuntime:
    """
    Stage-2 real-time traffic runtime.

    Pipeline:

        SUMO
          ↓
        TraCIManager
          ↓
        LiveTrafficRuntime
          ↓
        TrafficStateEngine
          ↓
        HistoricalTrafficStore
          ↓
        TrafficPredictor
          ↓
        LSTM prediction

    The LSTM predicts traffic only.
    It does not directly control signals.
    """

    def __init__(
        self,
        traci_manager,
        live_traffic_runtime,
        historical_store=None,
        predictor=None,
        decision_interval: float = 5.0,
    ) -> None:

        if decision_interval <= 0:
            raise ValueError(
                "decision_interval must be > 0."
            )

        self.traci_manager = traci_manager
        self.live_traffic_runtime = live_traffic_runtime
        self.historical_store = historical_store
        self.predictor = predictor
        self.decision_interval = float(decision_interval)

        self._last_decision_time: Optional[float] = None
        self._cycle_count = 0

    @staticmethod
    def _simulation_time(manager) -> float:
        value = manager.simulation_time

        if callable(value):
            value = value()

        return float(value)

    @property
    def cycle_count(self) -> int:
        return self._cycle_count

    @property
    def last_decision_time(self) -> Optional[float]:
        return self._last_decision_time

    def _history_count(self) -> int:
        if self.historical_store is None:
            return 0

        if hasattr(self.historical_store, "count"):
            return int(self.historical_store.count())

        if hasattr(self.historical_store, "get_all"):
            return len(self.historical_store.get_all())

        return 0

    def decision_due(self, now: float) -> bool:
        now = float(now)

        if self._last_decision_time is None:
            return True

        return (
            now - self._last_decision_time
            >= self.decision_interval
        )

    def record_state(self, state):
        """
        Convert NetworkTrafficState into a real
        TrafficHistoryRecord using the existing store API.
        """

        if self.historical_store is None:
            return None

        if hasattr(self.historical_store, "add_state"):
            return self.historical_store.add_state(state)

        raise TypeError(
            "historical_store must provide add_state(state)."
        )

    def _predict(self):
        """
        Run LSTM prediction only when enough real
        historical observations exist.
        """

        if self.predictor is None:
            return None

        if self.historical_store is None:
            return None

        history_count = self._history_count()

        sequence_length = (
            self.predictor.sequence_builder.sequence_length
        )

        if history_count < sequence_length:
            return None

        return self.predictor.predict_from_store(
            self.historical_store
        )

    def step(self) -> RuntimeCycleResult:
        """
        Execute one complete real-time observation cycle.
        """

        # --------------------------------------------------
        # 1. Advance SUMO
        # --------------------------------------------------

        self.traci_manager.step()

        # --------------------------------------------------
        # 2. Collect live traffic state
        # --------------------------------------------------

        state = self.live_traffic_runtime.update_state()

        # --------------------------------------------------
        # 3. Current simulation time
        # --------------------------------------------------

        now = self._simulation_time(
            self.traci_manager
        )

        # --------------------------------------------------
        # 4. Store real observation
        # --------------------------------------------------

        self.record_state(state)

        history_count = self._history_count()

        # --------------------------------------------------
        # 5. Check prediction interval
        # --------------------------------------------------

        prediction_due = (
            self.decision_due(now)
            and history_count
            >= self._required_sequence_length()
        )

        prediction = None

        if prediction_due:

            prediction = self._predict()

            self._last_decision_time = now

        self._cycle_count += 1

        return RuntimeCycleResult(
            timestamp=now,
            state=state,
            prediction_due=prediction_due,
            prediction=prediction,
            history_count=history_count,
        )

    def _required_sequence_length(self) -> int:
        """
        Determine the required LSTM history length.

        Defaults to 5 when no predictor is configured.
        """

        if self.predictor is None:
            return 5

        return int(
            self.predictor.sequence_builder.sequence_length
        )

    def reset(self) -> None:
        self._last_decision_time = None
        self._cycle_count = 0
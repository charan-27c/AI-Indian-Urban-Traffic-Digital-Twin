from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Optional
from signal_control.network_prediction_projector import (
    NetworkPredictionProjector,
)


@dataclass(frozen=True)
class SignalRuntimeCycleResult:
    """
    Result of one closed-loop signal-control cycle.
    """

    timestamp: float
    candidate_count: int
    selected_movement_id: Optional[str]
    decision_reason: str
    decision_safe: bool
    changed: bool


class ClosedLoopSignalRuntime:
    """
    Stage 3D closed-loop signal runtime.

    Responsibilities:

        SUMO
          ↓
        live traffic state
          ↓
        movement candidates
          ↓
        incident/prediction fusion
          ↓
        signal optimizer
          ↓
        signal control pipeline
          ↓
        SUMO

    This class orchestrates existing components.

    It does NOT replace:

        - TrafficStateEngine
        - IncidentPredictionFusion
        - IncidentAwareSignalOptimizer
        - SafeSignalExecutor
        - SignalControlPipeline
        - SUMOSignalAdapter
    """

    def __init__(
        self,
        traci_manager,
        live_traffic_runtime,
        candidate_builder,
        fusion,
        signal_pipeline,
        *,
        historical_store=None,
        predictor=None,
        prediction_projector=None,
        incident_impacts: Optional[Mapping[str, object]] = None,
        availability: Optional[Mapping[str, bool]] = None,
        blocked: Optional[Mapping[str, bool]] = None,
    ) -> None:
        if traci_manager is None:
            raise ValueError("traci_manager is required.")

        if live_traffic_runtime is None:
            raise ValueError(
                "live_traffic_runtime is required."
            )

        if candidate_builder is None:
            raise ValueError(
                "candidate_builder is required."
            )

        if fusion is None:
            raise ValueError("fusion is required.")

        if signal_pipeline is None:
            raise ValueError(
                "signal_pipeline is required."
            )

        self.traci_manager = traci_manager
        self.live_traffic_runtime = live_traffic_runtime
        self.candidate_builder = candidate_builder
        self.fusion = fusion
        self.signal_pipeline = signal_pipeline
        self.historical_store = historical_store
        self.predictor = predictor
        self.prediction_projector = (
            prediction_projector
            if prediction_projector is not None
            else NetworkPredictionProjector()
        )

        self.incident_impacts = (
            dict(incident_impacts)
            if incident_impacts is not None
            else {}
        )

        self.availability = (
            dict(availability)
            if availability is not None
            else {}
        )

        self.blocked = (
            dict(blocked)
            if blocked is not None
            else {}
        )

        self.last_result: Optional[
            SignalRuntimeCycleResult
        ] = None

    # ============================================================
    # TIME
    # ============================================================

    def simulation_time(self) -> float:
        """
        Read SUMO simulation time.

        Supports both method and property forms used
        by the project's TraCI abstraction.
        """

        value = self.traci_manager.simulation_time

        if callable(value):
            value = value()

        return float(value)

    # ============================================================
    # STATE
    # ============================================================

    def update_live_state(self):
        """
        Collect the latest SUMO vehicle state and build
        the TrafficStateEngine state.
        """

        return self.live_traffic_runtime.update_state()

    # ============================================================
    # MOVEMENT EXTRACTION
    # ============================================================

    @staticmethod
    def _extract_movements(network_state) -> list:
        """
        Dynamically extract movement states from the
        network state.

        No junction or movement IDs are hard-coded.
        """

        movements = []

        junctions = getattr(
            network_state,
            "junctions",
            {},
        )

        if isinstance(junctions, Mapping):
            junction_values = junctions.values()
        else:
            junction_values = junctions or ()

        for junction in junction_values:
            junction_movements = getattr(
                junction,
                "movements",
                {},
            )

            if isinstance(
                junction_movements,
                Mapping,
            ):
                junction_movements = (
                    junction_movements.values()
                )

            for movement in junction_movements or ():
                movements.append(movement)

        return movements

    # ============================================================
    # CANDIDATES
    # ============================================================

    def generate_prediction(self):
        """
        Generate an LSTM prediction when enough
        historical observations exist.

        Returns None when prediction is not ready.
        """

        if (
            self.predictor is None
            or self.historical_store is None
        ):
            return None

        sequence_length = (
            self.predictor
            .sequence_builder
            .sequence_length
        )
        history_count = self.historical_store.count()

        if history_count < sequence_length:
            return None

        return self.predictor.predict_from_store(
            self.historical_store
        )

    def build_candidates(
        self,
        network_state,
        prediction=None,
    ):
        movements = self._extract_movements(
            network_state
        )

        predictions = {}
        if prediction is not None:
            predictions = self.prediction_projector.project(
                movements,
                network_state,
                prediction,
            )

        return self.candidate_builder.build_candidates(
            movements,
            availability=self.availability,
            blocked=self.blocked,
            predictions=predictions,
        )

    # ============================================================
    # FUSION
    # ============================================================

    def fuse_candidates(
        self,
        candidates: Iterable,
    ):
        return self.fusion.fuse_candidates(
            candidates,
            impacts=self.incident_impacts,
            availability=self.availability,
            blocked=self.blocked,
        )

    # ============================================================
    # CONTROL
    # ============================================================

    def control_cycle(self):
        """
        Execute one closed-loop signal-control cycle.

        SUMO is advanced exactly once.
        """

        # --------------------------------------------------------
        # 1. Advance SUMO
        # --------------------------------------------------------

        self.traci_manager.step()

        # --------------------------------------------------------
        # 2. Read live traffic state
        # --------------------------------------------------------

        network_state = self.update_live_state()

        # --------------------------------------------------------
        # 3. Record current state and generate prediction
        # --------------------------------------------------------

        if self.historical_store is not None:
            self.historical_store.add_state(
                network_state
            )

        prediction = self.generate_prediction()

        # --------------------------------------------------------
        # 4. Build prediction-aware candidates
        # --------------------------------------------------------

        candidates = self.build_candidates(
            network_state,
            prediction=prediction,
        )

        # --------------------------------------------------------
        # 5. Fuse incident information
        # --------------------------------------------------------

        fused_candidates = self.fuse_candidates(
            candidates
        )

        # --------------------------------------------------------
        # 6. Ask signal-control pipeline to make
        #    the safety-controlled decision.
        #
        #    The pipeline remains the authority for
        #    signal execution.
        # --------------------------------------------------------

        result = self.signal_pipeline.select_and_control(
            fused_candidates,
            self.simulation_time(),
        )

        timestamp = self.simulation_time()

        decision = getattr(
            result,
            "decision",
            None,
        )

        if decision is None:
            movement_id = None
            reason = "no_decision"
            safe = False
        else:
            movement_id = getattr(
                decision,
                "movement_id",
                None,
            )
            reason = getattr(
                decision,
                "reason",
                "unknown",
            )
            safe = bool(
                getattr(
                    decision,
                    "safe",
                    False,
                )
            )

        changed = bool(
            getattr(
                result,
                "changed",
                False,
            )
        )

        cycle = SignalRuntimeCycleResult(
            timestamp=timestamp,
            candidate_count=len(
                fused_candidates
            ),
            selected_movement_id=movement_id,
            decision_reason=reason,
            decision_safe=safe,
            changed=changed,
        )

        self.last_result = cycle

        return cycle

    # ============================================================
    # RESET
    # ============================================================

    def reset(self) -> None:
        self.last_result = None

        reset = getattr(
            self.signal_pipeline,
            "reset",
            None,
        )

        if callable(reset):
            reset()
from dataclasses import dataclass

from control.closed_loop_signal_runtime import (
    ClosedLoopSignalRuntime,
)


@dataclass
class FakeMovement:
    movement_id: str
    pressure: float = 10.0
    vehicle_count: int = 2
    queue_length: int = 1
    waiting_time: float = 2.0


@dataclass
class FakeJunction:
    movements: dict


@dataclass
class FakeNetworkState:
    junctions: dict


class FakeTraCI:
    def __init__(self):
        self.time = 0.0
        self.steps = 0

    @property
    def simulation_time(self):
        return self.time

    def step(self):
        self.steps += 1
        self.time += 1.0


class FakeLiveRuntime:
    def __init__(self, state):
        self.state = state
        self.calls = 0

    def update_state(self):
        self.calls += 1
        return self.state


class FakeCandidateBuilder:
    def __init__(self, events=None):
        self.events = events if events is not None else []
        self.predictions = None

    def build_candidates(
        self,
        movements,
        *,
        availability=None,
        blocked=None,
        predictions=None,
    ):
        self.events.append("candidates")
        self.predictions = predictions
        return list(movements)


class FakeFusion:
    def fuse_candidates(
        self,
        candidates,
        *,
        impacts=None,
        availability=None,
        blocked=None,
    ):
        return list(candidates)


class FakeDecision:
    movement_id = "M1"
    reason = "highest_safe_pressure"
    safe = True


class FakeControlResult:
    decision = FakeDecision()
    changed = True


class FakeSignalPipeline:
    def __init__(self):
        self.calls = 0

    def select_and_control(
        self,
        candidates,
        now,
    ):
        self.calls += 1

        assert len(candidates) == 1
        assert now == 1.0

        return FakeControlResult()


class FakeHistoricalStore:
    def __init__(self, events):
        self.events = events
        self.records = []

    def add_state(self, state):
        self.events.append("store")
        self.records.append(state)
        return state

    def count(self):
        return len(self.records)


class FakePredictor:
    def __init__(self, events, sequence_length=1):
        self.events = events
        self.sequence_builder = type(
            "SequenceBuilder",
            (),
            {"sequence_length": sequence_length},
        )()
        self.calls = 0

    def predict_from_store(self, store):
        self.events.append("predict")
        self.calls += 1
        return object()


class FakePredictionProjector:
    def __init__(self, events):
        self.events = events
        self.prediction_map = {"M1": object()}

    def project(self, movements, network_state, prediction):
        self.events.append("project")
        return self.prediction_map


def make_runtime():
    movement = FakeMovement(
        movement_id="M1"
    )

    state = FakeNetworkState(
        junctions={
            "J1": FakeJunction(
                movements={
                    "M1": movement
                }
            )
        }
    )

    traci = FakeTraCI()

    live = FakeLiveRuntime(state)

    pipeline = FakeSignalPipeline()

    runtime = ClosedLoopSignalRuntime(
        traci_manager=traci,
        live_traffic_runtime=live,
        candidate_builder=FakeCandidateBuilder(),
        fusion=FakeFusion(),
        signal_pipeline=pipeline,
    )

    return runtime, traci, live, pipeline


def test_closed_loop_cycle():
    runtime, traci, live, pipeline = (
        make_runtime()
    )

    result = runtime.control_cycle()

    assert traci.steps == 1
    assert live.calls == 1
    assert pipeline.calls == 1

    assert result.timestamp == 1.0
    assert result.candidate_count == 1
    assert result.selected_movement_id == "M1"
    assert result.decision_reason == (
        "highest_safe_pressure"
    )
    assert result.decision_safe is True
    assert result.changed is True


def test_dynamic_movement_extraction():
    runtime, _, _, _ = make_runtime()

    state = FakeNetworkState(
        junctions={
            "J_A": FakeJunction(
                movements={
                    "A": FakeMovement("A"),
                    "B": FakeMovement("B"),
                }
            ),
            "J_B": FakeJunction(
                movements={
                    "C": FakeMovement("C"),
                }
            ),
        }
    )

    movements = (
        runtime._extract_movements(state)
    )

    assert {
        movement.movement_id
        for movement in movements
    } == {"A", "B", "C"}


def test_prediction_is_generated_after_recording_and_projected_to_candidates():
    events = []
    movement = FakeMovement(
        movement_id="M1"
    )
    state = FakeNetworkState(
        junctions={
            "J1": FakeJunction(
                movements={"M1": movement}
            )
        }
    )
    traci = FakeTraCI()
    live = FakeLiveRuntime(state)
    candidate_builder = FakeCandidateBuilder(events)
    store = FakeHistoricalStore(events)
    predictor = FakePredictor(events)
    projector = FakePredictionProjector(events)
    runtime = ClosedLoopSignalRuntime(
        traci_manager=traci,
        live_traffic_runtime=live,
        candidate_builder=candidate_builder,
        fusion=FakeFusion(),
        signal_pipeline=FakeSignalPipeline(),
        historical_store=store,
        predictor=predictor,
        prediction_projector=projector,
    )

    runtime.control_cycle()

    assert events == [
        "store",
        "predict",
        "project",
        "candidates",
    ]
    assert predictor.calls == 1
    assert candidate_builder.predictions is projector.prediction_map


def test_generate_prediction_waits_for_sequence_length():
    events = []
    runtime, _, _, _ = make_runtime()
    runtime.historical_store = FakeHistoricalStore(events)
    runtime.predictor = FakePredictor(
        events,
        sequence_length=2,
    )

    assert runtime.generate_prediction() is None
    assert runtime.predictor.calls == 0

    runtime.historical_store.add_state(object())

    assert runtime.generate_prediction() is None
    assert runtime.predictor.calls == 0

    runtime.historical_store.add_state(object())

    assert runtime.generate_prediction() is not None
    assert runtime.predictor.calls == 1


def test_reset():
    runtime, _, _, _ = make_runtime()

    runtime.control_cycle()

    assert runtime.last_result is not None

    runtime.reset()

    assert runtime.last_result is None
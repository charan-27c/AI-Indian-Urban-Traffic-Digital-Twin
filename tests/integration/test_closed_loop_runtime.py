from types import SimpleNamespace

from control.closed_loop_runtime import (
    ClosedLoopTrafficRuntime,
)


class FakeTraCIManager:

    def __init__(self):
        self.simulation_time = 0.0
        self.step_count = 0

    def step(self):
        self.simulation_time += 1.0
        self.step_count += 1


class FakeLiveTrafficRuntime:

    def __init__(self):
        self.calls = 0

    def update_state(self):
        self.calls += 1

        return SimpleNamespace(
            timestamp=float(self.calls),
            vehicle_count=self.calls,
        )


class FakeHistoricalStore:

    def __init__(self):
        self.records = []

    def add_state(self, state):
        self.records.append(state)
        return state

    def get_all(self):
        return list(self.records)

    def count(self):
        return len(self.records)

    def clear(self):
        self.records.clear()


class FakePredictor:

    def __init__(self, sequence_length=5):
        self.sequence_builder = SimpleNamespace(
            sequence_length=sequence_length
        )
        self.calls = 0

    def predict_from_store(self, store):
        self.calls += 1
        return SimpleNamespace(
            vehicle_count=10.0,
            queue_length=4.0,
            average_speed=8.0,
            pressure=12.0,
        )


def build_runtime(
    decision_interval=5.0,
    predictor=None,
):
    manager = FakeTraCIManager()
    live_runtime = FakeLiveTrafficRuntime()
    store = FakeHistoricalStore()

    runtime = ClosedLoopTrafficRuntime(
        traci_manager=manager,
        live_traffic_runtime=live_runtime,
        historical_store=store,
        predictor=predictor,
        decision_interval=decision_interval,
    )

    return runtime, manager, live_runtime, store


def test_first_cycle_is_due():

    runtime, _, _, _ = build_runtime(
        predictor=FakePredictor(
            sequence_length=1
        )
    )

    result = runtime.step()

    assert result.prediction_due is True
    assert result.timestamp == 1.0


def test_live_state_is_collected_each_cycle():

    runtime, _, live_runtime, _ = build_runtime()

    runtime.step()
    runtime.step()
    runtime.step()

    assert live_runtime.calls == 3
    assert runtime.cycle_count == 3


def test_decision_interval_is_respected():

    runtime, _, _, _ = build_runtime(
        decision_interval=5.0,
        predictor=FakePredictor(
            sequence_length=1
        ),
    )

    result1 = runtime.step()
    result2 = runtime.step()
    result3 = runtime.step()
    result4 = runtime.step()
    result5 = runtime.step()

    assert result1.prediction_due is True
    assert result2.prediction_due is False
    assert result3.prediction_due is False
    assert result4.prediction_due is False
    assert result5.prediction_due is False

    runtime.step()
    result7 = runtime.step()

    assert result7.prediction_due is False


def test_historical_state_is_recorded():

    runtime, _, _, store = build_runtime()

    runtime.step()
    runtime.step()

    assert len(store.records) == 2


def test_history_is_not_recorded_as_generic_append():

    runtime, _, _, store = build_runtime()

    runtime.step()

    assert len(store.records) == 1


def test_lstm_prediction_waits_for_five_observations():

    manager = FakeTraCIManager()
    live_runtime = FakeLiveTrafficRuntime()
    store = FakeHistoricalStore()
    predictor = FakePredictor(
        sequence_length=5
    )
    runtime = ClosedLoopTrafficRuntime(
        traci_manager=manager,
        live_traffic_runtime=live_runtime,
        historical_store=store,
        predictor=predictor,
        decision_interval=5.0,
    )

    results = []
    for _ in range(4):
        results.append(runtime.step())

    assert all(
        result.prediction is None
        for result in results
    )
    assert predictor.calls == 0


def test_lstm_prediction_runs_after_five_real_observations():

    manager = FakeTraCIManager()
    live_runtime = FakeLiveTrafficRuntime()
    store = FakeHistoricalStore()
    predictor = FakePredictor(
        sequence_length=5
    )
    runtime = ClosedLoopTrafficRuntime(
        traci_manager=manager,
        live_traffic_runtime=live_runtime,
        historical_store=store,
        predictor=predictor,
        decision_interval=5.0,
    )

    results = []
    for _ in range(5):
        results.append(runtime.step())

    assert len(store.records) == 5
    assert predictor.calls == 1
    assert results[-1].prediction_due is True
    assert results[-1].prediction is not None
    assert results[-1].history_count == 5


def test_prediction_waits_for_next_decision_interval():

    manager = FakeTraCIManager()
    live_runtime = FakeLiveTrafficRuntime()
    store = FakeHistoricalStore()
    predictor = FakePredictor(
        sequence_length=5
    )
    runtime = ClosedLoopTrafficRuntime(
        traci_manager=manager,
        live_traffic_runtime=live_runtime,
        historical_store=store,
        predictor=predictor,
        decision_interval=5.0,
    )

    for _ in range(5):
        runtime.step()

    assert predictor.calls == 1
    result6 = runtime.step()
    assert result6.prediction is None
    assert predictor.calls == 1


def test_prediction_runs_again_after_five_seconds():

    manager = FakeTraCIManager()
    live_runtime = FakeLiveTrafficRuntime()
    store = FakeHistoricalStore()
    predictor = FakePredictor(
        sequence_length=5
    )
    runtime = ClosedLoopTrafficRuntime(
        traci_manager=manager,
        live_traffic_runtime=live_runtime,
        historical_store=store,
        predictor=predictor,
        decision_interval=5.0,
    )

    for _ in range(5):
        runtime.step()

    assert predictor.calls == 1
    for _ in range(4):
        result = runtime.step()

    assert result.timestamp == 9.0
    assert predictor.calls == 1

    result = runtime.step()
    assert result.timestamp == 10.0
    assert result.prediction_due is True
    assert result.prediction is not None
    assert predictor.calls == 2


def test_reset_clears_runtime_state():

    runtime, _, _, _ = build_runtime(
        predictor=FakePredictor(
            sequence_length=1
        )
    )

    runtime.step()
    runtime.step()

    assert runtime.cycle_count == 2
    assert runtime.last_decision_time is not None

    runtime.reset()

    assert runtime.cycle_count == 0
    assert runtime.last_decision_time is None


def test_runtime_supports_simulation_time_method():

    class MethodTimeManager:

        def __init__(self):
            self._time = 0.0

        def step(self):
            self._time += 1.0

        def simulation_time(self):
            return self._time

    manager = MethodTimeManager()

    live_runtime = FakeLiveTrafficRuntime()

    runtime = ClosedLoopTrafficRuntime(
        traci_manager=manager,
        live_traffic_runtime=live_runtime,
    )

    manager.step()

    now = runtime._simulation_time(manager)

    assert now == 1.0
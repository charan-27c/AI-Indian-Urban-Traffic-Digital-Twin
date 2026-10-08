import pytest

from src.prediction.historical_store import (
    HistoricalTrafficStore,
)
from src.prediction.lstm_model import (
    TrafficLSTMModel,
)
from src.prediction.predictor import (
    TrafficPredictor,
)
from src.prediction.runtime import (
    TrafficPredictionRuntime,
)
from src.prediction.models import (
    TrafficHistoryRecord,
)


def make_record(timestamp: float) -> TrafficHistoryRecord:
    return TrafficHistoryRecord(
        timestamp=timestamp,
        vehicle_count=20,
        queue_length=5,
        average_speed=10.0,
        density=0.25,
        occupancy=0.30,
        waiting_time=5.0,
        delay=3.0,
        flow_rate=15.0,
        pressure=25.0,
    )


def create_runtime(
    prediction_interval: float = 5.0,
) -> TrafficPredictionRuntime:

    store = HistoricalTrafficStore()

    model = TrafficLSTMModel()

    predictor = TrafficPredictor(
        model=model,
    )

    return TrafficPredictionRuntime(
        store=store,
        predictor=predictor,
        prediction_interval=prediction_interval,
    )


# ============================================================
# INITIALIZATION
# ============================================================

def test_prediction_runtime_initializes():

    runtime = create_runtime()

    assert runtime is not None
    assert runtime.prediction_interval == 5.0


# ============================================================
# STORE TRAFFIC STATE
# ============================================================

def test_runtime_stores_traffic_history():

    runtime = create_runtime()

    runtime.update(
        make_record(0.0)
    )

    assert runtime.store.count() == 1


# ============================================================
# NOT ENOUGH HISTORY
# ============================================================

def test_runtime_waits_for_enough_history():

    runtime = create_runtime()

    for timestamp in range(4):

        result = runtime.update(
            make_record(float(timestamp))
        )

        assert result is None

    assert runtime.store.count() == 4


# ============================================================
# FIRST PREDICTION
# ============================================================

def test_runtime_generates_prediction():

    runtime = create_runtime()

    result = None

    for timestamp in range(5):

        result = runtime.update(
            make_record(float(timestamp))
        )

    assert result is not None

    assert result.vehicle_count is not None
    assert result.queue_length is not None
    assert result.average_speed is not None


# ============================================================
# PREDICTION INTERVAL
# ============================================================

def test_runtime_respects_prediction_interval():

    runtime = create_runtime(
        prediction_interval=5.0
    )

    for timestamp in range(5):

        runtime.update(
            make_record(float(timestamp))
        )

    first_prediction = runtime.last_prediction

    assert first_prediction is not None

    # No new prediction before interval expires.
    result = runtime.update(
        make_record(5.0)
    )

    assert result is None


# ============================================================
# NEXT PREDICTION
# ============================================================

def test_runtime_generates_next_prediction():

    runtime = create_runtime(
        prediction_interval=5.0
    )

    for timestamp in range(5):

        runtime.update(
            make_record(float(timestamp))
        )

    first_prediction = runtime.last_prediction

    assert first_prediction is not None

    runtime.update(
        make_record(5.0)
    )

    result = runtime.update(
        make_record(10.0)
    )

    assert result is not None

    assert runtime.last_prediction is not None


# ============================================================
# MANUAL PREDICTION
# ============================================================

def test_runtime_predict_now():

    runtime = create_runtime()

    for timestamp in range(5):

        runtime.update(
            make_record(float(timestamp))
        )

    prediction = runtime.predict_now()

    assert prediction is not None


# ============================================================
# RESET
# ============================================================

def test_runtime_reset():

    runtime = create_runtime()

    for timestamp in range(5):

        runtime.update(
            make_record(float(timestamp))
        )

    assert runtime.store.count() == 5

    runtime.reset()

    assert runtime.store.count() == 0
    assert runtime.last_prediction is None
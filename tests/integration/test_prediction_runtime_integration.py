from dataclasses import dataclass

import numpy as np

from src.prediction.historical_store import HistoricalTrafficStore
from src.prediction.lstm_model import TrafficLSTMModel, LSTMConfig
from src.prediction.predictor import TrafficPredictor
from src.prediction.runtime import TrafficPredictionRuntime
from src.prediction.models import TrafficHistoryRecord


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

    config = LSTMConfig(
        sequence_length=5,
        feature_count=9,
    )

    model = TrafficLSTMModel(
        config=config
    )

    predictor = TrafficPredictor(
        model=model
    )

    store = HistoricalTrafficStore()

    return TrafficPredictionRuntime(
        store=store,
        predictor=predictor,
        prediction_interval=prediction_interval,
    )


# ============================================================
# INITIALIZATION
# ============================================================

def test_runtime_initializes():

    runtime = create_runtime()

    assert runtime is not None
    assert runtime.prediction_interval == 5.0
    assert runtime.store.count() == 0
    assert runtime.last_prediction is None


# ============================================================
# LIVE STATE → HISTORY
# ============================================================

def test_live_state_enters_history():

    runtime = create_runtime()

    runtime.update(
        make_record(0.0)
    )

    assert runtime.store.count() == 1


# ============================================================
# WAIT FOR HISTORY
# ============================================================

def test_runtime_waits_for_history():

    runtime = create_runtime()

    for timestamp in range(4):

        result = runtime.update(
            make_record(float(timestamp))
        )

        assert result is None

    assert runtime.store.count() == 4
    assert runtime.last_prediction is None


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
    assert runtime.last_prediction is not None


# ============================================================
# PREDICTION TYPE
# ============================================================

def test_prediction_has_nine_features():

    runtime = create_runtime()

    for timestamp in range(5):

        runtime.update(
            make_record(float(timestamp))
        )

    prediction = runtime.last_prediction

    assert prediction is not None

    values = prediction.to_array()

    assert isinstance(
        values,
        np.ndarray,
    )

    assert values.shape == (9,)


# ============================================================
# PREDICTION INTERVAL
# ============================================================

def test_runtime_respects_interval():

    runtime = create_runtime(
        prediction_interval=5.0
    )

    for timestamp in range(5):

        runtime.update(
            make_record(float(timestamp))
        )

    first_prediction = (
        runtime.last_prediction
    )

    # Only one second later.
    result = runtime.update(
        make_record(6.0)
    )

    assert result is None

    assert (
        runtime.last_prediction
        is first_prediction
    )


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

    first_prediction = (
        runtime.last_prediction
    )

    result = runtime.update(
        make_record(10.0)
    )

    assert result is not None
    assert runtime.last_prediction is not None

    assert (
        runtime.last_prediction
        is result
    )


# ============================================================
# MANUAL PREDICTION
# ============================================================

def test_predict_now():

    runtime = create_runtime()

    for timestamp in range(5):

        runtime.update(
            make_record(float(timestamp))
        )

    prediction = runtime.predict_now()

    assert prediction is not None

    assert (
        runtime.last_prediction
        is prediction
    )


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
    assert runtime.last_prediction is not None

    runtime.reset()

    assert runtime.store.count() == 0
    assert runtime.last_prediction is None
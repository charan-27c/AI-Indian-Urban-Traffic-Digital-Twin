import numpy as np

from src.prediction import (
    LSTMConfig,
    TrafficHistoryRecord,
    TrafficLSTMModel,
    TrafficPredictor,
    TrafficSequenceBuilder,
)


def create_record(timestamp: float) -> TrafficHistoryRecord:
    return TrafficHistoryRecord(
        timestamp=timestamp,
        vehicle_count=20,
        queue_length=5,
        average_speed=8.0,
        density=0.25,
        occupancy=0.30,
        waiting_time=10.0,
        delay=5.0,
        flow_rate=15.0,
        pressure=25.0,
    )


def test_prediction_service():

    config = LSTMConfig(
        sequence_length=5,
        feature_count=9,
    )

    model = TrafficLSTMModel(
        config=config
    )

    builder = TrafficSequenceBuilder(
        sequence_length=5
    )

    predictor = TrafficPredictor(
        model=model,
        sequence_builder=builder,
    )

    records = [
        create_record(float(i))
        for i in range(6)
    ]

    prediction = predictor.predict(
        records
    )

    assert prediction is not None

    assert prediction.vehicle_count >= 0
    assert prediction.queue_length >= 0
    assert prediction.average_speed >= 0
    assert prediction.density >= 0
    assert 0 <= prediction.occupancy <= 1
    assert prediction.waiting_time >= 0
    assert prediction.delay >= 0
    assert prediction.flow_rate >= 0
    assert prediction.pressure >= 0


def test_prediction_array():

    model = TrafficLSTMModel()

    predictor = TrafficPredictor(
        model=model
    )

    X = np.random.rand(
        5,
        9,
    ).astype(
        np.float32
    )

    prediction = predictor.predict_array(
        X
    )

    assert prediction.to_array().shape == (
        9,
    )
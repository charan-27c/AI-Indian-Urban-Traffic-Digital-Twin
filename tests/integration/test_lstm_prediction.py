import numpy as np

from src.prediction import (
    HistoricalTrafficStore,
    TrafficHistoryRecord,
    TrafficLSTMModel,
    TrafficPredictor,
)


def make_record(
    timestamp: float,
    vehicle_count: int,
) -> TrafficHistoryRecord:

    return TrafficHistoryRecord(
        timestamp=timestamp,
        vehicle_count=vehicle_count,
        queue_length=vehicle_count // 2,
        average_speed=10.0,
        density=float(vehicle_count),
        occupancy=0.5,
        waiting_time=5.0,
        delay=3.0,
        flow_rate=float(vehicle_count),
        pressure=float(vehicle_count),
    )


def test_lstm_prediction_pipeline():

    # --------------------------------------------------------
    # Historical traffic
    # --------------------------------------------------------

    store = HistoricalTrafficStore()

    for timestamp in range(5):

        store.add(
            make_record(
                timestamp=float(timestamp),
                vehicle_count=10 + timestamp,
            )
        )

    assert store.count() == 5

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = TrafficLSTMModel()

    # --------------------------------------------------------
    # Predictor
    # --------------------------------------------------------

    predictor = TrafficPredictor(
        model=model
    )

    # --------------------------------------------------------
    # Predict from historical store
    # --------------------------------------------------------

    prediction = (
        predictor.predict_from_store(
            store
        )
    )

    # --------------------------------------------------------
    # Validate prediction
    # --------------------------------------------------------

    assert prediction is not None

    assert isinstance(
        prediction.vehicle_count,
        float,
    )

    assert isinstance(
        prediction.queue_length,
        float,
    )

    assert isinstance(
        prediction.average_speed,
        float,
    )

    assert isinstance(
        prediction.density,
        float,
    )

    assert isinstance(
        prediction.occupancy,
        float,
    )

    assert isinstance(
        prediction.waiting_time,
        float,
    )

    assert isinstance(
        prediction.delay,
        float,
    )

    assert isinstance(
        prediction.flow_rate,
        float,
    )

    assert isinstance(
        prediction.pressure,
        float,
    )

    # --------------------------------------------------------
    # Dictionary representation
    # --------------------------------------------------------

    result = prediction.to_dict()

    assert len(result) == 9

    assert (
        set(result.keys())
        == {
            "vehicle_count",
            "queue_length",
            "average_speed",
            "density",
            "occupancy",
            "waiting_time",
            "delay",
            "flow_rate",
            "pressure",
        }
    )


def test_lstm_prediction_vector_shape():

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

    prediction = model.predict_sequence(
        X
    )

    assert prediction.shape == (
        9,
    )

    result = (
        predictor.vector_to_prediction(
            prediction
        )
    )

    assert result.vehicle_count == float(
        prediction[0]
    )


def test_vector_to_prediction_preserves_raw_values():

    predictor = TrafficPredictor(
        model=TrafficLSTMModel()
    )
    vector = np.asarray(
        [-1.0, -2.0, -3.0, -4.0, 1.5, -6.0, -7.0, -8.0, -9.0],
        dtype=np.float32,
    )

    result = predictor.vector_to_prediction(
        vector
    )

    np.testing.assert_array_equal(
        result.to_array(),
        vector,
    )
import numpy as np

from src.prediction.models import (
    TrafficHistoryRecord,
)

from src.prediction.sequence_builder import (
    TrafficSequenceBuilder,
)


def make_record(
    timestamp: float,
    vehicles: int,
) -> TrafficHistoryRecord:

    return TrafficHistoryRecord(
        timestamp=timestamp,

        vehicle_count=vehicles,
        queue_length=vehicles // 2,

        average_speed=8.0,

        density=float(vehicles),
        occupancy=0.3,

        waiting_time=10.0,
        delay=5.0,

        flow_rate=float(vehicles),
        pressure=float(vehicles),
    )


def test_sequence_builder_creates_lstm_sequence():

    records = [
        make_record(
            float(i),
            10 + i,
        )
        for i in range(6)
    ]

    builder = TrafficSequenceBuilder(
        sequence_length=5
    )

    sequence = builder.build_sequence(
        records
    )

    assert sequence.X.shape == (
        5,
        9,
    )

    assert sequence.y.shape == (
        0,
    )

    assert (
        sequence.X[0][0]
        == 11
    )

    assert (
        sequence.X[-1][0]
        == 15
    )


def test_sequence_builder_inference_accepts_exact_sequence_length():

    records = [
        make_record(
            float(i),
            10 + i,
        )
        for i in range(5)
    ]

    builder = TrafficSequenceBuilder(
        sequence_length=5
    )

    sequence = builder.build_sequence(
        records
    )

    assert sequence.X.shape == (
        5,
        9,
    )
    assert sequence.y.shape == (
        0,
    )


def test_sequence_builder_creates_dataset():

    records = [
        make_record(
            float(i),
            10 + i,
        )
        for i in range(10)
    ]

    builder = TrafficSequenceBuilder(
        sequence_length=5
    )

    X, y = builder.build_dataset(
        records
    )

    assert X.shape == (
        5,
        5,
        9,
    )

    assert y.shape == (
        5,
        9,
    )

    assert isinstance(
        X,
        np.ndarray,
    )


def test_sequence_builder_requires_enough_data():

    records = [
        make_record(
            float(i),
            10,
        )
        for i in range(5)
    ]

    builder = TrafficSequenceBuilder(
        sequence_length=5
    )

    try:
        builder.build_dataset(records)
        assert False
    except ValueError:
        assert True
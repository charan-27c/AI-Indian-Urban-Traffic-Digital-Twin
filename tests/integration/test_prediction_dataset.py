import numpy as np
import pytest

from src.prediction.dataset import (
    TrafficDatasetBuilder,
    TrafficDatasetConfig,
)
from src.prediction.models import TrafficHistoryRecord


def make_record(timestamp: float) -> TrafficHistoryRecord:
    return TrafficHistoryRecord(
        timestamp=timestamp,
        vehicle_count=20 + timestamp,
        queue_length=5 + timestamp,
        average_speed=8.0,
        density=0.25,
        occupancy=0.30,
        waiting_time=10.0,
        delay=5.0,
        flow_rate=15.0,
        pressure=25.0,
    )


def make_records(count: int):
    return [
        make_record(float(i))
        for i in range(count)
    ]


# ============================================================
# CONFIGURATION
# ============================================================

def test_dataset_configuration():

    config = TrafficDatasetConfig(
        sequence_length=5,
        train_ratio=0.70,
        validation_ratio=0.15,
        test_ratio=0.15,
    )

    assert config.sequence_length == 5

    assert np.isclose(
        config.train_ratio
        + config.validation_ratio
        + config.test_ratio,
        1.0,
    )


# ============================================================
# RECORD → VECTOR
# ============================================================

def test_record_to_vector():

    builder = TrafficDatasetBuilder()

    record = make_record(0.0)

    vector = builder.record_to_vector(
        record
    )

    assert vector.shape == (9,)

    assert vector.dtype == np.float32

    assert vector[0] == 20.0


# ============================================================
# RECORDS → MATRIX
# ============================================================

def test_records_to_matrix():

    builder = TrafficDatasetBuilder()

    records = make_records(10)

    matrix = builder.records_to_matrix(
        records
    )

    assert matrix.shape == (
        10,
        9,
    )

    assert matrix.dtype == np.float32


# ============================================================
# MIN-MAX SCALER
# ============================================================

def test_min_max_normalization():

    builder = TrafficDatasetBuilder()

    matrix = np.asarray(
        [
            [0.0, 10.0],
            [5.0, 20.0],
            [10.0, 30.0],
        ],
        dtype=np.float32,
    )

    feature_min, feature_max = (
        builder.fit_min_max(
            matrix
        )
    )

    normalized = (
        builder.normalize_matrix(
            matrix,
            feature_min,
            feature_max,
        )
    )

    assert np.allclose(
        normalized[0],
        [0.0, 0.0],
    )

    assert np.allclose(
        normalized[-1],
        [1.0, 1.0],
    )


# ============================================================
# SLIDING WINDOW
# ============================================================

def test_create_sequences():

    builder = TrafficDatasetBuilder(
        TrafficDatasetConfig(
            sequence_length=5,
            normalize=False,
        )
    )

    matrix = np.random.rand(
        10,
        9,
    ).astype(
        np.float32
    )

    X, y = builder.create_sequences(
        matrix
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


# ============================================================
# INSUFFICIENT DATA
# ============================================================

def test_create_sequences_requires_enough_data():

    builder = TrafficDatasetBuilder(
        TrafficDatasetConfig(
            sequence_length=5,
            normalize=False,
        )
    )

    matrix = np.random.rand(
        5,
        9,
    ).astype(
        np.float32
    )

    with pytest.raises(
        ValueError
    ):
        builder.create_sequences(
            matrix
        )


# ============================================================
# DATASET BUILD
# ============================================================

def test_build_dataset():

    builder = TrafficDatasetBuilder(
        TrafficDatasetConfig(
            sequence_length=5,
            train_ratio=0.70,
            validation_ratio=0.15,
            test_ratio=0.15,
        )
    )

    records = make_records(100)

    dataset = builder.build(
        records
    )

    assert dataset.X_train.ndim == 3
    assert dataset.y_train.ndim == 2

    assert dataset.X_validation.ndim == 3
    assert dataset.y_validation.ndim == 2

    assert dataset.X_test.ndim == 3
    assert dataset.y_test.ndim == 2

    assert dataset.X_train.shape[1:] == (
        5,
        9,
    )

    assert dataset.y_train.shape[1] == 9


# ============================================================
# NORMALIZATION RANGE
# ============================================================

def test_dataset_normalization_range():

    builder = TrafficDatasetBuilder(
        TrafficDatasetConfig(
            sequence_length=5,
            normalize=True,
        )
    )

    records = make_records(100)

    dataset = builder.build(
        records
    )

    assert np.all(
        dataset.X_train >= 0.0
    )

    assert np.all(
        dataset.X_train <= 1.0
    )

    assert np.all(
        dataset.y_train >= 0.0
    )

    assert np.all(
        dataset.y_train <= 1.0
    )
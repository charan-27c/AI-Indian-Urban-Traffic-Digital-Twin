from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from .models import TrafficHistoryRecord


# ============================================================
# DATASET CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class TrafficDatasetConfig:
    sequence_length: int = 5
    train_ratio: float = 0.70
    validation_ratio: float = 0.15
    test_ratio: float = 0.15

    normalize: bool = True


# ============================================================
# DATASET RESULT
# ============================================================

@dataclass
class TrafficDataset:
    X_train: np.ndarray
    y_train: np.ndarray

    X_validation: np.ndarray
    y_validation: np.ndarray

    X_test: np.ndarray
    y_test: np.ndarray

    feature_min: np.ndarray | None = None
    feature_max: np.ndarray | None = None


# ============================================================
# DATASET BUILDER
# ============================================================

class TrafficDatasetBuilder:
    """
    Build supervised LSTM datasets from historical traffic
    records.

    Historical records:

        t0
        t1
        t2
        t3
        t4
        t5
        ...

    With sequence_length = 5:

        X = [t0,t1,t2,t3,t4]
        y = [t5]

        X = [t1,t2,t3,t4,t5]
        y = [t6]

    No fixed junction, lane, approach, or movement is assumed.
    """

    FEATURE_COUNT = 9

    def __init__(
        self,
        config: TrafficDatasetConfig | None = None,
    ) -> None:

        self.config = (
            config
            if config is not None
            else TrafficDatasetConfig()
        )

        if self.config.sequence_length <= 0:
            raise ValueError(
                "sequence_length must be greater than zero."
            )

        ratios = (
            self.config.train_ratio,
            self.config.validation_ratio,
            self.config.test_ratio,
        )

        if not np.isclose(
            sum(ratios),
            1.0,
        ):
            raise ValueError(
                "train_ratio + validation_ratio + "
                "test_ratio must equal 1.0."
            )

    # ========================================================
    # RECORD → VECTOR
    # ========================================================

    @staticmethod
    def record_to_vector(
        record: TrafficHistoryRecord,
    ) -> np.ndarray:

        return np.asarray(
            [
                record.vehicle_count,
                record.queue_length,
                record.average_speed,
                record.density,
                record.occupancy,
                record.waiting_time,
                record.delay,
                record.flow_rate,
                record.pressure,
            ],
            dtype=np.float32,
        )

    # ========================================================
    # RECORDS → MATRIX
    # ========================================================

    def records_to_matrix(
        self,
        records: List[TrafficHistoryRecord],
    ) -> np.ndarray:

        if not records:
            raise ValueError(
                "At least one traffic record is required."
            )

        return np.asarray(
            [
                self.record_to_vector(record)
                for record in records
            ],
            dtype=np.float32,
        )

    # ========================================================
    # NORMALIZATION
    # ========================================================

    @staticmethod
    def fit_min_max(
        matrix: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:

        feature_min = np.min(
            matrix,
            axis=0,
        )

        feature_max = np.max(
            matrix,
            axis=0,
        )

        return feature_min, feature_max

    @staticmethod
    def normalize_matrix(
        matrix: np.ndarray,
        feature_min: np.ndarray,
        feature_max: np.ndarray,
    ) -> np.ndarray:

        denominator = (
            feature_max - feature_min
        )

        denominator = np.where(
            denominator == 0,
            1.0,
            denominator,
        )

        return (
            matrix - feature_min
        ) / denominator

    # ========================================================
    # SLIDING WINDOWS
    # ========================================================

    def create_sequences(
        self,
        matrix: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:

        sequence_length = (
            self.config.sequence_length
        )

        if len(matrix) <= sequence_length:
            raise ValueError(
                f"At least {sequence_length + 1} "
                "records are required."
            )

        X = []
        y = []

        for index in range(
            len(matrix) - sequence_length
        ):

            X.append(
                matrix[
                    index:
                    index + sequence_length
                ]
            )

            y.append(
                matrix[
                    index + sequence_length
                ]
            )

        return (
            np.asarray(
                X,
                dtype=np.float32,
            ),
            np.asarray(
                y,
                dtype=np.float32,
            ),
        )

    # ========================================================
    # BUILD DATASET
    # ========================================================

    def build(
        self,
        records: List[TrafficHistoryRecord],
    ) -> TrafficDataset:

        matrix = self.records_to_matrix(
            records
        )

        total_records = len(matrix)

        train_end = int(
            total_records *
            self.config.train_ratio
        )

        validation_end = (
            train_end
            + int(
                total_records *
                self.config.validation_ratio
            )
        )

        if train_end <= 0:
            raise ValueError(
                "Training dataset is empty."
            )

        if validation_end <= train_end:
            raise ValueError(
                "Validation dataset is empty."
            )

        # ----------------------------------------------------
        # Fit scaler ONLY on training data
        # ----------------------------------------------------

        training_matrix = matrix[
            :train_end
        ]

        feature_min, feature_max = (
            self.fit_min_max(
                training_matrix
            )
        )

        if self.config.normalize:

            matrix = self.normalize_matrix(
                matrix,
                feature_min,
                feature_max,
            )

        # ----------------------------------------------------
        # Split normalized data
        # ----------------------------------------------------

        train_matrix = matrix[
            :train_end
        ]

        validation_matrix = matrix[
            train_end:validation_end
        ]

        test_matrix = matrix[
            validation_end:
        ]

        # ----------------------------------------------------
        # Create sequences
        # ----------------------------------------------------

        X_train, y_train = (
            self.create_sequences(
                train_matrix
            )
        )

        X_validation, y_validation = (
            self.create_sequences(
                validation_matrix
            )
        )

        X_test, y_test = (
            self.create_sequences(
                test_matrix
            )
        )

        return TrafficDataset(
            X_train=X_train,
            y_train=y_train,
            X_validation=X_validation,
            y_validation=y_validation,
            X_test=X_test,
            y_test=y_test,
            feature_min=feature_min,
            feature_max=feature_max,
        )
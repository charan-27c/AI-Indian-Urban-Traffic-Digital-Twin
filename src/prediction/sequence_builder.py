from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from .models import TrafficHistoryRecord


# ============================================================
# TRAFFIC FEATURE VECTOR
# ============================================================

@dataclass(frozen=True)
class TrafficFeatureVector:
    """
    Feature representation of one traffic observation.

    The feature order is intentionally fixed so the same
    representation can be used by the LSTM during training
    and inference.
    """

    vehicle_count: float
    queue_length: float
    average_speed: float

    density: float
    occupancy: float

    waiting_time: float
    delay: float

    flow_rate: float
    pressure: float

    def to_array(self) -> np.ndarray:
        """
        Convert the feature vector to a numeric array.
        """

        return np.asarray(
            [
                self.vehicle_count,
                self.queue_length,
                self.average_speed,
                self.density,
                self.occupancy,
                self.waiting_time,
                self.delay,
                self.flow_rate,
                self.pressure,
            ],
            dtype=np.float32,
        )


# ============================================================
# TRAFFIC SEQUENCE
# ============================================================

@dataclass(frozen=True)
class TrafficSequence:
    """
    One traffic sequence.

    X:
        Historical traffic observations.

    y:
        Traffic state immediately after the historical sequence for
        training, or an empty array when built for inference.
    """

    X: np.ndarray
    y: np.ndarray


# ============================================================
# TRAFFIC STATE -> FEATURE VECTOR
# ============================================================

def traffic_state_to_features(
    state,
) -> TrafficFeatureVector:
    """
    Convert a NetworkTrafficState-like object into
    a topology-independent feature vector.

    No fixed junction IDs are assumed.
    """

    return TrafficFeatureVector(
        vehicle_count=float(
            state.vehicle_count
        ),
        queue_length=float(
            state.queue_length
        ),
        average_speed=float(
            state.average_speed
        ),
        density=float(
            state.density
        ),
        occupancy=float(
            state.occupancy
        ),
        waiting_time=float(
            state.waiting_time
        ),
        delay=float(
            state.delay
        ),
        flow_rate=float(
            state.flow_rate
        ),
        pressure=float(
            state.pressure
        ),
    )


# ============================================================
# SEQUENCE -> MATRIX
# ============================================================

def sequence_to_matrix(
    sequence: TrafficSequence,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert a TrafficSequence into X and y matrices.

    X shape:
        (sequence_length, feature_count)

    y shape:
        (feature_count,) for training, or (0,) for inference.
    """

    return (
        np.asarray(
            sequence.X,
            dtype=np.float32,
        ),
        np.asarray(
            sequence.y,
            dtype=np.float32,
        ),
    )


# ============================================================
# TRAFFIC SEQUENCE BUILDER
# ============================================================

class TrafficSequenceBuilder:
    """
    Converts historical traffic records into sequences
    suitable for an LSTM.

    No junction IDs or fixed topology are assumed here.
    """

    FEATURE_NAMES = (
        "vehicle_count",
        "queue_length",
        "average_speed",
        "density",
        "occupancy",
        "waiting_time",
        "delay",
        "flow_rate",
        "pressure",
    )

    def __init__(
        self,
        sequence_length: int = 5,
    ) -> None:

        if sequence_length <= 0:
            raise ValueError(
                "sequence_length must be greater than zero."
            )

        self.sequence_length = sequence_length

    # ========================================================
    # RECORD -> FEATURES
    # ========================================================

    def record_to_features(
        self,
        record: TrafficHistoryRecord,
    ) -> List[float]:

        return [
            float(record.vehicle_count),
            float(record.queue_length),
            float(record.average_speed),
            float(record.density),
            float(record.occupancy),
            float(record.waiting_time),
            float(record.delay),
            float(record.flow_rate),
            float(record.pressure),
        ]

    # ========================================================
    # RECORD -> FEATURE VECTOR
    # ========================================================

    def record_to_feature_vector(
        self,
        record: TrafficHistoryRecord,
    ) -> TrafficFeatureVector:

        values = self.record_to_features(
            record
        )

        return TrafficFeatureVector(
            vehicle_count=values[0],
            queue_length=values[1],
            average_speed=values[2],
            density=values[3],
            occupancy=values[4],
            waiting_time=values[5],
            delay=values[6],
            flow_rate=values[7],
            pressure=values[8],
        )

    # ========================================================
    # BUILD ONE SEQUENCE
    # ========================================================

    def build_sequence(
        self,
        records: List[TrafficHistoryRecord],
    ) -> TrafficSequence:
        """
        Build one inference sequence from the latest observations.

        The future target is unknown during inference, so y is empty.
        Training datasets should use build_dataset(), which requires
        sequence_length + 1 records.
        """

        if len(records) < self.sequence_length:
            raise ValueError(
                f"At least {self.sequence_length} records are required "
                "for an inference sequence."
            )

        selected = records[
            -self.sequence_length:
        ]

        X = np.asarray(
            [
                self.record_to_features(
                    record
                )
                for record in selected
            ],
            dtype=np.float32,
        )

        y = np.empty(
            (0,),
            dtype=np.float32,
        )

        return TrafficSequence(
            X=X,
            y=y,
        )

    # ========================================================
    # BUILD MANY SEQUENCES
    # ========================================================

    def build_dataset(
        self,
        records: List[TrafficHistoryRecord],
    ) -> Tuple[np.ndarray, np.ndarray]:

        minimum = (
            self.sequence_length + 1
        )

        if len(records) < minimum:
            raise ValueError(
                f"At least {minimum} records are required."
            )

        X_sequences = []
        y_values = []

        for end_index in range(
            self.sequence_length,
            len(records),
        ):

            start_index = (
                end_index
                - self.sequence_length
            )

            window = records[
                start_index:end_index + 1
            ]

            X_records = window[
                :self.sequence_length
            ]

            target_record = window[
                self.sequence_length
            ]

            X_sequences.append(
                [
                    self.record_to_features(
                        record
                    )
                    for record in X_records
                ]
            )

            y_values.append(
                self.record_to_features(
                    target_record
                )
            )

        return (
            np.asarray(
                X_sequences,
                dtype=np.float32,
            ),
            np.asarray(
                y_values,
                dtype=np.float32,
            ),
        )

    # ========================================================
    # FEATURE COUNT
    # ========================================================

    @property
    def feature_count(self) -> int:

        return len(
            self.FEATURE_NAMES
        )
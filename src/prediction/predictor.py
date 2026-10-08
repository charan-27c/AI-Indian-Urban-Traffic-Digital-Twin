from dataclasses import dataclass
from typing import List

import numpy as np

from .models import TrafficHistoryRecord
from .sequence_builder import TrafficSequenceBuilder
from .lstm_model import TrafficLSTMModel


# ============================================================
# TRAFFIC PREDICTION
# ============================================================

@dataclass(frozen=True)
class TrafficPrediction:
    """
    Predicted traffic state for the next time step.
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

    def to_dict(self) -> dict:
        return {
            "vehicle_count": self.vehicle_count,
            "queue_length": self.queue_length,
            "average_speed": self.average_speed,
            "density": self.density,
            "occupancy": self.occupancy,
            "waiting_time": self.waiting_time,
            "delay": self.delay,
            "flow_rate": self.flow_rate,
            "pressure": self.pressure,
        }


# ============================================================
# TRAFFIC PREDICTOR
# ============================================================

class TrafficPredictor:
    """
    Runtime inference service for traffic prediction.

    Pipeline:

        HistoricalTrafficStore
                ↓
        TrafficSequenceBuilder
                ↓
        TrafficLSTMModel
                ↓
        Raw TrafficPrediction
                ↓
        Physical Post-processing
                ↓
        Valid TrafficPrediction

    No fixed junction, lane, approach, or movement is
    assumed here.
    """

    def __init__(
        self,
        model: TrafficLSTMModel,
        sequence_builder: TrafficSequenceBuilder | None = None,
    ) -> None:

        self.model = model

        self.sequence_builder = (
            sequence_builder
            if sequence_builder is not None
            else TrafficSequenceBuilder(
                sequence_length=model.config.sequence_length
            )
        )

        if (
            self.sequence_builder.sequence_length
            != model.config.sequence_length
        ):
            raise ValueError(
                "Sequence builder length must match "
                "the LSTM sequence length."
            )

    # ========================================================
    # PREDICT FROM HISTORICAL RECORDS
    # ========================================================

    def predict(
        self,
        records: List[TrafficHistoryRecord],
    ) -> TrafficPrediction:
        """
        Predict the next traffic state from historical records.

        For inference we use the latest N records.

        Example:

            t1 t2 t3 t4 t5
                     ↓
                    LSTM
                     ↓
                    t6
        """

        sequence = (
            self.sequence_builder.build_sequence(
                records
            )
        )

        prediction_vector = (
            self.model.predict_sequence(
                sequence.X
            )
        )

        raw_prediction = (
            self.vector_to_prediction(
                prediction_vector
            )
        )

        return self._post_process_prediction(
            raw_prediction
        )

    # ========================================================
    # POST-PROCESS PREDICTION
    # ========================================================

    def _post_process_prediction(
        self,
        prediction: TrafficPrediction,
    ) -> TrafficPrediction:
        """
        Convert raw neural-network output into a physically
        meaningful traffic prediction.

        This method is intentionally separate from
        vector_to_prediction(), which preserves raw model output.
        """

        return TrafficPrediction(
            vehicle_count=max(
                0.0,
                prediction.vehicle_count,
            ),

            queue_length=max(
                0.0,
                prediction.queue_length,
            ),

            average_speed=max(
                0.0,
                prediction.average_speed,
            ),

            density=max(
                0.0,
                prediction.density,
            ),

            occupancy=max(
                0.0,
                prediction.occupancy,
            ),

            waiting_time=max(
                0.0,
                prediction.waiting_time,
            ),

            delay=max(
                0.0,
                prediction.delay,
            ),

            flow_rate=max(
                0.0,
                prediction.flow_rate,
            ),

            pressure=max(
                0.0,
                prediction.pressure,
            ),
        )

    # ========================================================
    # PREDICT FROM HISTORICAL STORE
    # ========================================================

    def predict_from_store(
        self,
        store,
    ) -> TrafficPrediction:
        """
        Predict directly from HistoricalTrafficStore.
        """

        if store is None:
            raise ValueError(
                "store must not be None."
            )

        records = store.get_all()

        return self.predict(
            records
        )

    # ========================================================
    # PREDICT FROM ARRAY
    # ========================================================

    def predict_array(
        self,
        X: np.ndarray,
    ) -> TrafficPrediction:
        """
        Predict from one sequence.

        Expected shape:

            (sequence_length, feature_count)
        """

        X = np.asarray(
            X,
            dtype=np.float32,
        )

        prediction = (
            self.model.predict_sequence(
                X
            )
        )

        raw_prediction = (
            self.vector_to_prediction(
                prediction
            )
        )

        return self._post_process_prediction(
            raw_prediction
        )

    # ========================================================
    # VECTOR → PREDICTION
    # ========================================================

    def vector_to_prediction(
        self,
        vector: np.ndarray,
    ) -> TrafficPrediction:
        """
        Convert a 9-element LSTM prediction vector into
        a TrafficPrediction object.

        This method preserves the raw model output.

        Physical/business constraints are applied separately
        by _post_process_prediction().
        """

        vector = np.asarray(
            vector,
            dtype=np.float32,
        )

        if vector.ndim != 1:
            raise ValueError(
                "Prediction vector must be one-dimensional."
            )

        if vector.shape[0] != 9:
            raise ValueError(
                "Prediction vector must contain exactly "
                "9 traffic features."
            )

        return TrafficPrediction(
            vehicle_count=float(vector[0]),
            queue_length=float(vector[1]),
            average_speed=float(vector[2]),
            density=float(vector[3]),
            occupancy=float(vector[4]),
            waiting_time=float(vector[5]),
            delay=float(vector[6]),
            flow_rate=float(vector[7]),
            pressure=float(vector[8]),
        )

    # ========================================================
    # BACKWARD-COMPATIBILITY HELPER
    # ========================================================

    def _to_prediction(
        self,
        values: np.ndarray,
    ) -> TrafficPrediction:
        """
        Internal compatibility wrapper.
        """

        return self.vector_to_prediction(
            values
        )
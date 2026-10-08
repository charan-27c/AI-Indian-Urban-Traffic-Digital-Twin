from __future__ import annotations

from typing import Optional

from .historical_store import HistoricalTrafficStore
from .models import TrafficHistoryRecord
from .predictor import TrafficPrediction, TrafficPredictor


class TrafficPredictionRuntime:
    """
    Runtime coordinator for traffic prediction.

    Responsibilities:

    1. Store incoming historical traffic states.
    2. Wait until enough observations exist.
    3. Trigger LSTM prediction at the configured interval.
    4. Keep the latest prediction.
    5. Allow an explicit prediction request.
    6. Reset runtime prediction state.

    This class is topology-independent.
    It does not know about junctions, lanes, approaches,
    or movement IDs.
    """

    def __init__(
        self,
        store: HistoricalTrafficStore,
        predictor: TrafficPredictor,
        prediction_interval: float = 5.0,
    ) -> None:

        if store is None:
            raise ValueError(
                "store must not be None."
            )

        if predictor is None:
            raise ValueError(
                "predictor must not be None."
            )

        if prediction_interval <= 0:
            raise ValueError(
                "prediction_interval must be greater than zero."
            )

        self.store = store
        self.predictor = predictor
        self.prediction_interval = float(
            prediction_interval
        )

        self.last_prediction: Optional[
            TrafficPrediction
        ] = None

        self._last_prediction_timestamp: Optional[
            float
        ] = None

    # ========================================================
    # UPDATE
    # ========================================================

    def update(
        self,
        record: TrafficHistoryRecord,
    ) -> Optional[TrafficPrediction]:
        """
        Add a new traffic observation.

        A prediction is generated when:

        1. Enough historical records exist.
        2. No prediction has been generated yet, or
           the configured prediction interval has elapsed.

        Returns:
            TrafficPrediction when a prediction is generated.
            None otherwise.
        """

        if record is None:
            raise ValueError(
                "record must not be None."
            )

        self.store.add(record)

        if not self._enough_history():
            return None

        if not self._prediction_due(
            record.timestamp
        ):
            return None

        prediction = self.predict_now()

        self._last_prediction_timestamp = (
            float(record.timestamp)
        )

        return prediction

    # ========================================================
    # HISTORY CHECK
    # ========================================================

    def _enough_history(self) -> bool:
        """
        Determine whether enough records exist for LSTM
        inference.
        """

        required = (
            self.predictor
            .sequence_builder
            .sequence_length
        )

        return self.store.count() >= required

    # ========================================================
    # PREDICTION INTERVAL
    # ========================================================

    def _prediction_due(
        self,
        timestamp: float,
    ) -> bool:
        """
        Determine whether another prediction should run.
        """

        if self._last_prediction_timestamp is None:
            return True

        elapsed = (
            float(timestamp)
            - self._last_prediction_timestamp
        )

        return elapsed >= self.prediction_interval

    # ========================================================
    # PREDICT NOW
    # ========================================================

    def predict_now(
        self,
    ) -> Optional[TrafficPrediction]:
        """
        Generate a prediction immediately from the latest
        historical observations.

        Returns None when insufficient history exists.
        """

        if not self._enough_history():
            return None

        records = self.store.get_all()

        prediction = self.predictor.predict(
            records
        )

        self.last_prediction = prediction

        return prediction

    # ========================================================
    # RESET
    # ========================================================

    def reset(self) -> None:
        """
        Clear historical observations and prediction state.
        """

        self.store.clear()

        self.last_prediction = None

        self._last_prediction_timestamp = None
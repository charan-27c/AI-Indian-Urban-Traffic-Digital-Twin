from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Optional

from .incident_aware_optimizer import MovementCandidate


@dataclass(frozen=True)
class PredictedMovementTraffic:
    """
    Predicted traffic information for one movement.

    Prediction is advisory only.
    Safety decisions remain outside this class.
    """

    movement_id: str

    predicted_pressure: float = 0.0
    predicted_queue_length: float = 0.0
    predicted_waiting_time: float = 0.0


class PredictionAwareCandidateBuilder:
    """
    Convert current movement information + prediction
    into MovementCandidate objects.

    The LSTM prediction is used as an additional ranking
    signal. It never overrides safety constraints.

    No fixed junction or movement IDs are assumed.
    """

    def __init__(
        self,
        prediction_weight: float = 0.5,
        predicted_queue_weight: float = 0.25,
        predicted_waiting_weight: float = 0.10,
    ) -> None:

        if prediction_weight < 0.0:
            raise ValueError(
                "prediction_weight must be >= 0."
            )

        if predicted_queue_weight < 0.0:
            raise ValueError(
                "predicted_queue_weight must be >= 0."
            )

        if predicted_waiting_weight < 0.0:
            raise ValueError(
                "predicted_waiting_weight must be >= 0."
            )

        self.prediction_weight = float(
            prediction_weight
        )

        self.predicted_queue_weight = float(
            predicted_queue_weight
        )

        self.predicted_waiting_weight = float(
            predicted_waiting_weight
        )

    # ============================================================
    # PREDICTED PRESSURE
    # ============================================================

    def calculate_predicted_pressure(
        self,
        prediction: PredictedMovementTraffic,
    ) -> float:

        return max(
            0.0,
            float(prediction.predicted_pressure),
        )

    # ============================================================
    # EFFECTIVE PRESSURE
    # ============================================================

    def calculate_effective_pressure(
        self,
        current_pressure: float,
        prediction: Optional[
            PredictedMovementTraffic
        ] = None,
    ) -> float:
        """
        Combine current and predicted pressure.

        Formula:

            effective_pressure =
                current_pressure
                +
                prediction_weight *
                predicted_pressure

        Additional predicted queue/waiting signals are
        included as small secondary terms.
        """

        current = max(
            0.0,
            float(current_pressure),
        )

        if prediction is None:
            return current

        predicted_pressure = (
            self.calculate_predicted_pressure(
                prediction
            )
        )

        predicted_queue = max(
            0.0,
            float(
                prediction.predicted_queue_length
            ),
        )

        predicted_waiting = max(
            0.0,
            float(
                prediction.predicted_waiting_time
            ),
        )

        return (
            current
            + (
                self.prediction_weight
                * predicted_pressure
            )
            + (
                self.predicted_queue_weight
                * predicted_queue
            )
            + (
                self.predicted_waiting_weight
                * predicted_waiting
            )
        )

    # ============================================================
    # SINGLE CANDIDATE
    # ============================================================

    def build_candidate(
        self,
        movement_id: str,
        *,
        pressure: float = 0.0,
        vehicle_count: int = 0,
        queue_length: int = 0,
        waiting_time: float = 0.0,
        available: bool = True,
        blocked: bool = False,
        emergency: bool = False,
        starvation_wait: float = 0.0,
        prediction: Optional[
            PredictedMovementTraffic
        ] = None,
    ) -> MovementCandidate:

        if not movement_id:
            raise ValueError(
                "movement_id cannot be empty."
            )

        effective_pressure = (
            self.calculate_effective_pressure(
                pressure,
                prediction,
            )
        )

        return MovementCandidate(
            movement_id=movement_id,
            pressure=float(pressure),
            effective_pressure=effective_pressure,
            vehicle_count=max(
                0,
                int(vehicle_count),
            ),
            queue_length=max(
                0,
                int(queue_length),
            ),
            waiting_time=max(
                0.0,
                float(waiting_time),
            ),
            available=bool(available),
            blocked=bool(blocked),
            emergency=bool(emergency),
            starvation_wait=max(
                0.0,
                float(starvation_wait),
            ),
        )

    # ============================================================
    # MULTIPLE MOVEMENTS
    # ============================================================

    def build_candidates(
        self,
        movements: Iterable[Mapping],
        predictions: Optional[
            Mapping[str, PredictedMovementTraffic]
        ] = None,
    ) -> list[MovementCandidate]:
        """
        Build candidates dynamically.

        Each movement mapping may contain:

            movement_id
            pressure
            vehicle_count
            queue_length
            waiting_time
            available
            blocked
            emergency
            starvation_wait

        Predictions are matched using movement_id.
        """

        prediction_map = (
            predictions
            if predictions is not None
            else {}
        )

        candidates = []

        for movement in movements:

            movement_id = str(
                movement["movement_id"]
            )

            prediction = prediction_map.get(
                movement_id
            )

            candidates.append(
                self.build_candidate(
                    movement_id,
                    pressure=movement.get(
                        "pressure",
                        0.0,
                    ),
                    vehicle_count=movement.get(
                        "vehicle_count",
                        0,
                    ),
                    queue_length=movement.get(
                        "queue_length",
                        0,
                    ),
                    waiting_time=movement.get(
                        "waiting_time",
                        0.0,
                    ),
                    available=movement.get(
                        "available",
                        True,
                    ),
                    blocked=movement.get(
                        "blocked",
                        False,
                    ),
                    emergency=movement.get(
                        "emergency",
                        False,
                    ),
                    starvation_wait=movement.get(
                        "starvation_wait",
                        0.0,
                    ),
                    prediction=prediction,
                )
            )

        return candidates
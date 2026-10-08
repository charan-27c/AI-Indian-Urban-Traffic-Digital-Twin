from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Optional


@dataclass(frozen=True)
class ProjectedMovementPrediction:
    movement_id: str
    predicted_pressure: float
    predicted_queue_length: float
    predicted_waiting_time: float


class NetworkPredictionProjector:
    """
    Projects a network-level LSTM prediction onto
    dynamically discovered movements.

    The LSTM currently predicts network traffic, not
    individual movement traffic.

    Therefore the network forecast is converted into
    a relative demand multiplier while preserving the
    current movement pressure distribution.
    """

    def __init__(
        self,
        minimum_ratio: float = 0.50,
        maximum_ratio: float = 2.00,
    ) -> None:
        if minimum_ratio <= 0:
            raise ValueError(
                "minimum_ratio must be greater than zero."
            )

        if maximum_ratio < minimum_ratio:
            raise ValueError(
                "maximum_ratio must be >= minimum_ratio."
            )

        self.minimum_ratio = float(minimum_ratio)
        self.maximum_ratio = float(maximum_ratio)

    @staticmethod
    def _value(
        obj,
        name: str,
        default: float = 0.0,
    ) -> float:
        value = getattr(obj, name, default)

        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    def calculate_pressure_ratio(
        self,
        current_network_state,
        prediction,
    ) -> float:
        """
        Calculate predicted/current network pressure ratio.
        """

        current_pressure = max(
            0.0,
            self._value(
                current_network_state,
                "pressure",
            ),
        )

        predicted_pressure = max(
            0.0,
            self._value(
                prediction,
                "pressure",
            ),
        )

        if current_pressure <= 0.0:
            if predicted_pressure <= 0.0:
                return 1.0

            return self.maximum_ratio

        ratio = (
            predicted_pressure
            / current_pressure
        )

        return max(
            self.minimum_ratio,
            min(
                self.maximum_ratio,
                ratio,
            ),
        )

    def project(
        self,
        movements: Iterable,
        current_network_state,
        prediction,
    ) -> dict[str, ProjectedMovementPrediction]:
        """
        Project network prediction onto every dynamic movement.
        """

        ratio = self.calculate_pressure_ratio(
            current_network_state,
            prediction,
        )

        predicted_queue = max(
            0.0,
            self._value(
                prediction,
                "queue_length",
            ),
        )

        predicted_waiting = max(
            0.0,
            self._value(
                prediction,
                "waiting_time",
            ),
        )

        movement_list = list(movements)

        if not movement_list:
            return {}

        current_total_queue = sum(
            max(
                0.0,
                self._value(
                    movement,
                    "queue_length",
                ),
            )
            for movement in movement_list
        )

        result = {}

        for movement in movement_list:
            movement_id = str(
                getattr(
                    movement,
                    "movement_id",
                    "",
                )
            )

            if not movement_id:
                continue

            current_pressure = max(
                0.0,
                self._value(
                    movement,
                    "pressure",
                ),
            )

            movement_queue = max(
                0.0,
                self._value(
                    movement,
                    "queue_length",
                ),
            )

            if current_total_queue > 0.0:
                queue_share = (
                    movement_queue
                    / current_total_queue
                )
            else:
                queue_share = 0.0

            projected_pressure = (
                current_pressure * ratio
            )

            projected_queue = (
                predicted_queue
                * queue_share
            )

            projected_waiting = (
                predicted_waiting
                * queue_share
            )

            result[movement_id] = (
                ProjectedMovementPrediction(
                    movement_id=movement_id,
                    predicted_pressure=(
                        projected_pressure
                    ),
                    predicted_queue_length=(
                        projected_queue
                    ),
                    predicted_waiting_time=(
                        projected_waiting
                    ),
                )
            )

        return result
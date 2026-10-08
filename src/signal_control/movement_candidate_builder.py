from __future__ import annotations

from typing import Iterable, Mapping, Optional

from .incident_aware_optimizer import MovementCandidate
from .prediction_adapter import (
    PredictionAwareCandidateBuilder,
    PredictedMovementTraffic,
)


class DynamicMovementCandidateBuilder:
    """
    Converts dynamically discovered MovementTrafficState objects
    into MovementCandidate objects for the signal optimizer.

    No fixed junction, approach, lane, or movement IDs are assumed.
    """

    def __init__(
        self,
        prediction_builder: Optional[
            PredictionAwareCandidateBuilder
        ] = None,
    ) -> None:

        self.prediction_builder = (
            prediction_builder
            or PredictionAwareCandidateBuilder()
        )

    # ============================================================
    # SINGLE MOVEMENT
    # ============================================================

    def build_candidate(
        self,
        movement,
        *,
        available: bool = True,
        blocked: bool = False,
        emergency: bool = False,
        starvation_wait: float = 0.0,
        prediction: Optional[
            PredictedMovementTraffic
        ] = None,
    ) -> MovementCandidate:
        """
        Convert one MovementTrafficState into a
        MovementCandidate.

        Optional runtime flags are supplied by other
        pipeline layers.
        """

        movement_id = str(
            movement.movement_id
        )

        if not movement_id:
            raise ValueError(
                "movement_id cannot be empty."
            )

        return self.prediction_builder.build_candidate(
            movement_id,
            pressure=float(
                getattr(
                    movement,
                    "pressure",
                    0.0,
                )
            ),
            vehicle_count=int(
                getattr(
                    movement,
                    "vehicle_count",
                    0,
                )
            ),
            queue_length=int(
                getattr(
                    movement,
                    "queue_length",
                    0,
                )
            ),
            waiting_time=float(
                getattr(
                    movement,
                    "waiting_time",
                    0.0,
                )
            ),
            available=available,
            blocked=blocked,
            emergency=emergency,
            starvation_wait=starvation_wait,
            prediction=prediction,
        )

    # ============================================================
    # MULTIPLE MOVEMENTS
    # ============================================================

    def build_candidates(
        self,
        movements: Iterable,
        *,
        availability: Optional[
            Mapping[str, bool]
        ] = None,
        blocked: Optional[
            Mapping[str, bool]
        ] = None,
        emergency: Optional[
            Mapping[str, bool]
        ] = None,
        starvation_wait: Optional[
            Mapping[str, float]
        ] = None,
        predictions: Optional[
            Mapping[str, PredictedMovementTraffic]
        ] = None,
    ) -> list[MovementCandidate]:
        """
        Convert dynamically discovered movement states
        into optimizer candidates.

        All optional runtime information is keyed by the
        dynamically discovered movement_id.
        """

        availability = availability or {}
        blocked = blocked or {}
        emergency = emergency or {}
        starvation_wait = starvation_wait or {}
        predictions = predictions or {}

        candidates = []

        for movement in movements:

            movement_id = str(
                movement.movement_id
            )

            candidates.append(
                self.build_candidate(
                    movement,
                    available=availability.get(
                        movement_id,
                        True,
                    ),
                    blocked=blocked.get(
                        movement_id,
                        False,
                    ),
                    emergency=emergency.get(
                        movement_id,
                        False,
                    ),
                    starvation_wait=starvation_wait.get(
                        movement_id,
                        0.0,
                    ),
                    prediction=predictions.get(
                        movement_id
                    ),
                )
            )

        return candidates

    # ============================================================
    # NETWORK STATE
    # ============================================================

    def build_from_network_state(
        self,
        network_state,
        *,
        availability: Optional[
            Mapping[str, bool]
        ] = None,
        blocked: Optional[
            Mapping[str, bool]
        ] = None,
        emergency: Optional[
            Mapping[str, bool]
        ] = None,
        starvation_wait: Optional[
            Mapping[str, float]
        ] = None,
        predictions: Optional[
            Mapping[str, PredictedMovementTraffic]
        ] = None,
    ) -> list[MovementCandidate]:
        """
        Extract all dynamically discovered movement states
        from NetworkTrafficState.
        """

        movements = []

        junctions = getattr(
            network_state,
            "junctions",
            {},
        )

        if isinstance(junctions, Mapping):

            junction_values = (
                junctions.values()
            )

        else:

            junction_values = (
                junctions
            )

        for junction in junction_values:

            junction_movements = getattr(
                junction,
                "movements",
                {},
            )

            if isinstance(
                junction_movements,
                Mapping,
            ):

                movements.extend(
                    junction_movements.values()
                )

            else:

                movements.extend(
                    junction_movements
                )

        return self.build_candidates(
            movements,
            availability=availability,
            blocked=blocked,
            emergency=emergency,
            starvation_wait=starvation_wait,
            predictions=predictions,
        )
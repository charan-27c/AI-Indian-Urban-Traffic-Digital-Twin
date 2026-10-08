import pytest

from signal_control.prediction_adapter import (
    PredictionAwareCandidateBuilder,
    PredictedMovementTraffic,
)
from signal_control.incident_aware_optimizer import (
    IncidentAwareSignalOptimizer,
)


def test_prediction_increases_effective_pressure():

    builder = PredictionAwareCandidateBuilder(
        prediction_weight=0.5
    )

    candidate = builder.build_candidate(
        "movement_A",
        pressure=20.0,
        prediction=PredictedMovementTraffic(
            movement_id="movement_A",
            predicted_pressure=40.0,
        ),
    )

    assert candidate.pressure == 20.0
    assert candidate.effective_pressure == 40.0


def test_no_prediction_keeps_current_pressure():

    builder = PredictionAwareCandidateBuilder()

    candidate = builder.build_candidate(
        "movement_A",
        pressure=25.0,
    )

    assert candidate.effective_pressure == 25.0


def test_prediction_never_makes_blocked_movement_safe():

    builder = PredictionAwareCandidateBuilder()

    candidate = builder.build_candidate(
        "movement_A",
        pressure=10.0,
        blocked=True,
        prediction=PredictedMovementTraffic(
            movement_id="movement_A",
            predicted_pressure=1000.0,
        ),
    )

    optimizer = IncidentAwareSignalOptimizer()

    assert not optimizer.is_safe_candidate(
        candidate
    )

    assert optimizer.calculate_score(
        candidate
    ) == float("-inf")


def test_prediction_can_change_optimizer_selection():

    builder = PredictionAwareCandidateBuilder(
        prediction_weight=1.0
    )

    candidates = builder.build_candidates(
        [
            {
                "movement_id": "movement_A",
                "pressure": 30.0,
            },
            {
                "movement_id": "movement_B",
                "pressure": 20.0,
            },
        ],
        predictions={
            "movement_B": PredictedMovementTraffic(
                movement_id="movement_B",
                predicted_pressure=30.0,
            )
        },
    )

    optimizer = IncidentAwareSignalOptimizer()

    decision = optimizer.select_movement(
        candidates
    )

    assert decision.safe
    assert decision.movement_id == "movement_B"


def test_emergency_still_has_priority():

    builder = PredictionAwareCandidateBuilder()

    candidates = builder.build_candidates(
        [
            {
                "movement_id": "normal",
                "pressure": 100.0,
            },
            {
                "movement_id": "emergency",
                "pressure": 1.0,
                "emergency": True,
            },
        ]
    )

    optimizer = IncidentAwareSignalOptimizer()

    decision = optimizer.select_movement(
        candidates
    )

    assert decision.safe
    assert decision.movement_id == "emergency"


def test_blocked_high_prediction_movement_is_rejected():

    builder = PredictionAwareCandidateBuilder(
        prediction_weight=10.0
    )

    candidates = builder.build_candidates(
        [
            {
                "movement_id": "blocked",
                "pressure": 1.0,
                "blocked": True,
            },
            {
                "movement_id": "normal",
                "pressure": 10.0,
            },
        ],
        predictions={
            "blocked": PredictedMovementTraffic(
                movement_id="blocked",
                predicted_pressure=10000.0,
            )
        },
    )

    optimizer = IncidentAwareSignalOptimizer()

    decision = optimizer.select_movement(
        candidates
    )

    assert decision.safe
    assert decision.movement_id == "normal"


def test_negative_prediction_is_clamped():

    builder = PredictionAwareCandidateBuilder()

    candidate = builder.build_candidate(
        "movement_A",
        pressure=20.0,
        prediction=PredictedMovementTraffic(
            movement_id="movement_A",
            predicted_pressure=-100.0,
        ),
    )

    assert candidate.effective_pressure == 20.0


def test_prediction_map_is_dynamic():

    builder = PredictionAwareCandidateBuilder()

    candidates = builder.build_candidates(
        [
            {
                "movement_id": "dynamic_001",
                "pressure": 10.0,
            },
            {
                "movement_id": "dynamic_002",
                "pressure": 10.0,
            },
        ],
        predictions={
            "dynamic_002": PredictedMovementTraffic(
                movement_id="dynamic_002",
                predicted_pressure=50.0,
            )
        },
    )

    assert len(candidates) == 2

    assert (
        candidates[0].movement_id
        == "dynamic_001"
    )

    assert (
        candidates[1].effective_pressure
        > candidates[0].effective_pressure
    )


def test_invalid_prediction_weight():

    with pytest.raises(ValueError):

        PredictionAwareCandidateBuilder(
            prediction_weight=-1.0
        )
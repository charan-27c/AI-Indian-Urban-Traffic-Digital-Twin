from dataclasses import dataclass, field

from signal_control.movement_candidate_builder import (
    DynamicMovementCandidateBuilder,
)
from signal_control.prediction_adapter import (
    PredictedMovementTraffic,
)
from signal_control.incident_aware_optimizer import (
    IncidentAwareSignalOptimizer,
)


@dataclass
class FakeMovement:
    movement_id: str
    vehicle_count: int = 0
    queue_length: int = 0
    average_speed: float = 0.0
    waiting_time: float = 0.0
    delay: float = 0.0
    pressure: float = 0.0


@dataclass
class FakeJunction:
    movements: dict = field(
        default_factory=dict
    )


@dataclass
class FakeNetworkState:
    junctions: dict = field(
        default_factory=dict
    )


def test_build_candidate_from_movement_state():

    builder = (
        DynamicMovementCandidateBuilder()
    )

    movement = FakeMovement(
        movement_id="dynamic_move_001",
        vehicle_count=12,
        queue_length=5,
        waiting_time=20.0,
        pressure=22.0,
    )

    candidate = builder.build_candidate(
        movement
    )

    assert (
        candidate.movement_id
        == "dynamic_move_001"
    )

    assert candidate.vehicle_count == 12
    assert candidate.queue_length == 5
    assert candidate.waiting_time == 20.0
    assert candidate.pressure == 22.0
    assert candidate.effective_pressure == 22.0


def test_dynamic_movement_ids_are_not_hardcoded():

    builder = (
        DynamicMovementCandidateBuilder()
    )

    movements = [
        FakeMovement(
            movement_id="M_ALPHA",
            pressure=10.0,
        ),
        FakeMovement(
            movement_id="M_BETA",
            pressure=20.0,
        ),
        FakeMovement(
            movement_id="M_GAMMA",
            pressure=30.0,
        ),
    ]

    candidates = builder.build_candidates(
        movements
    )

    assert [
        candidate.movement_id
        for candidate in candidates
    ] == [
        "M_ALPHA",
        "M_BETA",
        "M_GAMMA",
    ]


def test_prediction_is_applied_to_dynamic_movement():

    builder = (
        DynamicMovementCandidateBuilder()
    )

    movement = FakeMovement(
        movement_id="M_PREDICTED",
        pressure=10.0,
    )

    candidate = builder.build_candidate(
        movement,
        prediction=PredictedMovementTraffic(
            movement_id="M_PREDICTED",
            predicted_pressure=40.0,
        ),
    )

    assert (
        candidate.effective_pressure
        > candidate.pressure
    )


def test_blocked_dynamic_movement_cannot_be_selected():

    builder = (
        DynamicMovementCandidateBuilder()
    )

    movements = [
        FakeMovement(
            movement_id="BLOCKED_DYNAMIC",
            pressure=1000.0,
        ),
        FakeMovement(
            movement_id="SAFE_DYNAMIC",
            pressure=10.0,
        ),
    ]

    candidates = builder.build_candidates(
        movements,
        blocked={
            "BLOCKED_DYNAMIC": True,
        },
    )

    optimizer = (
        IncidentAwareSignalOptimizer()
    )

    decision = optimizer.select_movement(
        candidates
    )

    assert decision.safe
    assert (
        decision.movement_id
        == "SAFE_DYNAMIC"
    )


def test_emergency_flag_is_preserved():

    builder = (
        DynamicMovementCandidateBuilder()
    )

    movement = FakeMovement(
        movement_id="EMERGENCY_DYNAMIC",
        pressure=1.0,
    )

    candidate = builder.build_candidate(
        movement,
        emergency=True,
    )

    assert candidate.emergency is True


def test_starvation_wait_is_preserved():

    builder = (
        DynamicMovementCandidateBuilder()
    )

    movement = FakeMovement(
        movement_id="STARVING_DYNAMIC",
        pressure=5.0,
    )

    candidate = builder.build_candidate(
        movement,
        starvation_wait=75.0,
    )

    assert (
        candidate.starvation_wait
        == 75.0
    )


def test_network_state_discovers_all_movements():

    builder = (
        DynamicMovementCandidateBuilder()
    )

    junction_a = FakeJunction(
        movements={
            "movement_A": FakeMovement(
                movement_id="movement_A",
                pressure=10.0,
            ),
            "movement_B": FakeMovement(
                movement_id="movement_B",
                pressure=20.0,
            ),
        }
    )

    junction_b = FakeJunction(
        movements={
            "movement_C": FakeMovement(
                movement_id="movement_C",
                pressure=30.0,
            ),
        }
    )

    state = FakeNetworkState(
        junctions={
            "runtime_junction_1": junction_a,
            "runtime_junction_2": junction_b,
        }
    )

    candidates = (
        builder.build_from_network_state(
            state
        )
    )

    assert len(candidates) == 3

    assert {
        candidate.movement_id
        for candidate in candidates
    } == {
        "movement_A",
        "movement_B",
        "movement_C",
    }


def test_network_state_predictions_are_dynamic():

    builder = (
        DynamicMovementCandidateBuilder()
    )

    junction = FakeJunction(
        movements={
            "M1": FakeMovement(
                movement_id="M1",
                pressure=10.0,
            ),
            "M2": FakeMovement(
                movement_id="M2",
                pressure=10.0,
            ),
        }
    )

    state = FakeNetworkState(
        junctions={
            "runtime_junction": junction,
        }
    )

    candidates = (
        builder.build_from_network_state(
            state,
            predictions={
                "M2": PredictedMovementTraffic(
                    movement_id="M2",
                    predicted_pressure=50.0,
                )
            },
        )
    )

    by_id = {
        candidate.movement_id: candidate
        for candidate in candidates
    }

    assert (
        by_id["M2"].effective_pressure
        > by_id["M1"].effective_pressure
    )


def test_empty_network_state_returns_no_candidates():

    builder = (
        DynamicMovementCandidateBuilder()
    )

    state = FakeNetworkState()

    candidates = (
        builder.build_from_network_state(
            state
        )
    )

    assert candidates == []
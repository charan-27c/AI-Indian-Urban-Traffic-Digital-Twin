from src.signal_control.incident_aware_optimizer import (
    MovementCandidate,
    IncidentAwareSignalOptimizer,
)


def test_blocked_movement_is_never_selected():

    optimizer = IncidentAwareSignalOptimizer()

    blocked = MovementCandidate(
        movement_id="M1",
        pressure=500.0,
        effective_pressure=1000.0,
        vehicle_count=50,
        queue_length=20,
        blocked=True,
        available=False,
    )

    normal = MovementCandidate(
        movement_id="M2",
        pressure=20.0,
        effective_pressure=20.0,
        vehicle_count=5,
        queue_length=2,
        blocked=False,
        available=True,
    )

    decision = optimizer.select_movement(
        [blocked, normal]
    )

    assert decision.movement_id == "M2"
    assert decision.safe is True


def test_unblocked_high_pressure_movement_is_selected():

    optimizer = IncidentAwareSignalOptimizer()

    m1 = MovementCandidate(
        movement_id="M1",
        pressure=20.0,
        effective_pressure=20.0,
        available=True,
    )

    m2 = MovementCandidate(
        movement_id="M2",
        pressure=80.0,
        effective_pressure=80.0,
        available=True,
    )

    decision = optimizer.select_movement(
        [m1, m2]
    )

    assert decision.movement_id == "M2"
    assert decision.safe is True


def test_unavailable_movement_is_not_selected():

    optimizer = IncidentAwareSignalOptimizer()

    unavailable = MovementCandidate(
        movement_id="M1",
        pressure=1000.0,
        effective_pressure=1000.0,
        available=False,
    )

    available = MovementCandidate(
        movement_id="M2",
        pressure=10.0,
        effective_pressure=10.0,
        available=True,
    )

    decision = optimizer.select_movement(
        [unavailable, available]
    )

    assert decision.movement_id == "M2"


def test_emergency_gets_priority_when_safe():

    optimizer = IncidentAwareSignalOptimizer()

    normal = MovementCandidate(
        movement_id="M1",
        pressure=100.0,
        effective_pressure=100.0,
        available=True,
    )

    emergency = MovementCandidate(
        movement_id="M2",
        pressure=20.0,
        effective_pressure=20.0,
        emergency=True,
        available=True,
    )

    decision = optimizer.select_movement(
        [normal, emergency]
    )

    assert decision.movement_id == "M2"
    assert decision.reason == "emergency_priority"


def test_blocked_emergency_does_not_override_safety():

    optimizer = IncidentAwareSignalOptimizer()

    emergency = MovementCandidate(
        movement_id="M1",
        pressure=10.0,
        effective_pressure=1000.0,
        emergency=True,
        blocked=True,
        available=False,
    )

    normal = MovementCandidate(
        movement_id="M2",
        pressure=30.0,
        effective_pressure=30.0,
        available=True,
    )

    decision = optimizer.select_movement(
        [emergency, normal]
    )

    assert decision.movement_id == "M2"


def test_starvation_can_increase_priority():

    optimizer = IncidentAwareSignalOptimizer()

    normal = MovementCandidate(
        movement_id="M1",
        pressure=50.0,
        effective_pressure=50.0,
        starvation_wait=0.0,
        available=True,
    )

    starving = MovementCandidate(
        movement_id="M2",
        pressure=40.0,
        effective_pressure=40.0,
        starvation_wait=20.0,
        available=True,
    )

    decision = optimizer.select_movement(
        [normal, starving]
    )

    assert decision.movement_id == "M2"


def test_no_safe_movement_returns_no_decision():

    optimizer = IncidentAwareSignalOptimizer()

    movements = [
        MovementCandidate(
            movement_id="M1",
            pressure=100.0,
            blocked=True,
            available=False,
        ),
        MovementCandidate(
            movement_id="M2",
            pressure=200.0,
            blocked=True,
            available=False,
        ),
    ]

    decision = optimizer.select_movement(
        movements
    )

    assert decision.movement_id is None
    assert decision.safe is False


def test_minimum_green_prevents_normal_switch():

    optimizer = IncidentAwareSignalOptimizer()

    current = MovementCandidate(
        movement_id="M1",
        pressure=50.0,
        effective_pressure=50.0,
        available=True,
    )

    candidate = MovementCandidate(
        movement_id="M2",
        pressure=100.0,
        effective_pressure=100.0,
        available=True,
    )

    should_switch = (
        optimizer.should_switch_from_current(
            current_movement=current,
            candidates=[current, candidate],
            minimum_green_completed=False,
        )
    )

    assert should_switch is False
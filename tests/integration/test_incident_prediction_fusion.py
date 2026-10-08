from dataclasses import dataclass

from signal_control.incident_aware_optimizer import (
    IncidentAwareSignalOptimizer,
    MovementCandidate,
)
from signal_control.incident_prediction_fusion import (
    IncidentPredictionFusion,
)


@dataclass
class FakeImpact:
    capacity_reduction: float = 0.0
    blocked: bool = False
    severity: float = 0.0
    spillback_risk: float = 0.0


def test_no_incident_preserves_pressure():

    fusion = IncidentPredictionFusion()

    candidate = MovementCandidate(
        movement_id="M1",
        pressure=50.0,
        effective_pressure=80.0,
    )

    result = fusion.fuse_candidate(
        candidate
    )

    assert result.effective_pressure == 80.0
    assert result.available is True
    assert result.blocked is False


def test_blocked_incident_marks_candidate_unsafe():

    fusion = IncidentPredictionFusion()

    candidate = MovementCandidate(
        movement_id="M1",
        pressure=50.0,
        effective_pressure=80.0,
    )

    impact = FakeImpact(
        blocked=True,
        capacity_reduction=1.0,
        severity=1.0,
        spillback_risk=1.0,
    )

    result = fusion.fuse_candidate(
        candidate,
        impact=impact,
    )

    assert result.blocked is True
    assert result.available is False


def test_incident_increases_effective_pressure():

    fusion = IncidentPredictionFusion()

    candidate = MovementCandidate(
        movement_id="M1",
        pressure=50.0,
        effective_pressure=80.0,
    )

    impact = FakeImpact(
        capacity_reduction=0.5,
        severity=0.5,
        spillback_risk=0.5,
    )

    result = fusion.fuse_candidate(
        candidate,
        impact=impact,
    )

    assert result.effective_pressure > 80.0
    assert result.blocked is False


def test_prediction_and_incident_are_combined():

    fusion = IncidentPredictionFusion()

    candidate = MovementCandidate(
        movement_id="M1",
        pressure=40.0,
        effective_pressure=100.0,
    )

    impact = FakeImpact(
        capacity_reduction=0.4,
        severity=0.5,
        spillback_risk=0.4,
    )

    result = fusion.fuse_candidate(
        candidate,
        impact=impact,
    )

    assert result.effective_pressure > 100.0


def test_runtime_availability_can_block_candidate():

    fusion = IncidentPredictionFusion()

    candidate = MovementCandidate(
        movement_id="M1",
        pressure=100.0,
        effective_pressure=120.0,
    )

    result = fusion.fuse_candidate(
        candidate,
        available=False,
    )

    assert result.available is False


def test_explicit_blocked_flag_cannot_be_overridden():

    fusion = IncidentPredictionFusion()

    candidate = MovementCandidate(
        movement_id="M1",
        pressure=100.0,
        effective_pressure=120.0,
        blocked=True,
    )

    result = fusion.fuse_candidate(
        candidate,
        blocked=False,
    )

    assert result.blocked is True
    assert result.available is False


def test_optimizer_never_selects_blocked_movement():

    fusion = IncidentPredictionFusion()

    optimizer = IncidentAwareSignalOptimizer()

    blocked_candidate = MovementCandidate(
        movement_id="BLOCKED",
        pressure=500.0,
        effective_pressure=5000.0,
    )

    safe_candidate = MovementCandidate(
        movement_id="SAFE",
        pressure=20.0,
        effective_pressure=20.0,
    )

    impact = FakeImpact(
        blocked=True,
        severity=1.0,
    )

    candidates = fusion.fuse_candidates(
        [
            blocked_candidate,
            safe_candidate,
        ],
        impacts={
            "BLOCKED": impact,
        },
    )

    decision = optimizer.select_movement(
        candidates
    )

    assert decision.movement_id == "SAFE"


def test_multiple_movements_are_fused_dynamically():

    fusion = IncidentPredictionFusion()

    candidates = [
        MovementCandidate(
            movement_id="NORTH_STRAIGHT",
            pressure=30.0,
            effective_pressure=40.0,
        ),
        MovementCandidate(
            movement_id="EAST_LEFT",
            pressure=50.0,
            effective_pressure=60.0,
        ),
        MovementCandidate(
            movement_id="WEST_RIGHT",
            pressure=20.0,
            effective_pressure=25.0,
        ),
    ]

    impacts = {
        "EAST_LEFT": FakeImpact(
            capacity_reduction=0.5,
            severity=0.5,
        )
    }

    result = fusion.fuse_candidates(
        candidates,
        impacts=impacts,
    )

    assert len(result) == 3

    east = next(
        item
        for item in result
        if item.movement_id == "EAST_LEFT"
    )

    assert east.effective_pressure > 60.0
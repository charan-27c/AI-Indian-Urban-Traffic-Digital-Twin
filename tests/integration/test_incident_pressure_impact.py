from src.incidents.pressure_impact import (
    IncidentPressureAdjustment,
)


def test_normal_pressure_is_unchanged():

    adjustment = (
        IncidentPressureAdjustment()
    )

    assert adjustment.adjusted_pressure(
        50.0
    ) == 50.0


def test_capacity_reduction_increases_pressure():

    adjustment = (
        IncidentPressureAdjustment(
            capacity_reduction=0.5
        )
    )

    result = adjustment.adjusted_pressure(
        50.0
    )

    assert result > 50.0


def test_spillback_increases_pressure():

    adjustment = (
        IncidentPressureAdjustment(
            spillback_risk=0.8
        )
    )

    result = adjustment.adjusted_pressure(
        50.0
    )

    assert result > 50.0


def test_severity_increases_pressure():

    adjustment = (
        IncidentPressureAdjustment(
            severity=0.8
        )
    )

    result = adjustment.adjusted_pressure(
        50.0
    )

    assert result > 50.0


def test_blocked_movement_gets_highest_priority():

    adjustment = (
        IncidentPressureAdjustment(
            blocked=True,
            capacity_reduction=1.0,
            spillback_risk=1.0,
            severity=1.0,
        )
    )

    result = adjustment.adjusted_pressure(
        50.0
    )

    assert result == 200.0


def test_zero_pressure_stays_non_negative():

    adjustment = (
        IncidentPressureAdjustment(
            capacity_reduction=1.0,
            spillback_risk=1.0,
            severity=1.0,
        )
    )

    result = adjustment.adjusted_pressure(
        0.0
    )

    assert result >= 0.0
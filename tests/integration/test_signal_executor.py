import pytest

from src.signal_control.signal_executor import (
    SafeSignalExecutor,
    SignalState,
)


def test_initial_state_is_all_red():
    executor = SafeSignalExecutor()

    assert executor.state == SignalState.ALL_RED
    assert executor.active_junction_id is None
    assert executor.active_movement_id is None


def test_green_requires_all_red_clearance():
    executor = SafeSignalExecutor(
        all_red_time=3.0
    )

    result = executor.request_switch(
        "J1",
        "J1_NORTH_STRAIGHT",
        now=0.0,
    )

    assert result is None
    assert executor.state == SignalState.ALL_RED

    result = executor.request_switch(
        "J1",
        "J1_NORTH_STRAIGHT",
        now=1.0,
    )

    assert result is None
    assert executor.state == SignalState.ALL_RED

    result = executor.request_switch(
        "J1",
        "J1_NORTH_STRAIGHT",
        now=3.0,
    )

    assert result is not None
    assert result.state == SignalState.GREEN
    assert executor.active_junction_id == "J1"
    assert executor.active_movement_id == "J1_NORTH_STRAIGHT"


def test_minimum_green_prevents_switch():
    executor = SafeSignalExecutor(
        min_green_time=20.0,
        all_red_time=0.0,
    )

    executor.request_switch(
        "J1",
        "M1",
        now=0.0,
    )

    assert executor.state == SignalState.GREEN

    result = executor.request_switch(
        "J1",
        "M2",
        now=10.0,
    )

    assert result is None
    assert executor.state == SignalState.GREEN
    assert executor.active_movement_id == "M1"


def test_green_to_yellow_after_minimum_green():
    executor = SafeSignalExecutor(
        min_green_time=20.0,
        all_red_time=0.0,
    )

    executor.request_switch(
        "J1",
        "M1",
        now=3.0,
    )

    result = executor.request_switch(
        "J1",
        "M2",
        now=23.0,
    )

    assert result is None
    assert executor.state == SignalState.YELLOW
    assert executor.active_movement_id == "M1"


def test_yellow_to_all_red():
    executor = SafeSignalExecutor(
        min_green_time=20.0,
        yellow_time=3.0,
        all_red_time=3.0,
    )

    executor.request_switch(
        "J1",
        "M1",
        now=3.0,
    )

    executor.request_switch(
        "J1",
        "M2",
        now=23.0,
    )

    assert executor.state == SignalState.YELLOW

    executor.request_switch(
        "J1",
        "M2",
        now=26.0,
    )

    assert executor.state == SignalState.ALL_RED
    assert executor.active_junction_id is None
    assert executor.active_movement_id is None


def test_all_red_before_next_green():
    executor = SafeSignalExecutor(
        min_green_time=20.0,
        yellow_time=3.0,
        all_red_time=3.0,
    )

    executor.request_switch(
        "J1",
        "M1",
        now=3.0,
    )

    executor.request_switch(
        "J1",
        "M2",
        now=23.0,
    )

    executor.request_switch(
        "J1",
        "M2",
        now=26.0,
    )

    assert executor.state == SignalState.ALL_RED

    result = executor.request_switch(
        "J2",
        "M3",
        now=28.0,
    )

    assert result is None
    assert executor.state == SignalState.ALL_RED

    result = executor.request_switch(
        "J2",
        "M3",
        now=29.0,
    )

    assert result is not None
    assert executor.state == SignalState.GREEN
    assert executor.active_junction_id == "J2"
    assert executor.active_movement_id == "M3"


def test_blocked_movement_cannot_become_green():
    executor = SafeSignalExecutor(
        all_red_time=0.0,
    )

    result = executor.request_switch(
        "J1",
        "M1",
        now=0.0,
        movement_available=False,
        movement_blocked=True,
    )

    assert result is None
    assert executor.state == SignalState.ALL_RED


def test_only_one_movement_is_active():
    executor = SafeSignalExecutor(
        min_green_time=20.0,
        yellow_time=3.0,
        all_red_time=3.0,
    )

    executor.request_switch(
        "J1",
        "M1",
        now=3.0,
    )

    assert executor.active_movement_id == "M1"

    executor.request_switch(
        "J1",
        "M2",
        now=13.0,
    )

    assert executor.active_movement_id == "M1"


def test_reset():
    executor = SafeSignalExecutor(
        all_red_time=0.0,
    )

    executor.request_switch(
        "J1",
        "M1",
        now=0.0,
    )

    executor.reset()

    assert executor.state == SignalState.ALL_RED
    assert executor.active_junction_id is None
    assert executor.active_movement_id is None
    assert executor.transitions() == ()


def test_invalid_green_configuration():
    with pytest.raises(ValueError):
        SafeSignalExecutor(
            min_green_time=90.0,
            max_green_time=20.0,
        )
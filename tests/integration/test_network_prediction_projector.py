from dataclasses import dataclass

import pytest

from signal_control.network_prediction_projector import (
    NetworkPredictionProjector,
)


@dataclass
class MockMovement:
    movement_id: str
    pressure: float
    queue_length: float


@dataclass
class MockNetworkState:
    pressure: float


@dataclass
class MockPrediction:
    pressure: float
    queue_length: float
    waiting_time: float


def test_prediction_increases_movement_pressure_proportionally():
    projector = NetworkPredictionProjector()

    network = MockNetworkState(pressure=100.0)
    prediction = MockPrediction(
        pressure=150.0,
        queue_length=30.0,
        waiting_time=60.0,
    )

    movements = [
        MockMovement("M1", pressure=40.0, queue_length=10.0),
        MockMovement("M2", pressure=60.0, queue_length=20.0),
    ]

    result = projector.project(
        movements,
        network,
        prediction,
    )

    assert result["M1"].predicted_pressure == pytest.approx(60.0)
    assert result["M2"].predicted_pressure == pytest.approx(90.0)


def test_prediction_decreases_movement_pressure_proportionally():
    projector = NetworkPredictionProjector()

    network = MockNetworkState(pressure=200.0)
    prediction = MockPrediction(
        pressure=100.0,
        queue_length=20.0,
        waiting_time=40.0,
    )

    movements = [
        MockMovement("M1", pressure=80.0, queue_length=8.0),
        MockMovement("M2", pressure=120.0, queue_length=12.0),
    ]

    result = projector.project(
        movements,
        network,
        prediction,
    )

    assert result["M1"].predicted_pressure == pytest.approx(40.0)
    assert result["M2"].predicted_pressure == pytest.approx(60.0)


def test_zero_current_network_pressure_is_handled_safely():
    projector = NetworkPredictionProjector()

    network = MockNetworkState(pressure=0.0)
    prediction = MockPrediction(
        pressure=50.0,
        queue_length=10.0,
        waiting_time=20.0,
    )

    movements = [
        MockMovement("M1", pressure=0.0, queue_length=5.0),
        MockMovement("M2", pressure=0.0, queue_length=5.0),
    ]

    result = projector.project(
        movements,
        network,
        prediction,
    )

    assert result["M1"].predicted_pressure == pytest.approx(0.0)
    assert result["M2"].predicted_pressure == pytest.approx(0.0)


def test_prediction_ratio_is_clamped_to_configured_maximum():
    projector = NetworkPredictionProjector(
        minimum_ratio=0.5,
        maximum_ratio=2.0,
    )

    network = MockNetworkState(pressure=10.0)
    prediction = MockPrediction(
        pressure=1000.0,
        queue_length=0.0,
        waiting_time=0.0,
    )

    ratio = projector.calculate_pressure_ratio(
        network,
        prediction,
    )

    assert ratio == pytest.approx(2.0)


def test_prediction_ratio_is_clamped_to_configured_minimum():
    projector = NetworkPredictionProjector(
        minimum_ratio=0.5,
        maximum_ratio=2.0,
    )

    network = MockNetworkState(pressure=1000.0)
    prediction = MockPrediction(
        pressure=10.0,
        queue_length=0.0,
        waiting_time=0.0,
    )

    ratio = projector.calculate_pressure_ratio(
        network,
        prediction,
    )

    assert ratio == pytest.approx(0.5)


def test_queue_and_waiting_time_are_projected_using_current_queue_share():
    projector = NetworkPredictionProjector()

    network = MockNetworkState(pressure=100.0)
    prediction = MockPrediction(
        pressure=100.0,
        queue_length=40.0,
        waiting_time=80.0,
    )

    movements = [
        MockMovement("M1", pressure=50.0, queue_length=10.0),
        MockMovement("M2", pressure=50.0, queue_length=30.0),
    ]

    result = projector.project(
        movements,
        network,
        prediction,
    )

    assert result["M1"].predicted_queue_length == pytest.approx(10.0)
    assert result["M2"].predicted_queue_length == pytest.approx(30.0)

    assert result["M1"].predicted_waiting_time == pytest.approx(20.0)
    assert result["M2"].predicted_waiting_time == pytest.approx(60.0)


def test_multiple_dynamic_movements_are_supported():
    projector = NetworkPredictionProjector()

    network = MockNetworkState(pressure=100.0)
    prediction = MockPrediction(
        pressure=120.0,
        queue_length=50.0,
        waiting_time=100.0,
    )

    movements = [
        MockMovement("JUNCTION_A_MOVE_1", 20.0, 5.0),
        MockMovement("JUNCTION_A_MOVE_2", 30.0, 10.0),
        MockMovement("JUNCTION_B_MOVE_1", 50.0, 35.0),
    ]

    result = projector.project(
        movements,
        network,
        prediction,
    )

    assert set(result.keys()) == {
        "JUNCTION_A_MOVE_1",
        "JUNCTION_A_MOVE_2",
        "JUNCTION_B_MOVE_1",
    }

    assert result["JUNCTION_A_MOVE_1"].predicted_pressure == pytest.approx(24.0)
    assert result["JUNCTION_A_MOVE_2"].predicted_pressure == pytest.approx(36.0)
    assert result["JUNCTION_B_MOVE_1"].predicted_pressure == pytest.approx(60.0)


def test_empty_movement_list_returns_empty_result():
    projector = NetworkPredictionProjector()

    network = MockNetworkState(pressure=100.0)
    prediction = MockPrediction(
        pressure=120.0,
        queue_length=50.0,
        waiting_time=100.0,
    )

    result = projector.project(
        [],
        network,
        prediction,
    )

    assert result == {}


def test_invalid_ratio_configuration_is_rejected():
    with pytest.raises(ValueError):
        NetworkPredictionProjector(minimum_ratio=0.0)

    with pytest.raises(ValueError):
        NetworkPredictionProjector(
            minimum_ratio=2.0,
            maximum_ratio=1.0,
        )
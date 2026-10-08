from dataclasses import dataclass

import pytest

from marl.models import (
    JunctionAction,
    JunctionObservation,
    MARLReward,
)
from marl.observation import MARLObservationBuilder
from marl.reward import MARLRewardCalculator


@dataclass
class MockJunction:
    junction_id: str
    vehicle_count: float = 20.0
    queue_length: float = 10.0
    waiting_time: float = 30.0
    average_speed: float = 8.0
    pressure: float = 40.0


def test_junction_observation_vector_has_expected_size():
    observation = JunctionObservation(
        junction_id="DYNAMIC_A",
    )

    vector = observation.to_vector()

    assert len(vector) == 15


def test_junction_observation_preserves_dynamic_id():
    observation = JunctionObservation(
        junction_id="CUSTOM_JUNCTION_99",
    )

    assert observation.junction_id == "CUSTOM_JUNCTION_99"


def test_observation_builder_reads_dynamic_junction():
    builder = MARLObservationBuilder()

    junction = MockJunction(
        junction_id="JUNCTION_X",
        vehicle_count=50,
        queue_length=20,
        waiting_time=80,
        average_speed=6,
        pressure=90,
    )

    observation = builder.build(junction)

    assert observation.junction_id == "JUNCTION_X"
    assert observation.vehicle_count == 50.0
    assert observation.queue_length == 20.0
    assert observation.waiting_time == 80.0
    assert observation.average_speed == 6.0
    assert observation.pressure == 90.0


def test_observation_values_are_non_negative():
    builder = MARLObservationBuilder()

    junction = MockJunction(
        junction_id="JUNCTION_X",
        vehicle_count=-10,
        queue_length=-5,
        waiting_time=-20,
        average_speed=-3,
        pressure=-50,
    )

    observation = builder.build(junction)

    assert observation.vehicle_count == 0.0
    assert observation.queue_length == 0.0
    assert observation.waiting_time == 0.0
    assert observation.average_speed == 0.0
    assert observation.pressure == 0.0


def test_emergency_route_relevance_is_clamped():
    builder = MARLObservationBuilder()

    junction = MockJunction(
        junction_id="JUNCTION_EMERGENCY",
    )

    high = builder.build(
        junction,
        emergency_route_relevance=5.0,
    )

    low = builder.build(
        junction,
        emergency_route_relevance=-5.0,
    )

    assert high.emergency_route_relevance == 1.0
    assert low.emergency_route_relevance == 0.0


def test_incident_severity_is_clamped():
    builder = MARLObservationBuilder()

    junction = MockJunction(
        junction_id="JUNCTION_INCIDENT",
    )

    high = builder.build(
        junction,
        incident_severity=3.0,
    )

    low = builder.build(
        junction,
        incident_severity=-2.0,
    )

    assert high.incident_severity == 1.0
    assert low.incident_severity == 0.0


def test_emergency_features_are_preserved():
    builder = MARLObservationBuilder()

    junction = MockJunction(
        junction_id="JUNCTION_EMERGENCY",
    )

    observation = builder.build(
        junction,
        emergency_vehicle_count=2,
        emergency_distance=150.0,
        emergency_route_relevance=0.9,
    )

    assert observation.emergency_vehicle_count == 2.0
    assert observation.emergency_distance == 150.0
    assert observation.emergency_route_relevance == 0.9


def test_dynamic_multiple_junction_observations():
    builder = MARLObservationBuilder()

    junctions = [
        MockJunction("ALPHA"),
        MockJunction("BETA"),
        MockJunction("GAMMA"),
        MockJunction("CUSTOM_100"),
    ]

    observations = builder.build_all(junctions)

    assert set(observations.keys()) == {
        "ALPHA",
        "BETA",
        "GAMMA",
        "CUSTOM_100",
    }


def test_junction_action_supports_dynamic_movement():
    action = JunctionAction(
        junction_id="JUNCTION_42",
        movement_id="MOVEMENT_17",
        green_extension=5.0,
        emergency_priority=True,
    )

    assert action.junction_id == "JUNCTION_42"
    assert action.movement_id == "MOVEMENT_17"
    assert action.green_extension == 5.0
    assert action.emergency_priority is True


def test_marl_reward_total():
    reward = MARLReward(
        junction_id="JUNCTION_A",
        throughput_reward=10.0,
        queue_penalty=-3.0,
        waiting_penalty=-2.0,
        emergency_reward=5.0,
        coordination_reward=2.0,
        safety_penalty=-1.0,
    )

    assert reward.total == pytest.approx(11.0)


def test_reward_calculator_rewards_throughput():
    calculator = MARLRewardCalculator()

    before = JunctionObservation(
        junction_id="JUNCTION_A",
        vehicle_count=30,
        queue_length=20,
        waiting_time=50,
    )

    after = JunctionObservation(
        junction_id="JUNCTION_A",
        vehicle_count=20,
        queue_length=10,
        waiting_time=30,
    )

    reward = calculator.calculate(
        before,
        after,
    )

    assert reward.throughput_reward > 0
    assert reward.queue_penalty > 0
    assert reward.waiting_penalty > 0


def test_emergency_service_receives_positive_reward():
    calculator = MARLRewardCalculator(
        emergency_weight=10.0,
    )

    before = JunctionObservation(
        junction_id="JUNCTION_A",
        vehicle_count=20,
    )

    after = JunctionObservation(
        junction_id="JUNCTION_A",
        vehicle_count=20,
    )

    reward = calculator.calculate(
        before,
        after,
        emergency_served=True,
    )

    assert reward.emergency_reward == 10.0
    assert reward.total > 0


def test_unsafe_action_receives_safety_penalty():
    calculator = MARLRewardCalculator(
        safety_weight=20.0,
    )

    before = JunctionObservation(
        junction_id="JUNCTION_A",
    )

    after = JunctionObservation(
        junction_id="JUNCTION_A",
    )

    reward = calculator.calculate(
        before,
        after,
        unsafe_action=True,
    )

    assert reward.safety_penalty == -20.0
    assert reward.total < 0


def test_coordination_receives_reward():
    calculator = MARLRewardCalculator(
        coordination_weight=4.0,
    )

    before = JunctionObservation(
        junction_id="JUNCTION_A",
    )

    after = JunctionObservation(
        junction_id="JUNCTION_A",
    )

    reward = calculator.calculate(
        before,
        after,
        coordinated=True,
    )

    assert reward.coordination_reward == 4.0


def test_no_hardcoded_junction_names():
    builder = MARLObservationBuilder()

    junctions = [
        MockJunction("HYD_TRAFFIC_01"),
        MockJunction("HYD_TRAFFIC_02"),
        MockJunction("TELANGANA_JUNCTION_A"),
    ]

    observations = builder.build_all(junctions)

    assert len(observations) == 3

    assert "HYD_TRAFFIC_01" in observations
    assert "HYD_TRAFFIC_02" in observations
    assert "TELANGANA_JUNCTION_A" in observations
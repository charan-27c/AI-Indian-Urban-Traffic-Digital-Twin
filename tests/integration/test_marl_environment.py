from dataclasses import dataclass

from marl.environment import MARLEnvironment
from marl.models import JunctionAction


@dataclass
class MockMovement:
    movement_id: str


@dataclass
class MockJunction:
    junction_id: str
    vehicle_count: float = 20.0
    queue_length: float = 10.0
    waiting_time: float = 20.0
    average_speed: float = 8.0
    pressure: float = 40.0
    movements: tuple = ()


def test_environment_discovers_dynamic_agents():

    junctions = [
        MockJunction(
            "HYD_A",
            movements=(
                MockMovement("M_A1"),
                MockMovement("M_A2"),
            ),
        ),
        MockJunction(
            "HYD_B",
            movements=(
                MockMovement("M_B1"),
            ),
        ),
        MockJunction(
            "CUSTOM_99",
            movements=(
                MockMovement("M_X"),
            ),
        ),
    ]

    environment = MARLEnvironment()

    observations = environment.reset(
        junctions
    )

    assert set(
        environment.agent_ids
    ) == {
        "HYD_A",
        "HYD_B",
        "CUSTOM_99",
    }

    assert set(
        observations.keys()
    ) == {
        "HYD_A",
        "HYD_B",
        "CUSTOM_99",
    }


def test_environment_builds_neighbor_pressure():

    junctions = [
        MockJunction(
            "A",
            pressure=100,
        ),
        MockJunction(
            "B",
            pressure=50,
        ),
    ]

    environment = MARLEnvironment()

    observations = environment.reset(
        junctions
    )

    assert (
        observations["A"].neighbor_pressure
        == 50.0
    )

    assert (
        observations["B"].neighbor_pressure
        == 100.0
    )


def test_action_mask_is_dynamic():

    junction = MockJunction(
        "JUNCTION_X",
        movements=(
            MockMovement("LEFT_X"),
            MockMovement("STRAIGHT_X"),
            MockMovement("RIGHT_X"),
        ),
    )

    environment = MARLEnvironment()

    mask = environment.get_action_mask(
        junction,
        blocked_movements={
            "STRAIGHT_X": True,
        },
    )

    assert mask["LEFT_X"] is True
    assert mask["STRAIGHT_X"] is False
    assert mask["RIGHT_X"] is True


def test_unavailable_movement_is_masked():

    junction = MockJunction(
        "JUNCTION_X",
        movements=(
            MockMovement("M1"),
            MockMovement("M2"),
        ),
    )

    environment = MARLEnvironment()

    mask = environment.get_action_mask(
        junction,
        available_movements={
            "M1": True,
            "M2": False,
        },
    )

    assert mask["M1"] is True
    assert mask["M2"] is False


def test_blocked_learned_action_is_rejected():

    junction = MockJunction(
        "JUNCTION_X",
        movements=(
            MockMovement("M1"),
        ),
    )

    environment = MARLEnvironment()

    action = JunctionAction(
        junction_id="JUNCTION_X",
        movement_id="M1",
    )

    validated = environment.validate_action(
        action,
        action_mask={
            "M1": False,
        },
    )

    assert validated.request_all_red is True
    assert validated.movement_id is None


def test_environment_step_returns_rewards():

    junctions = [
        MockJunction(
            "A",
            vehicle_count=30,
            queue_length=20,
            waiting_time=40,
            pressure=70,
            movements=(
                MockMovement("M1"),
            ),
        ),
    ]

    environment = MARLEnvironment()

    environment.reset(
        junctions
    )

    actions = {
        "A": JunctionAction(
            junction_id="A",
            movement_id="M1",
            green_extension=5.0,
        )
    }

    result = environment.step(
        actions,
        junctions,
        timestamp=5.0,
    )

    assert result.timestamp == 5.0
    assert "A" in result.observations
    assert "A" in result.rewards
    assert "A" in result.actions
    assert result.terminated is False


def test_environment_step_count_increments():

    junctions = [
        MockJunction(
            "A",
            movements=(
                MockMovement("M1"),
            ),
        ),
    ]

    environment = MARLEnvironment(
        max_episode_steps=3
    )

    environment.reset(
        junctions
    )

    action = JunctionAction(
        junction_id="A",
        movement_id="M1",
    )

    environment.step(
        {"A": action},
        junctions,
    )

    assert environment.step_count == 1


def test_environment_truncates_at_episode_limit():

    junctions = [
        MockJunction(
            "A",
            movements=(
                MockMovement("M1"),
            ),
        ),
    ]

    environment = MARLEnvironment(
        max_episode_steps=2
    )

    environment.reset(
        junctions
    )

    action = JunctionAction(
        junction_id="A",
        movement_id="M1",
    )

    result1 = environment.step(
        {"A": action},
        junctions,
    )

    assert result1.truncated is False

    result2 = environment.step(
        {"A": action},
        junctions,
    )

    assert result2.truncated is True


def test_environment_supports_new_junction_without_hardcoding():

    junctions = [
        MockJunction(
            "TELANGANA_NEW_JUNCTION",
            movements=(
                MockMovement(
                    "TELANGANA_MOVEMENT_1"
                ),
                MockMovement(
                    "TELANGANA_MOVEMENT_2"
                ),
            ),
        ),
    ]

    environment = MARLEnvironment()

    observations = environment.reset(
        junctions
    )

    assert (
        "TELANGANA_NEW_JUNCTION"
        in observations
    )

    assert (
        "TELANGANA_NEW_JUNCTION"
        in environment.agent_ids
    )
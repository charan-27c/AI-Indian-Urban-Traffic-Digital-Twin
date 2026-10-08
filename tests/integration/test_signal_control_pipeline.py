from dataclasses import dataclass

from src.signal_control.incident_aware_optimizer import (
    IncidentAwareSignalOptimizer,
    MovementCandidate,
)

from src.signal_control.signal_executor import (
    SafeSignalExecutor,
    SignalState,
)

from src.signal_control.sumo_movement_mapper import (
    SUMOMovementMapper,
)

from src.signal_control.sumo_signal_adapter import (
    SUMOSignalAdapter,
)

from src.signal_control.controller_pipeline import (
    SignalControlPipeline,
)


@dataclass
class FakeMovement:
    movement_id: str
    source_lanes: tuple[str, ...]
    destination_lanes: tuple[str, ...]


@dataclass
class FakeJunction:
    junction_id: str
    movements: tuple[FakeMovement, ...]


@dataclass
class FakeTopology:
    junctions: tuple[FakeJunction, ...]


class FakeTrafficLight:

    def __init__(self):
        self.states = {
            "tls_alpha": "RRRR",
        }

        self.calls = []

        self.controlled_links = {
            "tls_alpha": [
                [
                    ("lane_in_1", "lane_out_1"),
                ],
                [
                    ("lane_in_2", "lane_out_2"),
                ],
                [
                    ("lane_in_3", "lane_out_3"),
                ],
                [
                    ("lane_in_4", "lane_out_4"),
                ],
            ],
        }

    def getIDList(self):
        return list(self.states.keys())

    def getRedYellowGreenState(
        self,
        tls_id,
    ):
        return self.states[tls_id]

    def setRedYellowGreenState(
        self,
        tls_id,
        state,
    ):
        self.states[tls_id] = state
        self.calls.append(
            (tls_id, state)
        )

    def getControlledLinks(
        self,
        tls_id,
    ):
        return self.controlled_links[tls_id]


class FakeTraCI:

    def __init__(self):
        self.trafficlight = (
            FakeTrafficLight()
        )


def build_pipeline():

    topology = FakeTopology(
        junctions=(
            FakeJunction(
                junction_id="dynamic_a",
                movements=(
                    FakeMovement(
                        movement_id="movement_1",
                        source_lanes=(
                            "lane_in_1",
                        ),
                        destination_lanes=(
                            "lane_out_1",
                        ),
                    ),
                    FakeMovement(
                        movement_id="movement_2",
                        source_lanes=(
                            "lane_in_2",
                        ),
                        destination_lanes=(
                            "lane_out_2",
                        ),
                    ),
                ),
            ),
        )
    )

    traci = FakeTraCI()

    mapper = SUMOMovementMapper(
        topology=topology
    )

    mapper.discover_controlled_links(
        traci
    )

    mapper.build_mappings()

    adapter = SUMOSignalAdapter(
        traci
    )

    adapter.discover_traffic_lights()

    optimizer = IncidentAwareSignalOptimizer()

    executor = SafeSignalExecutor(
        min_green_time=20.0,
        yellow_time=3.0,
        all_red_time=3.0,
    )

    pipeline = SignalControlPipeline(
        optimizer=optimizer,
        executor=executor,
        movement_mapper=mapper,
        sumo_adapter=adapter,
    )

    return pipeline, traci


def test_pipeline_waits_for_initial_all_red():

    pipeline, traci = build_pipeline()

    result = pipeline.select_and_control(
        movements=(
            MovementCandidate(
                movement_id="movement_1",
                pressure=100.0,
                vehicle_count=10,
            ),
        ),
        now=1.0,
    )

    assert result.executor_state == (
        SignalState.ALL_RED
    )

    assert result.applied_signal == "RRRR"

    assert traci.trafficlight.states[
        "tls_alpha"
    ] == "RRRR"


def test_pipeline_activates_best_movement():

    pipeline, traci = build_pipeline()

    result = pipeline.select_and_control(
        movements=(
            MovementCandidate(
                movement_id="movement_1",
                pressure=100.0,
                vehicle_count=10,
            ),
            MovementCandidate(
                movement_id="movement_2",
                pressure=20.0,
                vehicle_count=2,
            ),
        ),
        now=3.0,
    )

    assert result.movement_id == "movement_1"

    assert result.tls_id == "tls_alpha"

    assert result.signal_indices == (0,)

    assert result.executor_state == (
        SignalState.GREEN
    )

    assert result.applied_signal == "GRRR"

    assert traci.trafficlight.states[
        "tls_alpha"
    ] == "GRRR"


def test_pipeline_blocks_unsafe_movement():

    pipeline, traci = build_pipeline()

    result = pipeline.select_and_control(
        movements=(
            MovementCandidate(
                movement_id="movement_1",
                pressure=1000.0,
                vehicle_count=100,
                blocked=True,
                available=False,
            ),
            MovementCandidate(
                movement_id="movement_2",
                pressure=10.0,
                vehicle_count=1,
            ),
        ),
        now=3.0,
    )

    assert result.movement_id == "movement_2"

    assert result.executor_state == (
        SignalState.GREEN
    )

    assert result.applied_signal == "RGRR"


def test_pipeline_returns_no_safe_movement():

    pipeline, traci = build_pipeline()

    result = pipeline.select_and_control(
        movements=(
            MovementCandidate(
                movement_id="movement_1",
                pressure=100.0,
                blocked=True,
                available=False,
            ),
            MovementCandidate(
                movement_id="movement_2",
                pressure=90.0,
                blocked=True,
                available=False,
            ),
        ),
        now=3.0,
    )

    assert result.movement_id is None

    assert result.decision is not None
    assert result.decision.movement_id is None
    assert result.decision.safe is False

    assert result.reason == "no_safe_movement"

    assert result.applied_signal is None


def test_pipeline_moves_to_yellow_after_minimum_green():

    pipeline, traci = build_pipeline()

    pipeline.select_and_control(
        movements=(
            MovementCandidate(
                movement_id="movement_1",
                pressure=100.0,
            ),
            MovementCandidate(
                movement_id="movement_2",
                pressure=20.0,
            ),
        ),
        now=3.0,
    )

    result = pipeline.select_and_control(
        movements=(
            MovementCandidate(
                movement_id="movement_1",
                pressure=20.0,
            ),
            MovementCandidate(
                movement_id="movement_2",
                pressure=100.0,
            ),
        ),
        now=23.0,
    )

    assert result.executor_state == (
        SignalState.YELLOW
    )

    assert result.applied_signal == "RYRR"
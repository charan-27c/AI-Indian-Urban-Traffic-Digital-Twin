from pathlib import Path

from simulation.live_topology_runtime import (
    LiveSUMOTopologyRuntime,
)
from traffic_state.engine import TrafficStateEngine


PROJECT_ROOT = Path(__file__).resolve().parents[2]

NETWORK_PATH = (
    PROJECT_ROOT
    / "simulation"
    / "networks"
    / "ai_traffic.net.xml"
)


def _count_topology(junctions):

    approaches = 0
    lanes = 0
    connections = 0
    movements = 0

    for junction in junctions.values():

        approaches += len(
            junction.get_approaches()
        )

        lanes += sum(
            len(approach.get_lanes())
            for approach in junction.get_approaches()
        )

        connections += (
            junction.connection_count()
        )

        movements += (
            junction.movement_count()
        )

    return (
        len(junctions),
        approaches,
        lanes,
        connections,
        movements,
    )


def test_live_topology_discovers_dynamic_network():

    engine = TrafficStateEngine()

    runtime = LiveSUMOTopologyRuntime(
        network_file=NETWORK_PATH,
        traffic_state_engine=engine,
    )

    junctions = runtime.discover()

    assert junctions
    assert isinstance(junctions, dict)

    (
        junction_count,
        approach_count,
        lane_count,
        connection_count,
        movement_count,
    ) = _count_topology(junctions)

    assert junction_count > 0
    assert approach_count > 0
    assert lane_count > 0
    assert connection_count > 0

    # Your dynamic movement discovery must remain active.
    assert movement_count > 0


def test_live_topology_is_registered_with_engine():

    engine = TrafficStateEngine()

    runtime = LiveSUMOTopologyRuntime(
        network_file=NETWORK_PATH,
        traffic_state_engine=engine,
    )

    junctions = runtime.apply_to_engine()

    assert junctions
    assert engine.topology is junctions


def test_topology_contains_dynamic_junctions():

    engine = TrafficStateEngine()

    runtime = LiveSUMOTopologyRuntime(
        network_file=NETWORK_PATH,
        traffic_state_engine=engine,
    )

    junctions = runtime.apply_to_engine()

    assert junctions

    for junction_id, junction in junctions.items():

        assert junction_id
        assert junction is not None

        assert junction.get_approaches()
        assert junction.get_movements()


def test_repeated_refresh_preserves_topology():

    engine = TrafficStateEngine()

    runtime = LiveSUMOTopologyRuntime(
        network_file=NETWORK_PATH,
        traffic_state_engine=engine,
    )

    first = runtime.apply_to_engine()
    second = runtime.refresh()

    assert set(first.keys()) == set(
        second.keys()
    )

    assert second is engine.topology
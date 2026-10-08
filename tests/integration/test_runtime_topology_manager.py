from pathlib import Path

from src.perception.junction.sumo_topology import (
    SumoTopology,
)

from src.perception.junction.runtime_topology_manager import (
    RuntimeTopologyManager,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

NETWORK_PATH = (
    PROJECT_ROOT
    / "simulation"
    / "networks"
    / "ai_traffic.net.xml"
)


def test_runtime_topology_manager():

    provider = SumoTopology(
        str(NETWORK_PATH)
    )

    manager = RuntimeTopologyManager(
        provider
    )

    snapshot = manager.discover()

    assert snapshot.junction_count > 0

    assert snapshot.approach_count > 0

    assert snapshot.lane_count > 0

    assert snapshot.connection_count > 0

    assert snapshot.movement_count > 0

    assert snapshot.version.startswith(
        "topology-"
    )


def test_repeated_discovery_keeps_same_version():

    provider = SumoTopology(
        str(NETWORK_PATH)
    )

    manager = RuntimeTopologyManager(
        provider
    )

    first = manager.discover()

    second = manager.discover()

    assert first.version == second.version


def test_current_snapshot():

    provider = SumoTopology(
        str(NETWORK_PATH)
    )

    manager = RuntimeTopologyManager(
        provider
    )

    discovered = manager.discover()

    current = (
        manager.get_current_snapshot()
    )

    assert (
        current.version
        == discovered.version
    )
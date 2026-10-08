from pathlib import Path

from src.perception.junction.sumo_topology import (
    SumoTopology,
)

from src.perception.junction.topology_version import (
    TopologyVersionManager,
)


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

NETWORK_PATH = (
    PROJECT_ROOT
    / "simulation"
    / "networks"
    / "ai_traffic.net.xml"
)


def test_topology_version_generation():

    assert NETWORK_PATH.exists(), (
        f"SUMO network not found: "
        f"{NETWORK_PATH}"
    )

    # --------------------------------------------------------
    # DISCOVER
    # --------------------------------------------------------

    loader = SumoTopology(
        str(NETWORK_PATH)
    )

    topology = (
        loader.discover_all_junctions()
    )

    assert topology

    # --------------------------------------------------------
    # CREATE VERSION
    # --------------------------------------------------------

    manager = TopologyVersionManager()

    version = manager.create_version(
        topology
    )

    # --------------------------------------------------------
    # VALIDATE
    # --------------------------------------------------------

    assert version.fingerprint

    assert len(
        version.fingerprint
    ) == 64

    assert version.junction_count > 0

    assert version.approach_count > 0

    assert version.lane_count > 0

    assert version.connection_count > 0

    assert len(
        version.short_id
    ) == 12

    # --------------------------------------------------------
    # PRINT
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("TOPOLOGY VERSION")
    print("=" * 70)

    print(
        f"Version      : "
        f"{version.short_id}"
    )

    print(
        f"Fingerprint  : "
        f"{version.fingerprint}"
    )

    print(
        f"Junctions    : "
        f"{version.junction_count}"
    )

    print(
        f"Approaches   : "
        f"{version.approach_count}"
    )

    print(
        f"Lanes        : "
        f"{version.lane_count}"
    )

    print(
        f"Connections   : "
        f"{version.connection_count}"
    )

    print("=" * 70)


def test_identical_topology_has_same_version():

    loader = SumoTopology(
        str(NETWORK_PATH)
    )

    topology = (
        loader.discover_all_junctions()
    )

    manager = TopologyVersionManager()

    version_1 = manager.create_version(
        topology
    )

    version_2 = manager.create_version(
        topology
    )

    assert (
        version_1.fingerprint
        ==
        version_2.fingerprint
    )

    assert (
        version_1.short_id
        ==
        version_2.short_id
    )


def test_changed_topology_produces_new_version():

    loader = SumoTopology(
        str(NETWORK_PATH)
    )

    topology = (
        loader.discover_all_junctions()
    )

    manager = TopologyVersionManager()

    original = manager.create_version(
        topology
    )

    # --------------------------------------------------------
    # Make a deep semantic copy by rebuilding the topology
    # structure through a modified connection representation.
    #
    # We intentionally modify the topology object itself.
    # --------------------------------------------------------

    first_junction = next(
        iter(topology.values())
    )

    connections = (
        first_junction.get_connections()
    )

    if not connections:
        return

    original_connection = connections[0]

    original_to_lane = (
        original_connection.to_lane_id
    )

    original_connection.to_lane_id = (
        original_to_lane
        + "_VERSION_TEST"
    )

    modified = manager.create_version(
        topology
    )

    assert manager.has_changed(
        original,
        modified,
    )

    comparison = manager.compare(
        original,
        modified,
    )

    assert comparison["changed"] is True
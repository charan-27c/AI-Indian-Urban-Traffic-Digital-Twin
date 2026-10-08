from pathlib import Path

from src.perception.junction.sumo_topology import (
    SumoTopology,
)

from src.perception.junction.topology_graph import (
    TopologyGraphBuilder,
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


def test_runtime_topology_graph():

    assert NETWORK_PATH.exists(), (
        f"SUMO network not found: "
        f"{NETWORK_PATH}"
    )

    # --------------------------------------------------------
    # DISCOVER SUMO TOPOLOGY
    # --------------------------------------------------------

    loader = SumoTopology(
        str(NETWORK_PATH)
    )

    topologies = (
        loader.discover_all_junctions()
    )

    assert topologies

    # --------------------------------------------------------
    # BUILD GRAPH
    # --------------------------------------------------------

    builder = TopologyGraphBuilder()

    graph = builder.build(
        topologies
    )

    # --------------------------------------------------------
    # VALIDATE
    # --------------------------------------------------------

    errors = builder.validate()

    assert not errors, (
        "Topology graph validation failed:\n"
        + "\n".join(errors)
    )

    # --------------------------------------------------------
    # BASIC GRAPH VALIDATION
    # --------------------------------------------------------

    summary = graph.get_summary()

    assert summary["junctions"] > 0

    assert summary["approaches"] > 0

    assert summary["lanes"] > 0

    assert summary["movements"] > 0

    assert summary["node_count"] > 0

    assert summary["edge_count"] > 0

    # --------------------------------------------------------
    # MOVEMENTS MUST HAVE SOURCE LANES
    # --------------------------------------------------------

    movements = (
        graph.get_nodes_by_type(
            "movement"
        )
    )

    for movement in movements:

        source_lane = movement.attributes[
            "from_lane_id"
        ]

        assert source_lane

    # --------------------------------------------------------
    # PRINT RESULT
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("RUNTIME TOPOLOGY GRAPH")
    print("=" * 70)

    print(
        f"Nodes       : "
        f"{summary['node_count']}"
    )

    print(
        f"Graph edges : "
        f"{summary['edge_count']}"
    )

    print(
        f"Junctions   : "
        f"{summary['junctions']}"
    )

    print(
        f"Approaches  : "
        f"{summary['approaches']}"
    )

    print(
        f"Lanes       : "
        f"{summary['lanes']}"
    )

    print(
        f"Movements   : "
        f"{summary['movements']}"
    )

    print("=" * 70)
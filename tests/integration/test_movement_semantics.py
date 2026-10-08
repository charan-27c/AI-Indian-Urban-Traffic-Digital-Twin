from pathlib import Path

from src.perception.junction.sumo_topology import (
    SumoTopology,
)

from src.perception.junction.movement_semantics import (
    MovementSemanticAnalyzer,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

NETWORK_PATH = (
    PROJECT_ROOT
    / "simulation"
    / "networks"
    / "ai_traffic.net.xml"
)


def test_dynamic_movement_semantics():

    assert NETWORK_PATH.exists()

    loader = SumoTopology(
        str(NETWORK_PATH)
    )

    junctions = (
        loader.discover_all_junctions()
    )

    assert len(junctions) > 0

    analyzer = (
        MovementSemanticAnalyzer()
    )

    total_movements = 0

    for junction_id, topology in (
        junctions.items()
    ):

        semantics = analyzer.analyze(
            topology
        )

        movements = (
            topology.get_movements()
        )

        assert len(semantics) == len(
            movements
        )

        for movement in movements:

            assert (
                movement.movement_id
                in semantics
            )

            semantic = semantics[
                movement.movement_id
            ]

            assert semantic.movement_id

            assert semantic.from_lane_id

            assert semantic.to_lane_id

            assert semantic.from_approach_id

            assert semantic.to_edge_id

            assert semantic.movement_type

            total_movements += 1

    assert total_movements > 0
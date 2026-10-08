from pathlib import Path

from src.perception.junction.sumo_topology import (
    SumoTopology,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

NETWORK_PATH = (
    PROJECT_ROOT
    / "simulation"
    / "networks"
    / "ai_traffic.net.xml"
)


def test_movements_are_discovered_dynamically():

    provider = SumoTopology(
        str(NETWORK_PATH)
    )

    junctions = (
        provider.discover_all_junctions()
    )

    assert junctions

    for junction_id, topology in (
        junctions.items()
    ):

        assert (
            topology.movement_count()
            == topology.connection_count()
        )

        for movement in (
            topology.get_movements()
        ):

            assert movement.junction_id == (
                junction_id
            )

            assert movement.connection_id in (
                topology.connections
            )

            assert movement.from_lane_id

            assert movement.to_lane_id

            assert movement.from_approach_id in (
                topology.approaches
            )


def test_movement_ids_are_unique():

    provider = SumoTopology(
        str(NETWORK_PATH)
    )

    junctions = (
        provider.discover_all_junctions()
    )

    for topology in junctions.values():

        movement_ids = [
            movement.movement_id
            for movement in (
                topology.get_movements()
            )
        ]

        assert len(movement_ids) == (
            len(set(movement_ids))
        )
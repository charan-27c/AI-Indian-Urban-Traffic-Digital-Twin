from pathlib import Path
import sys


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from src.perception.junction.sumo_topology import (
    SumoTopology,
)


NETWORK_PATH = (
    PROJECT_ROOT
    / "simulation"
    / "networks"
    / "ai_traffic.net.xml"
)


def test_dynamic_topology():

    assert NETWORK_PATH.exists()

    loader = SumoTopology(
        str(NETWORK_PATH)
    )

    junctions = (
        loader.discover_all_junctions()
    )

    # --------------------------------------------------------
    # At least one junction must exist
    # --------------------------------------------------------

    assert len(junctions) > 0

    # --------------------------------------------------------
    # Validate every discovered junction
    # --------------------------------------------------------

    for junction_id, topology in (
        junctions.items()
    ):

        assert junction_id

        # Every junction must have approaches.
        assert (
            topology.approach_count()
            > 0
        )

        # Every junction must have lanes.
        assert (
            topology.lane_count()
            > 0
        )

        # Connections may legitimately be zero
        # for some unusual topology, so we don't
        # force a positive number here.

        # ----------------------------------------------------
        # Validate approaches
        # ----------------------------------------------------

        for approach in (
            topology.get_approaches()
        ):

            assert approach.approach_id

            # Every discovered approach should
            # contain at least one lane.
            assert (
                len(approach.lanes)
                > 0
            )

        # ----------------------------------------------------
        # Validate connections
        # ----------------------------------------------------

        for connection in (
            topology.get_connections()
        ):

            assert connection.connection_id

            assert (
                connection.from_lane_id
            )

            assert (
                connection.to_lane_id
            )

            assert (
                connection.from_edge_id
            )

            assert (
                connection.to_edge_id
            )

        # ----------------------------------------------------
        # Validate movements
        # ----------------------------------------------------

        for movement in (
            topology.get_movements()
        ):

            assert movement.movement_id

            assert (
                movement.from_lane_id
            )

            assert (
                movement.to_lane_id
            )

            assert (
                movement.from_approach_id
            )

            assert (
                movement.to_edge_id
            )

            assert (
                movement.connection_id
            )
from pathlib import Path
import sys


PROJECT_ROOT = (
    Path(__file__).resolve().parents[2]
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT)
    )


from src.perception.junction.sumo_topology import (
    SumoTopology,
)

from src.perception.junction.movement_analyzer import (
    MovementAnalyzer,
)


NETWORK_PATH = (
    PROJECT_ROOT
    / "simulation"
    / "networks"
    / "ai_traffic.net.xml"
)


def test_dynamic_movement_analysis():

    assert NETWORK_PATH.exists()

    topology_loader = SumoTopology(
        str(NETWORK_PATH)
    )

    topologies = (
        topology_loader
        .discover_all_junctions()
    )

    assert topologies

    analyzer = MovementAnalyzer(
        str(NETWORK_PATH)
    )

    total_movements = 0

    print("\n")
    print("=" * 70)
    print("DYNAMIC MOVEMENT ANALYSIS")
    print("=" * 70)

    for junction_id, topology in (
        topologies.items()
    ):

        print("\n" + "-" * 70)

        print(
            f"JUNCTION: {junction_id}"
        )

        for connection in (
            topology.get_connections()
        ):

            movement = (
                analyzer.analyze_connection(
                    junction_id=(
                        junction_id
                    ),
                    movement_id=(
                        connection.connection_id
                    ),
                    from_lane_id=(
                        connection.from_lane_id
                    ),
                    to_lane_id=(
                        connection.to_lane_id
                    ),
                    from_edge_id=(
                        connection.from_edge_id
                    ),
                    to_edge_id=(
                        connection.to_edge_id
                    ),
                )
            )

            total_movements += 1

            print(
                f"\n  {movement.movement_id}"
            )

            print(
                f"    "
                f"{movement.from_lane_id}"
                f" -> "
                f"{movement.to_lane_id}"
            )

            print(
                f"    "
                f"heading_in="
                f"{movement.heading_in}"
            )

            print(
                f"    "
                f"heading_out="
                f"{movement.heading_out}"
            )

            print(
                f"    "
                f"heading_change="
                f"{movement.heading_change}"
            )

            print(
                f"    "
                f"type="
                f"{movement.movement_type}"
            )

    print("\n" + "=" * 70)

    print(
        f"Total analyzed movements: "
        f"{total_movements}"
    )

    print("=" * 70)

    assert total_movements > 0
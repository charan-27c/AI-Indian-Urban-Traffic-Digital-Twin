from pathlib import Path
import sys


# ============================================================
# PROJECT ROOT
# ============================================================

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


# ============================================================
# SUMO NETWORK
# ============================================================

NETWORK_PATH = (
    PROJECT_ROOT
    / "simulation"
    / "networks"
    / "ai_traffic.net.xml"
)


def main():

    print("=" * 75)

    print(
        "PHASE 0 - DYNAMIC SUMO TOPOLOGY DISCOVERY"
    )

    print("=" * 75)

    # --------------------------------------------------------
    # CHECK NETWORK
    # --------------------------------------------------------

    if not NETWORK_PATH.exists():

        raise FileNotFoundError(
            f"SUMO network not found:\n"
            f"{NETWORK_PATH}"
        )

    print(
        f"\nSUMO network:\n"
        f"{NETWORK_PATH}"
    )

    # --------------------------------------------------------
    # LOAD SUMO
    # --------------------------------------------------------

    topology_loader = SumoTopology(
        str(NETWORK_PATH)
    )

    # --------------------------------------------------------
    # DISCOVER ALL JUNCTIONS
    # --------------------------------------------------------

    print(
        "\nDiscovering runtime topology..."
    )

    junctions = (
        topology_loader
        .discover_all_junctions()
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print(
        f"\nDiscovered junctions: "
        f"{len(junctions)}"
    )

    # --------------------------------------------------------
    # PRINT EACH JUNCTION
    # --------------------------------------------------------

    for (
        junction_id,
        topology
    ) in junctions.items():

        print(
            "\n" + "=" * 75
        )

        print(
            f"JUNCTION: {junction_id}"
        )

        print(
            f"Approaches: "
            f"{topology.approach_count()}"
        )

        print(
            f"Lanes: "
            f"{topology.lane_count()}"
        )

        print(
            f"Connections: "
            f"{topology.connection_count()}"
        )

        print(
            f"Movements: "
            f"{topology.movement_count()}"
        )

        # ----------------------------------------------------
        # APPROACHES
        # ----------------------------------------------------

        print(
            "\nINCOMING APPROACHES"
        )

        for approach in (
            topology.get_approaches()
        ):

            print(
                f"\n  Approach: "
                f"{approach.approach_id}"
            )

            print(
                f"  Name: "
                f"{approach.name}"
            )

            print(
                f"  Lanes: "
                f"{len(approach.lanes)}"
            )

            for lane in (
                approach.get_lanes()
            ):

                print(
                    f"      {lane.lane_id}"
                )

        # ----------------------------------------------------
        # CONNECTIONS
        # ----------------------------------------------------

        print(
            "\nLANE CONNECTIONS"
        )

        for connection in (
            topology.get_connections()
        ):

            print(
                f"  "
                f"{connection.from_lane_id}"
                f" -> "
                f"{connection.to_lane_id}"
            )

        # ----------------------------------------------------
        # MOVEMENTS
        # ----------------------------------------------------

        print(
            "\nDYNAMIC MOVEMENTS"
        )

        for movement in (
            topology.get_movements()
        ):

            print(
                f"  {movement.movement_id}: "
                f"{movement.from_lane_id}"
                f" -> "
                f"{movement.to_lane_id}"
                f" | "
                f"approach="
                f"{movement.from_approach_id}"
                f" | "
                f"to_edge="
                f"{movement.to_edge_id}"
            )

    # --------------------------------------------------------
    # FINISHED
    # --------------------------------------------------------

    print(
        "\n" + "=" * 75
    )

    print(
        "DYNAMIC TOPOLOGY DISCOVERY COMPLETED"
    )

    print(
        "=" * 75
    )


if __name__ == "__main__":
    main()
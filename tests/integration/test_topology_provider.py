from pathlib import Path
import sys


# ---------------------------------------------------------
# PROJECT ROOT
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.perception.junction.config_topology import (
    ConfigTopologyProvider,
)


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

CONFIG_PATH = (
    PROJECT_ROOT
    / "config"
    / "junctions.yaml"
)


def main():

    print("=" * 70)
    print("PHASE 0 - TOPOLOGY PROVIDER TEST")
    print("=" * 70)

    # -----------------------------------------------------
    # CHECK CONFIGURATION
    # -----------------------------------------------------

    if not CONFIG_PATH.exists():

        print(
            "\nERROR: Junction configuration not found:"
        )

        print(CONFIG_PATH)

        return

    # -----------------------------------------------------
    # CREATE PROVIDER
    # -----------------------------------------------------

    provider = ConfigTopologyProvider(
        str(CONFIG_PATH)
    )

    # -----------------------------------------------------
    # DISCOVER JUNCTIONS
    # -----------------------------------------------------

    junctions = (
        provider.discover_all_junctions()
    )

    print(
        f"\nDiscovered junctions: "
        f"{len(junctions)}"
    )

    # -----------------------------------------------------
    # DISPLAY TOPOLOGY
    # -----------------------------------------------------

    for junction_id, topology in junctions.items():

        print("\n" + "-" * 60)

        print(
            f"Junction: {junction_id}"
        )

        print(
            f"Approach count: "
            f"{topology.approach_count()}"
        )

        print(
            f"Lane count: "
            f"{topology.lane_count()}"
        )

        print(
            f"Connection count: "
            f"{topology.connection_count()}"
        )

        print("\nApproaches:")

        for approach in (
            topology.get_approaches()
        ):

            print(
                f"  - {approach.approach_id}"
            )

            print(
                f"    Name: {approach.name}"
            )

            print(
                f"    Lanes: "
                f"{len(approach.get_lanes())}"
            )

            for lane in (
                approach.get_lanes()
            ):

                print(
                    f"      - {lane.lane_id}"
                )

        # -------------------------------------------------
        # CONNECTIONS
        # -------------------------------------------------

        if topology.get_connections():

            print("\nConnections:")

            for connection in (
                topology.get_connections()
            ):

                print(
                    f"  - "
                    f"{connection.from_lane_id}"
                    f" -> "
                    f"{connection.to_lane_id}"
                )

        else:

            print(
                "\nConnections: "
                "0 "
                "(expected for configuration provider)"
            )

        # -------------------------------------------------
        # SUMMARY
        # -------------------------------------------------

        print("\nSummary:")

        summary = topology.get_summary()

        print(
            f"  Junction ID: "
            f"{summary['junction_id']}"
        )

        print(
            f"  Approaches: "
            f"{summary['approach_count']}"
        )

        print(
            f"  Lanes: "
            f"{summary['lane_count']}"
        )

        print(
            f"  Connections: "
            f"{summary['connection_count']}"
        )

    # -----------------------------------------------------
    # VERIFICATION
    # -----------------------------------------------------

    print("\n" + "=" * 70)
    print("VERIFICATION")
    print("=" * 70)

    for junction_id, topology in junctions.items():

        print(
            f"{junction_id}: "
            f"{topology.approach_count()} approaches | "
            f"{topology.lane_count()} lanes | "
            f"{topology.connection_count()} connections"
        )

    print("\nTopology provider test completed.")


if __name__ == "__main__":
    main()
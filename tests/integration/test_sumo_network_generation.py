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


from src.digital_twin.network.network_builder import (
    NetworkBuilder,
)

from src.digital_twin.network.sumo_network_generator import (
    SumoNetworkGenerator,
)


# ============================================================
# CONFIGURATION
# ============================================================

CONFIG_PATH = (
    PROJECT_ROOT
    / "config"
    / "network.yaml"
)


def main():

    print("=" * 70)

    print(
        "PHASE 0 - ADVANCED SUMO NETWORK GENERATION"
    )

    print("=" * 70)

    # --------------------------------------------------------
    # CHECK CONFIG
    # --------------------------------------------------------

    if not CONFIG_PATH.exists():

        raise FileNotFoundError(
            f"Network configuration not found:\n"
            f"{CONFIG_PATH}"
        )

    print(
        f"\nConfiguration:\n"
        f"{CONFIG_PATH}"
    )

    # --------------------------------------------------------
    # LOAD NETWORK
    # --------------------------------------------------------

    builder = NetworkBuilder(
        str(CONFIG_PATH)
    )

    # --------------------------------------------------------
    # VALIDATE
    # --------------------------------------------------------

    print(
        "\nValidating network..."
    )

    builder.validate()

    print(
        "Network validation: PASS"
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    summary = (
        builder.get_summary()
    )

    print(
        "\nNetwork:"
    )

    print(
        f"  Name: "
        f"{summary['name']}"
    )

    print(
        f"  Junctions: "
        f"{summary['junction_count']}"
    )

    print(
        f"  Nodes: "
        f"{summary['node_count']}"
    )

    print(
        f"  Edges: "
        f"{summary['edge_count']}"
    )

    # --------------------------------------------------------
    # GENERATOR
    # --------------------------------------------------------

    output_directory = (
        PROJECT_ROOT
        / "simulation"
        / "networks"
    )

    generator = (
        SumoNetworkGenerator(
            output_directory=str(
                output_directory
            )
        )
    )

    # --------------------------------------------------------
    # GENERATE
    # --------------------------------------------------------

    print(
        "\nGenerating SUMO network..."
    )

    network_file = (
        generator.generate_from_builder(
            builder
        )
    )

    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    print(
        "\nSUMO network generated:"
    )

    print(
        network_file
    )

    print(
        f"\nFile size: "
        f"{network_file.stat().st_size} bytes"
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "SUMO NETWORK GENERATION COMPLETED"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()
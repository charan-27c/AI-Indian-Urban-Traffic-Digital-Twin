from pathlib import Path
import sys


PROJECT_ROOT = Path(
    __file__
).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT)
    )


from src.perception.junction.sumo_topology import (
    SumoTopology,
)

from src.perception.junction.topology_validator import (
    TopologyValidator,
)


NETWORK_PATH = (
    PROJECT_ROOT
    / "simulation"
    / "networks"
    / "ai_traffic.net.xml"
)


def test_dynamic_topology_validation():

    assert NETWORK_PATH.exists(), (
        f"SUMO network not found: "
        f"{NETWORK_PATH}"
    )

    topology_loader = SumoTopology(
        str(NETWORK_PATH)
    )

    topologies = (
        topology_loader
        .discover_all_junctions()
    )

    assert topologies

    validator = TopologyValidator()

    result = validator.validate_all(
        topologies
    )

    print("\n")
    print("=" * 70)
    print("DYNAMIC TOPOLOGY VALIDATION")
    print("=" * 70)

    print(
        f"Valid: {result.valid}"
    )

    print(
        f"Junctions: "
        f"{result.checked_junctions}"
    )

    print(
        f"Approaches: "
        f"{result.checked_approaches}"
    )

    print(
        f"Lanes: "
        f"{result.checked_lanes}"
    )

    print(
        f"Connections: "
        f"{result.checked_connections}"
    )

    print(
        f"Movements: "
        f"{result.checked_movements}"
    )

    print("\nWarnings:")

    for warning in result.warnings:
        print(
            f"  WARNING: {warning}"
        )

    print("\nErrors:")

    for error in result.errors:
        print(
            f"  ERROR: {error}"
        )

    print("=" * 70)

    assert result.valid, (
        "\n".join(result.errors)
    )
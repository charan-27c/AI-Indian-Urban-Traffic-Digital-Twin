from pathlib import Path
import sys


# ---------------------------------------------------------
# PROJECT ROOT
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Add project root to Python import path
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.perception.junction.junction_config import JunctionConfig


CONFIG_PATH = (
    PROJECT_ROOT
    / "config"
    / "junctions.yaml"
)


def main():

    print("=" * 60)
    print("PHASE 0 - DYNAMIC JUNCTION CONFIGURATION TEST")
    print("=" * 60)

    # ---------------------------------------------------------
    # LOAD CONFIGURATION
    # ---------------------------------------------------------

    junction_config = JunctionConfig(
        str(CONFIG_PATH)
    )

    print(
        f"\nTotal junctions: "
        f"{junction_config.get_junction_count()}"
    )

    # ---------------------------------------------------------
    # READ ALL JUNCTIONS
    # ---------------------------------------------------------

    for junction_id, junction in (
        junction_config.get_all_junctions().items()
    ):

        print("\n" + "-" * 40)

        print(
            f"Junction ID: "
            f"{junction.junction_id}"
        )

        print(
            f"Name: "
            f"{junction.name}"
        )

        print(
            f"Approaches: "
            f"{junction.get_approaches()}"
        )

        print(
            f"Approach count: "
            f"{junction.approach_count()}"
        )

    # ---------------------------------------------------------
    # VERIFY DIFFERENT JUNCTION SIZES
    # ---------------------------------------------------------

    print("\n" + "=" * 60)
    print("DYNAMIC JUNCTION VERIFICATION")
    print("=" * 60)

    for junction_id, junction in (
        junction_config.get_all_junctions().items()
    ):

        print(
            f"{junction_id}: "
            f"{junction.approach_count()} approaches"
        )

    print("\nTest completed successfully.")


if __name__ == "__main__":
    main()
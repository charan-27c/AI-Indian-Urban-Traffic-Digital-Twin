from pathlib import Path
import sys


# ---------------------------------------------------------
# PROJECT ROOT
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.perception.video.camera_config_loader import (
    CameraConfigLoader,
)

from src.perception.video.camera_validator import (
    CameraConfigValidator,
)

from src.perception.junction.junction_config import (
    JunctionConfig,
)


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

JUNCTION_CONFIG_PATH = (
    PROJECT_ROOT
    / "config"
    / "junctions.yaml"
)

CAMERA_CONFIG_PATH = (
    PROJECT_ROOT
    / "config"
    / "cameras.yaml"
)


def main():

    print("=" * 70)
    print("PHASE 1 - CAMERA CONFIGURATION VALIDATION")
    print("=" * 70)

    # -----------------------------------------------------
    # CHECK FILES
    # -----------------------------------------------------

    if not JUNCTION_CONFIG_PATH.exists():

        print(
            "\nERROR: Junction configuration not found:"
        )

        print(
            JUNCTION_CONFIG_PATH
        )

        return

    if not CAMERA_CONFIG_PATH.exists():

        print(
            "\nERROR: Camera configuration not found:"
        )

        print(
            CAMERA_CONFIG_PATH
        )

        return

    # -----------------------------------------------------
    # LOAD JUNCTION CONFIG
    # -----------------------------------------------------

    junction_config = JunctionConfig(
        str(JUNCTION_CONFIG_PATH)
    )

    # -----------------------------------------------------
    # LOAD CAMERA CONFIG
    # -----------------------------------------------------

    camera_loader = CameraConfigLoader(
        str(CAMERA_CONFIG_PATH)
    )

    cameras = (
        camera_loader.get_all_cameras()
    )

    print(
        f"\nJunctions loaded: "
        f"{len(junction_config.get_all_junctions())}"
    )

    print(
        f"Cameras loaded: "
        f"{len(cameras)}"
    )

    # -----------------------------------------------------
    # CREATE VALIDATOR
    # -----------------------------------------------------

    validator = CameraConfigValidator(
        junction_config
    )

    # -----------------------------------------------------
    # VALIDATE
    # -----------------------------------------------------

    errors = validator.validate_all(
        cameras
    )

    # -----------------------------------------------------
    # RESULT
    # -----------------------------------------------------

    if errors:

        print("\n" + "-" * 70)

        print(
            "VALIDATION FAILED"
        )

        print("-" * 70)

        for error in errors:

            print(
                f"\nERROR: {error}"
            )

        print(
            "\nCamera configuration contains "
            "invalid junction/approach mappings."
        )

        return

    print("\n" + "-" * 70)

    print(
        "VALIDATION PASSED"
    )

    print("-" * 70)

    for camera_id, camera in cameras.items():

        print(
            f"\n{camera_id}"
        )

        print(
            f"  Junction: "
            f"{camera.junction_id}"
        )

        print(
            f"  Approach: "
            f"{camera.approach_id}"
        )

    print("\n" + "=" * 70)

    print(
        "CAMERA CONFIGURATION IS VALID"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()
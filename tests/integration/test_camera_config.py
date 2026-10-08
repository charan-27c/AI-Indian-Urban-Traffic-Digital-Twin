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


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

CONFIG_PATH = (
    PROJECT_ROOT
    / "config"
    / "cameras.yaml"
)


def main():

    print("=" * 70)
    print("PHASE 1 - CAMERA CONFIGURATION TEST")
    print("=" * 70)

    # -----------------------------------------------------
    # CHECK CONFIGURATION
    # -----------------------------------------------------

    if not CONFIG_PATH.exists():

        print(
            "\nERROR: Camera configuration not found:"
        )

        print(CONFIG_PATH)

        return

    # -----------------------------------------------------
    # LOAD CONFIGURATION
    # -----------------------------------------------------

    loader = CameraConfigLoader(
        str(CONFIG_PATH)
    )

    cameras = (
        loader.get_all_cameras()
    )

    print(
        f"\nTotal cameras: "
        f"{len(cameras)}"
    )

    # -----------------------------------------------------
    # DISPLAY CAMERAS
    # -----------------------------------------------------

    print("\n" + "-" * 70)
    print("CAMERAS")
    print("-" * 70)

    for camera_id, camera in cameras.items():

        print(
            f"\nCamera: {camera_id}"
        )

        print(
            f"  Source: "
            f"{camera.source}"
        )

        print(
            f"  Junction: "
            f"{camera.junction_id}"
        )

        print(
            f"  Approach: "
            f"{camera.approach_id}"
        )

        print(
            f"  ROI count: "
            f"{len(camera.get_rois())}"
        )

        # -------------------------------------------------
        # ROIS
        # -------------------------------------------------

        for roi in camera.get_rois():

            print(
                f"    ROI: "
                f"{roi.roi_id}"
            )

            print(
                f"      Points: "
                f"{roi.get_points()}"
            )

    # -----------------------------------------------------
    # TEST JUNCTION MAPPING
    # -----------------------------------------------------

    print("\n" + "-" * 70)
    print("CAMERA → JUNCTION MAPPING")
    print("-" * 70)

    junction_ids = sorted(
        {
            camera.junction_id
            for camera in cameras.values()
            if camera.junction_id
        }
    )

    for junction_id in junction_ids:

        junction_cameras = (
            loader.get_cameras_for_junction(
                junction_id
            )
        )

        print(
            f"\n{junction_id}: "
            f"{len(junction_cameras)} camera(s)"
        )

        for camera_id in (
            junction_cameras
        ):

            camera = (
                junction_cameras[
                    camera_id
                ]
            )

            print(
                f"  {camera_id}"
                f" → "
                f"{camera.approach_id}"
            )

    # -----------------------------------------------------
    # TEST APPROACH MAPPING
    # -----------------------------------------------------

    print("\n" + "-" * 70)
    print("CAMERA → JUNCTION → APPROACH")
    print("-" * 70)

    for camera_id, camera in cameras.items():

        print(
            f"{camera_id}"
            f" → "
            f"{camera.junction_id}"
            f" → "
            f"{camera.approach_id}"
        )

    # -----------------------------------------------------
    # TEST INDIVIDUAL CAMERA
    # -----------------------------------------------------

    if cameras:

        first_camera_id = next(
            iter(cameras)
        )

        camera = loader.get_camera(
            first_camera_id
        )

        print("\n" + "-" * 70)
        print("INDIVIDUAL CAMERA TEST")
        print("-" * 70)

        print(
            camera.get_metadata()
        )

    # -----------------------------------------------------
    # COMPLETED
    # -----------------------------------------------------

    print("\n" + "=" * 70)
    print("CAMERA CONFIGURATION TEST COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()
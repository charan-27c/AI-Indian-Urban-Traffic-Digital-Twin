from pathlib import Path
import sys


# ---------------------------------------------------------
# PROJECT ROOT
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.perception.video.video_input import (
    VideoInput,
)

from src.perception.video.camera_config_loader import (
    CameraConfigLoader,
)

from src.perception.video.roi_validator import (
    ROIValidator,
)


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

CAMERA_CONFIG_PATH = (
    PROJECT_ROOT
    / "config"
    / "cameras.yaml"
)


def main():

    print("=" * 70)
    print("PHASE 1 - ROI VALIDATION TEST")
    print("=" * 70)

    # -----------------------------------------------------
    # LOAD CAMERA CONFIGURATION
    # -----------------------------------------------------

    loader = CameraConfigLoader(
        str(CAMERA_CONFIG_PATH)
    )

    cameras = loader.get_all_cameras()

    if not cameras:

        print(
            "\nERROR: No cameras configured."
        )

        return

    # -----------------------------------------------------
    # VALIDATE EACH CAMERA
    # -----------------------------------------------------

    total_rois = 0
    valid_rois = 0
    invalid_rois = 0

    for camera_id, camera in cameras.items():

        print("\n" + "-" * 70)

        print(
            f"Camera: {camera_id}"
        )

        print(
            f"Junction: "
            f"{camera.junction_id}"
        )

        print(
            f"Approach: "
            f"{camera.approach_id}"
        )

        # -------------------------------------------------
        # RESOLVE SOURCE
        # -------------------------------------------------

        source = Path(
            camera.source
        )

        if not source.is_absolute():

            source = (
                PROJECT_ROOT
                / source
            )

        print(
            f"Source: {source}"
        )

        # -------------------------------------------------
        # OPEN VIDEO
        # -------------------------------------------------

        video = VideoInput(
              str(source),
          camera.camera_id,
      )

        if not video.open():

            print(
                "ERROR: Could not open video."
            )

            continue

        print(
            f"Frame size: "
            f"{video.width} x {video.height}"
        )

        print(
            f"FPS: "
            f"{video.fps}"
        )

        # -------------------------------------------------
        # CREATE ROI VALIDATOR
        # -------------------------------------------------

        validator = ROIValidator(
            width=video.width,
            height=video.height,
        )

        # -------------------------------------------------
        # VALIDATE ROIS
        # -------------------------------------------------

        rois = camera.get_rois()

        print(
            f"ROI count: "
            f"{len(rois)}"
        )

        for roi in rois:

            total_rois += 1

            errors = validator.validate_roi(
                roi
            )

            if errors:

                invalid_rois += 1

                print(
                    f"\n❌ INVALID ROI: "
                    f"{roi.roi_id}"
                )

                for error in errors:

                    print(
                        f"   {error}"
                    )

            else:

                valid_rois += 1

                print(
                    f"\n✅ VALID ROI: "
                    f"{roi.roi_id}"
                )

                print(
                    f"   Points: "
                    f"{roi.get_points()}"
                )

        # -------------------------------------------------
        # RELEASE VIDEO
        # -------------------------------------------------

        video.release()

    # -----------------------------------------------------
    # SUMMARY
    # -----------------------------------------------------

    print("\n" + "=" * 70)
    print("ROI VALIDATION SUMMARY")
    print("=" * 70)

    print(
        f"Total ROIs:   {total_rois}"
    )

    print(
        f"Valid ROIs:   {valid_rois}"
    )

    print(
        f"Invalid ROIs: {invalid_rois}"
    )

    # -----------------------------------------------------
    # FINAL RESULT
    # -----------------------------------------------------

    if invalid_rois > 0:

        print(
            "\n❌ ROI VALIDATION FAILED"
        )

    else:

        print(
            "\n✅ ROI VALIDATION PASSED"
        )

    print("=" * 70)


if __name__ == "__main__":
    main()
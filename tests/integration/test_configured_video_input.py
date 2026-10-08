from pathlib import Path
import sys

import cv2


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
    print("PHASE 1 - CONFIGURED VIDEO INPUT TEST")
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
    # SELECT CAMERA
    # -----------------------------------------------------
    #
    # For the first test, use the first configured camera.
    #
    # Later we can select cameras by:
    #
    #   camera_id
    #   junction_id
    #   approach_id
    #

    camera_id = next(
        iter(cameras)
    )

    camera = loader.get_camera(
        camera_id
    )

    print("\nCamera configuration:")

    print(
        f"  Camera ID: "
        f"{camera.camera_id}"
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
        f"  Source: "
        f"{camera.source}"
    )

    print(
        f"  ROIs: "
        f"{len(camera.get_rois())}"
    )

    # -----------------------------------------------------
    # RESOLVE VIDEO PATH
    # -----------------------------------------------------

    source = Path(
        camera.source
    )

    if not source.is_absolute():

        source = (
            PROJECT_ROOT
            / source
        )

    # -----------------------------------------------------
    # CREATE VIDEO INPUT
    # -----------------------------------------------------

    video = VideoInput(
    str(source),
    camera.camera_id,
   )

    # -----------------------------------------------------
    # OPEN VIDEO
    # -----------------------------------------------------

    if not video.open():

        print(
            "\nERROR: Could not open video:"
        )

        print(source)

        return

    print("\nVideo opened successfully.")

    # -----------------------------------------------------
    # METADATA
    # -----------------------------------------------------

    print("\nVideo metadata:")

    metadata = (
        video.get_metadata()
    )

    for key, value in metadata.items():

        print(
            f"  {key}: {value}"
        )

    # -----------------------------------------------------
    # READ FRAMES
    # -----------------------------------------------------

    frame_number = 0

    while True:

        success, frame = (
            video.read()
        )

        if not success:

            break

        frame_number += 1

        # ---------------------------------------------
        # DISPLAY CAMERA CONTEXT
        # ---------------------------------------------

        text = (
            f"{camera.camera_id} | "
            f"{camera.junction_id} | "
            f"{camera.approach_id}"
        )

        cv2.putText(
            frame,
            text,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )

        cv2.putText(
            frame,
            f"Frame: {frame_number}",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        # ---------------------------------------------
        # DRAW ROI
        # ---------------------------------------------

        for roi in camera.get_rois():

            points = roi.get_points()

            if len(points) >= 2:

                for index in range(
                    len(points)
                ):

                    p1 = points[index]

                    p2 = points[
                        (index + 1)
                        % len(points)
                    ]

                    cv2.line(
                        frame,
                        p1,
                        p2,
                        (0, 255, 255),
                        2,
                    )

        # ---------------------------------------------
        # DISPLAY
        # ---------------------------------------------

        cv2.imshow(
            "Configured CCTV Input",
            frame,
        )

        key = (
            cv2.waitKey(1)
            & 0xFF
        )

        if key == ord("q"):

            break

    # -----------------------------------------------------
    # RELEASE
    # -----------------------------------------------------

    video.release()

    cv2.destroyAllWindows()

    # -----------------------------------------------------
    # STATISTICS
    # -----------------------------------------------------

    print(
        f"\nFrames processed: "
        f"{frame_number}"
    )

    print("\nConfigured video input test completed.")


if __name__ == "__main__":
    main()
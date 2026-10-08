from pathlib import Path

import cv2

from src.perception.video.video_input import VideoInput
from src.perception.video.frame_processor import (
    FrameProcessor,
    resize_frame,
    validate_frame,
)
from src.perception.video.overlay import draw_frame_info


PROJECT_ROOT = Path(__file__).resolve().parents[2]

VIDEO_PATH = (
    PROJECT_ROOT
    / "data"
    / "video"
    / "traffic_test.mp4"
)


def main():

    print("=" * 60)
    print("PHASE 1 - CCTV VIDEO PIPELINE")
    print("=" * 60)

    # ---------------------------------------------------------
    # CAMERA CONFIGURATION
    # ---------------------------------------------------------
    #
    # These values identify where this video belongs.
    #
    # For the current test video, we use TEST values.
    # Later these will come automatically from cameras.yaml.
    #
    camera_id = "CAM_TEST_01"
    junction_id = "TEST_JUNCTION"
    approach = "TEST"

    # ---------------------------------------------------------
    # VIDEO INPUT
    # ---------------------------------------------------------

    video = VideoInput(
        source=str(VIDEO_PATH),
        camera_id=camera_id,
        junction_id=junction_id,
        approach=approach,
    )

    if not video.open():
        print("ERROR: Could not open video.")
        return

    # ---------------------------------------------------------
    # VIDEO METADATA
    # ---------------------------------------------------------

    print("\nCamera Information:")
    print(f"camera_id: {video.camera_id}")
    print(f"junction_id: {video.junction_id}")
    print(f"approach: {video.approach}")

    print("\nVideo Metadata:")

    metadata = video.get_metadata()

    for key, value in metadata.items():
        print(f"{key}: {value}")

    # ---------------------------------------------------------
    # FRAME PROCESSOR
    # ---------------------------------------------------------

    processor = FrameProcessor()
    processor.start()

    frame_number = 0

    # ---------------------------------------------------------
    # VIDEO LOOP
    # ---------------------------------------------------------

    while True:

        success, frame = video.read()

        if not success:
            break

        frame_number += 1

        # -----------------------------------------------------
        # VALIDATE FRAME
        # -----------------------------------------------------

        if not validate_frame(frame):

            print(
                f"Invalid frame: {frame_number}"
            )

            continue

        # -----------------------------------------------------
        # RESIZE FRAME
        # -----------------------------------------------------

        frame = resize_frame(
            frame,
            width=1280,
        )

        # -----------------------------------------------------
        # FRAME PROCESSING
        # -----------------------------------------------------

        frame = processor.process(frame)

        # -----------------------------------------------------
        # DRAW INFORMATION
        # -----------------------------------------------------

        frame = draw_frame_info(
            frame,
            frame_number,
            video.fps,
        )

        # -----------------------------------------------------
        # DISPLAY
        # -----------------------------------------------------

        cv2.imshow(
            "AI Traffic Digital Twin - CCTV",
            frame,
        )

        key = cv2.waitKey(1) & 0xFF

        # Press Q to quit
        if key == ord("q"):
            break

    # ---------------------------------------------------------
    # RELEASE RESOURCES
    # ---------------------------------------------------------

    video.release()

    cv2.destroyAllWindows()

    # ---------------------------------------------------------
    # PROCESSING STATISTICS
    # ---------------------------------------------------------

    print("\nProcessing statistics:")

    stats = processor.get_statistics()

    for key, value in stats.items():
        print(f"{key}: {value}")

    print("\nPhase 1 CCTV test completed.")


if __name__ == "__main__":
    main()
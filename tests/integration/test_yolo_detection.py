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

from src.perception.detection.yolo_detector import (
    YOLODetector,
)


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

VIDEO_PATH = (
    PROJECT_ROOT
    / "data"
    / "video"
    / "traffic_test.mp4"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "yolo11n.pt"
)


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():

    print("=" * 70)
    print("PHASE 2 - YOLO VEHICLE DETECTION TEST")
    print("=" * 70)

    # -----------------------------------------------------
    # CHECK VIDEO
    # -----------------------------------------------------

    if not VIDEO_PATH.exists():

        print(
            "\nERROR: Video not found:"
        )

        print(VIDEO_PATH)

        return

    # -----------------------------------------------------
    # CHECK MODEL
    # -----------------------------------------------------

    if not MODEL_PATH.exists():

        print(
            "\nERROR: YOLO model not found:"
        )

        print(MODEL_PATH)

        print(
            "\nPlace yolo11n.pt inside:"
        )

        print(
            PROJECT_ROOT / "models"
        )

        return

    # -----------------------------------------------------
    # CREATE VIDEO INPUT
    # -----------------------------------------------------

    video = VideoInput(
        str(VIDEO_PATH),
        "YOLO_TEST_CAMERA",
    )

    if not video.open():

        print(
            "\nERROR: Could not open video."
        )

        return

    print("\nVideo opened successfully.")

    print(
        f"Frame size: "
        f"{video.width} x {video.height}"
    )

    print(
        f"FPS: {video.fps}"
    )

    # -----------------------------------------------------
    # CREATE YOLO DETECTOR
    # -----------------------------------------------------

    detector = YOLODetector(
        model_path=str(MODEL_PATH),
        confidence=0.25,
        iou=0.45,
    )

    print(
        "\nYOLO detector initialized."
    )

    # -----------------------------------------------------
    # PROCESS VIDEO
    # -----------------------------------------------------

    frame_number = 0
    total_detections = 0

    while True:

        success, frame = (
            video.read()
        )

        if not success:
            break

        frame_number += 1

        # -------------------------------------------------
        # YOLO DETECTION
        # -------------------------------------------------

        detections = detector.detect(
            frame
        )

        total_detections += len(
            detections
        )

        # -------------------------------------------------
        # DRAW DETECTIONS
        # -------------------------------------------------

        for detection in detections:

            x1 = detection.x1
            y1 = detection.y1
            x2 = detection.x2
            y2 = detection.y2

            label = (
                f"{detection.class_name} "
                f"{detection.confidence:.2f}"
            )

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2,
            )

            cv2.putText(
                frame,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

        # -------------------------------------------------
        # FRAME INFORMATION
        # -------------------------------------------------

        cv2.putText(
            frame,
            f"Frame: {frame_number}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )

        cv2.putText(
            frame,
            f"Detections: {len(detections)}",
            (20, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )

        # -------------------------------------------------
        # DISPLAY
        # -------------------------------------------------

        cv2.imshow(
            "Phase 2 - YOLO Detection",
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

    print("\n" + "=" * 70)
    print("YOLO DETECTION SUMMARY")
    print("=" * 70)

    print(
        f"Frames processed: "
        f"{frame_number}"
    )

    print(
        f"Total detections: "
        f"{total_detections}"
    )

    if frame_number > 0:

        average = (
            total_detections
            / frame_number
        )

        print(
            f"Average detections/frame: "
            f"{average:.2f}"
        )

    print(
        "\nYOLO detection test completed."
    )

    print("=" * 70)


if __name__ == "__main__":
    main()
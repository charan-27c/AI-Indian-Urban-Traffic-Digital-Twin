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

from src.perception.detection.class_filter import (
    DetectionClassFilter,
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
# TRAFFIC CLASSES
# ---------------------------------------------------------

TRAFFIC_CLASSES = [
    "car",
    "motorcycle",
    "bus",
    "truck",
    "bicycle",
]


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():

    print("=" * 70)
    print("PHASE 2 - YOLO + TRAFFIC CLASS FILTER")
    print("=" * 70)

    # -----------------------------------------------------
    # CHECK FILES
    # -----------------------------------------------------

    if not VIDEO_PATH.exists():

        print(
            "\nERROR: Video not found:"
        )

        print(VIDEO_PATH)

        return

    if not MODEL_PATH.exists():

        print(
            "\nERROR: YOLO model not found:"
        )

        print(MODEL_PATH)

        return

    # -----------------------------------------------------
    # VIDEO INPUT
    # -----------------------------------------------------

    video = VideoInput(
        str(VIDEO_PATH),
        "YOLO_FILTER_TEST",
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
    # YOLO
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
    # CLASS FILTER
    # -----------------------------------------------------

    class_filter = DetectionClassFilter(
        allowed_classes=TRAFFIC_CLASSES
    )

    print(
        "\nTraffic classes:"
    )

    for class_name in TRAFFIC_CLASSES:

        print(
            f"  - {class_name}"
        )

    # -----------------------------------------------------
    # PROCESSING
    # -----------------------------------------------------

    frame_number = 0

    total_raw = 0
    total_filtered = 0

    class_counts = {}

    # -----------------------------------------------------
    # VIDEO LOOP
    # -----------------------------------------------------

    while True:

        success, frame = (
            video.read()
        )

        if not success:
            break

        frame_number += 1

        # -------------------------------------------------
        # YOLO
        # -------------------------------------------------

        raw_detections = (
            detector.detect(frame)
        )

        total_raw += len(
            raw_detections
        )

        # -------------------------------------------------
        # FILTER
        # -------------------------------------------------

        traffic_detections = (
            class_filter.filter(
                raw_detections
            )
        )

        total_filtered += len(
            traffic_detections
        )

        # -------------------------------------------------
        # CLASS COUNTS
        # -------------------------------------------------

        for detection in (
            traffic_detections
        ):

            class_name = (
                detection.class_name
            )

            class_counts[class_name] = (
                class_counts.get(
                    class_name,
                    0,
                )
                + 1
            )

        # -------------------------------------------------
        # DRAW ONLY FILTERED OBJECTS
        # -------------------------------------------------

        for detection in (
            traffic_detections
        ):

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
                (
                    x1,
                    max(y1 - 10, 20),
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

        # -------------------------------------------------
        # INFORMATION
        # -------------------------------------------------

        cv2.putText(
            frame,
            f"Frame: {frame_number}",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )

        cv2.putText(
            frame,
            (
                f"Raw: {len(raw_detections)} "
                f"| Traffic: "
                f"{len(traffic_detections)}"
            ),
            (15, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )

        # -------------------------------------------------
        # DISPLAY
        # -------------------------------------------------

        cv2.imshow(
            "Phase 2 - YOLO Traffic Filter",
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
    # SUMMARY
    # -----------------------------------------------------

    print("\n" + "=" * 70)
    print("YOLO + CLASS FILTER SUMMARY")
    print("=" * 70)

    print(
        f"Frames processed: "
        f"{frame_number}"
    )

    print(
        f"Raw detections: "
        f"{total_raw}"
    )

    print(
        f"Traffic detections: "
        f"{total_filtered}"
    )

    if frame_number > 0:

        print(
            f"Average raw/frame: "
            f"{total_raw / frame_number:.2f}"
        )

        print(
            f"Average traffic/frame: "
            f"{total_filtered / frame_number:.2f}"
        )

    print(
        "\nTraffic class counts:"
    )

    if class_counts:

        for class_name, count in sorted(
            class_counts.items()
        ):

            print(
                f"  {class_name}: {count}"
            )

    else:

        print(
            "  No configured traffic "
            "classes detected."
        )

    print("\n" + "=" * 70)
    print(
        "YOLO + CLASS FILTER TEST COMPLETED"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
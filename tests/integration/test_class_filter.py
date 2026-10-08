from pathlib import Path
import sys


# ---------------------------------------------------------
# PROJECT ROOT
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.perception.detection.detector import (
    Detection,
)

from src.perception.detection.class_filter import (
    DetectionClassFilter,
)


def main():

    print("=" * 70)
    print("PHASE 2 - DETECTION CLASS FILTER TEST")
    print("=" * 70)

    # -----------------------------------------------------
    # SAMPLE DETECTIONS
    # -----------------------------------------------------

    detections = [

        Detection(
            class_id=2,
            class_name="car",
            confidence=0.91,
            bbox=(100, 100, 200, 200),
        ),

        Detection(
            class_id=3,
            class_name="motorcycle",
            confidence=0.87,
            bbox=(220, 100, 300, 200),
        ),

        Detection(
            class_id=5,
            class_name="bus",
            confidence=0.94,
            bbox=(320, 100, 500, 300),
        ),

        Detection(
            class_id=0,
            class_name="person",
            confidence=0.88,
            bbox=(520, 100, 570, 250),
        ),
    ]

    print(
        f"\nInput detections: "
        f"{len(detections)}"
    )

    # -----------------------------------------------------
    # CONFIGURE TRAFFIC CLASSES
    # -----------------------------------------------------

    traffic_classes = [
        "car",
        "motorcycle",
        "bus",
        "truck",
        "bicycle",
    ]

    detector_filter = DetectionClassFilter(
        allowed_classes=traffic_classes
    )

    print(
        "\nAllowed classes:"
    )

    for class_name in (
        detector_filter.get_allowed_classes()
    ):

        print(
            f"  - {class_name}"
        )

    # -----------------------------------------------------
    # FILTER
    # -----------------------------------------------------

    filtered = (
        detector_filter.filter(
            detections
        )
    )

    # -----------------------------------------------------
    # DISPLAY
    # -----------------------------------------------------

    print(
        "\nFiltered detections:"
    )

    for detection in filtered:

        print(
            f"  {detection.class_name} "
            f"| confidence="
            f"{detection.confidence:.2f} "
            f"| bbox="
            f"{detection.bbox}"
        )

    # -----------------------------------------------------
    # VERIFICATION
    # -----------------------------------------------------

    print("\n" + "-" * 70)
    print("VERIFICATION")
    print("-" * 70)

    if len(filtered) != 3:

        print(
            "❌ FAILED: Expected 3 "
            "traffic detections."
        )

        return

    if any(
        detection.class_name == "person"
        for detection in filtered
    ):

        print(
            "❌ FAILED: Person was not "
            "filtered out."
        )

        return

    print(
        "✅ Traffic-class filtering passed."
    )

    print("\n" + "=" * 70)
    print("CLASS FILTER TEST COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()
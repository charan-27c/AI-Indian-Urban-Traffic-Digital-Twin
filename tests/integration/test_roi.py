from pathlib import Path

import cv2

from src.perception.video.video_input import VideoInput
from src.perception.roi.roi_manager import ROIManager


ROOT = Path(__file__).resolve().parents[2]

VIDEO = (
    ROOT
    / "data"
    / "video"
    / "traffic_test.mp4"
)


def main():

    video = VideoInput(
        str(VIDEO),
        "J1_CCTV"
    )

    if not video.open():

        print("Unable to open video")

        return

    roi = ROIManager()

    roi.add_roi(
        "intersection",
        [
            [400, 200],
            [1000, 200],
            [1100, 700],
            [300, 700]
        ]
    )

    roi.add_roi(
        "entry_north",
        [
            [600, 0],
            [800, 0],
            [800, 250],
            [600, 250]
        ]
    )

    while True:

        success, frame = video.read()

        if not success:
            break

        frame = roi.draw(frame)

        cv2.imshow(
            "ROI Test",
            frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    video.release()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
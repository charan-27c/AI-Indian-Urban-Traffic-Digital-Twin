from pathlib import Path
from typing import Optional

import cv2


class VideoFrameExtractor:
    """
    Extract sampled frames from a video.

    This component is independent of:
        - junctions
        - approaches
        - lanes
        - movements
        - traffic signals

    It only performs:

        video -> sampled images
    """

    def __init__(
        self,
        video_path: str,
        output_dir: str,
        interval_seconds: float = 1.0,
        image_format: str = "jpg",
        max_frames: Optional[int] = None,
    ):
        self.video_path = Path(video_path)
        self.output_dir = Path(output_dir)

        self.interval_seconds = interval_seconds
        self.image_format = image_format.lower()
        self.max_frames = max_frames

        if self.interval_seconds <= 0:
            raise ValueError(
                "interval_seconds must be greater than 0."
            )

        if self.image_format not in {
            "jpg",
            "jpeg",
            "png",
        }:
            raise ValueError(
                "image_format must be jpg, jpeg, or png."
            )

    def extract(self) -> int:
        """
        Extract frames at the configured interval.

        Returns:
            Number of images successfully written.
        """

        if not self.video_path.exists():
            raise FileNotFoundError(
                f"Video does not exist: "
                f"{self.video_path}"
            )

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        cap = cv2.VideoCapture(
            str(self.video_path)
        )

        if not cap.isOpened():
            raise RuntimeError(
                f"Unable to open video: "
                f"{self.video_path}"
            )

        fps = float(
            cap.get(cv2.CAP_PROP_FPS)
        )

        if fps <= 0:
            cap.release()

            raise RuntimeError(
                "Unable to determine video FPS."
            )

        frame_interval = max(
            1,
            round(
                fps * self.interval_seconds
            ),
        )

        frame_number = 0
        saved_count = 0

        try:

            while True:

                success, frame = cap.read()

                if not success:
                    break

                if (
                    frame_number
                    % frame_interval
                    == 0
                ):

                    filename = (
                        f"frame_{saved_count + 1:06d}."
                        f"{self.image_format}"
                    )

                    output_path = (
                        self.output_dir
                        / filename
                    )

                    success_write = cv2.imwrite(
                        str(output_path),
                        frame,
                    )

                    if not success_write:
                        raise RuntimeError(
                            f"Failed to write: "
                            f"{output_path}"
                        )

                    saved_count += 1

                    if (
                        self.max_frames
                        is not None
                        and saved_count
                        >= self.max_frames
                    ):
                        break

                frame_number += 1

        finally:
            cap.release()

        return saved_count
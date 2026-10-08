from pathlib import Path
from typing import Optional, Union

import cv2


VideoSource = Union[str, int]


class VideoInput:
    """
    Unified video input interface.

    Supported sources:
        - Local video file
        - Webcam index
        - RTSP/IP camera URL

    Camera metadata:
        - camera_id
        - junction_id
        - approach
    """

    def __init__(
        self,
        source: VideoSource,
        camera_id: str = "CAM_01",
        junction_id: Optional[str] = None,
        approach: Optional[str] = None,
    ):
        self.source = source

        # Camera identity / location metadata
        self.camera_id = camera_id
        self.junction_id = junction_id
        self.approach = approach

        self.cap: Optional[cv2.VideoCapture] = None

        self.width = 0
        self.height = 0
        self.fps = 0.0
        self.frame_count = 0

    def open(self) -> bool:
        """Open the video source."""

        # Validate local video files
        if isinstance(self.source, str):

            # Do not treat RTSP/IP camera URLs as local paths
            if not (
                self.source.startswith("rtsp://")
                or self.source.startswith("http://")
                or self.source.startswith("https://")
            ):
                source_path = Path(self.source)

                if not source_path.exists():
                    raise FileNotFoundError(
                        f"Video source does not exist: {source_path}"
                    )

        self.cap = cv2.VideoCapture(self.source)

        if not self.cap.isOpened():
            return False

        self.width = int(
            self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        )

        self.height = int(
            self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        )

        self.fps = float(
            self.cap.get(cv2.CAP_PROP_FPS)
        )

        self.frame_count = int(
            self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        return True

    def read(self):
        """Read the next frame."""

        if self.cap is None:
            raise RuntimeError(
                "Video source is not open. Call open() first."
            )

        success, frame = self.cap.read()

        if not success:
            return False, None

        return True, frame

    def get_metadata(self) -> dict:
        """Return video and camera metadata."""

        return {
            "camera_id": self.camera_id,
            "junction_id": self.junction_id,
            "approach": self.approach,

            "source": str(self.source),

            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "frame_count": self.frame_count,
        }

    def release(self):
        """Release the video source."""

        if self.cap is not None:
            self.cap.release()

        self.cap = None

    def __enter__(self):
        if not self.open():
            raise RuntimeError(
                f"Unable to open video source: {self.source}"
            )

        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback
    ):
        self.release()
import time
import cv2
import numpy as np


def validate_frame(frame) -> bool:

    if frame is None:
        return False

    if not isinstance(frame, np.ndarray):
        return False

    if frame.size == 0:
        return False

    if len(frame.shape) != 3:
        return False

    if frame.shape[2] != 3:
        return False

    return True



def resize_frame(frame, width=1280):
    """
    Resize while maintaining aspect ratio.
    """

    height, original_width = frame.shape[:2]

    if original_width <= width:
        return frame

    scale = width / original_width

    new_height = int(height * scale)

    return cv2.resize(
        frame,
        (width, new_height),
        interpolation=cv2.INTER_AREA,
    )



class FrameProcessor:

    def __init__(self):
        self.processed_frames = 0
        self.start_time = None


    def start(self):
        self.start_time = time.perf_counter()
        self.processed_frames = 0


    def process(self, frame):

        if self.start_time is None:
            self.start()

        if not validate_frame(frame):
            return None

        frame = resize_frame(frame)

        self.processed_frames += 1

        return frame


    def get_statistics(self):

        if self.start_time is None:
            return {
                "processed_frames": 0,
                "processing_fps": 0.0,
            }

        elapsed = time.perf_counter() - self.start_time

        fps = (
            self.processed_frames / elapsed
            if elapsed > 0
            else 0.0
        )

        return {
            "processed_frames": self.processed_frames,
            "processing_fps": round(fps, 2),
        }
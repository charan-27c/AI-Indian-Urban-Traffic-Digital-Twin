from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple


BoundingBox = Tuple[int, int, int, int]


@dataclass
class Detection:
    """
    One object detected in a video frame.

    This class is completely junction-independent.
    """

    class_id: int
    class_name: str
    confidence: float
    bbox: BoundingBox

    @property
    def x1(self) -> int:
        return self.bbox[0]

    @property
    def y1(self) -> int:
        return self.bbox[1]

    @property
    def x2(self) -> int:
        return self.bbox[2]

    @property
    def y2(self) -> int:
        return self.bbox[3]

    @property
    def center(self) -> Tuple[int, int]:
        x1, y1, x2, y2 = self.bbox

        return (
            (x1 + x2) // 2,
            (y1 + y2) // 2,
        )


class ObjectDetector(ABC):
    """
    Generic object detector interface.

    YOLO is one implementation of this interface.
    """

    @abstractmethod
    def detect(
        self,
        frame,
    ) -> List[Detection]:
        """
        Detect objects in one frame.
        """
        raise NotImplementedError
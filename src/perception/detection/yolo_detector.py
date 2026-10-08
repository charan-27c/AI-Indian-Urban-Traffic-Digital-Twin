from pathlib import Path
from typing import List, Optional

from ultralytics import YOLO

from src.perception.detection.detector import (
    Detection,
    ObjectDetector,
)


class YOLODetector(ObjectDetector):
    """
    YOLO-based object detector.

    This detector is completely junction-independent.

    It only performs:

        frame
          ↓
        YOLO
          ↓
        detections

    It does NOT know about:

        - junctions
        - approaches
        - lanes
        - movements
        - traffic signals
    """

    def __init__(
        self,
        model_path: str = "yolo11n.pt",
        confidence: float = 0.25,
        iou: float = 0.45,
        device: Optional[str] = None,
    ):
        self.model_path = Path(
            model_path
        )

        self.confidence = confidence
        self.iou = iou
        self.device = device

        self.model = YOLO(
            str(self.model_path)
        )

        self.class_names = (
            self.model.names
        )

    # -----------------------------------------------------
    # DETECTION
    # -----------------------------------------------------

    def detect(
        self,
        frame,
    ) -> List[Detection]:
        """
        Detect objects in one frame.
        """

        results = self.model.predict(
            source=frame,
            conf=self.confidence,
            iou=self.iou,
            device=self.device,
            verbose=False,
        )

        detections = []

        if not results:
            return detections

        result = results[0]

        if result.boxes is None:
            return detections

        boxes = result.boxes

        for index in range(
            len(boxes)
        ):

            xyxy = (
                boxes.xyxy[index]
                .tolist()
            )

            confidence = float(
                boxes.conf[index]
                .item()
            )

            class_id = int(
                boxes.cls[index]
                .item()
            )

            x1 = int(
                xyxy[0]
            )

            y1 = int(
                xyxy[1]
            )

            x2 = int(
                xyxy[2]
            )

            y2 = int(
                xyxy[3]
            )

            class_name = (
                self.class_names[
                    class_id
                ]
            )

            detection = Detection(
                class_id=class_id,
                class_name=class_name,
                confidence=confidence,
                bbox=(
                    x1,
                    y1,
                    x2,
                    y2,
                ),
            )

            detections.append(
                detection
            )

        return detections
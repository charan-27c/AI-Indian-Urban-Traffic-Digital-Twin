from typing import Iterable, List, Optional, Set

from src.perception.detection.detector import Detection


class DetectionClassFilter:
    """
    Filters YOLO detections by object class.

    This class is completely independent of:

        - junctions
        - approaches
        - lanes
        - movements
        - traffic signals
    """

    def __init__(
        self,
        allowed_classes: Optional[Iterable[str]] = None,
    ):
        if allowed_classes is None:
            self.allowed_classes: Optional[Set[str]] = None
        else:
            self.allowed_classes = {
                name.strip().lower()
                for name in allowed_classes
            }

    # -----------------------------------------------------
    # FILTER
    # -----------------------------------------------------

    def filter(
        self,
        detections: List[Detection],
    ) -> List[Detection]:
        """
        Return only detections belonging to
        the configured classes.

        If allowed_classes is None,
        every detection is returned.
        """

        if self.allowed_classes is None:
            return list(detections)

        filtered = []

        for detection in detections:

            class_name = (
                detection.class_name
                .strip()
                .lower()
            )

            if class_name in self.allowed_classes:

                filtered.append(
                    detection
                )

        return filtered

    # -----------------------------------------------------
    # CHECK ONE CLASS
    # -----------------------------------------------------

    def is_allowed(
        self,
        class_name: str,
    ) -> bool:

        if self.allowed_classes is None:
            return True

        return (
            class_name.strip().lower()
            in self.allowed_classes
        )

    # -----------------------------------------------------
    # GET CONFIGURATION
    # -----------------------------------------------------

    def get_allowed_classes(self):

        if self.allowed_classes is None:
            return None

        return sorted(
            self.allowed_classes
        )
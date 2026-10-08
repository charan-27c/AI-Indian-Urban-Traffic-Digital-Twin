from typing import List, Tuple

from src.perception.video.camera_config import (
    ROI,
)


Point = Tuple[int, int]


class ROIValidator:
    """
    Validates ROI coordinates against the actual
    video frame dimensions.

    No fixed resolution is assumed.
    """

    def __init__(
        self,
        width: int,
        height: int,
    ):
        if width <= 0:
            raise ValueError(
                "Video width must be greater than 0."
            )

        if height <= 0:
            raise ValueError(
                "Video height must be greater than 0."
            )

        self.width = width
        self.height = height

    # -----------------------------------------------------
    # VALIDATE ONE POINT
    # -----------------------------------------------------

    def validate_point(
        self,
        point: Point,
    ) -> bool:

        if len(point) != 2:
            return False

        x, y = point

        return (
            0 <= x < self.width
            and
            0 <= y < self.height
        )

    # -----------------------------------------------------
    # VALIDATE ONE ROI
    # -----------------------------------------------------

    def validate_roi(
        self,
        roi: ROI,
    ) -> List[str]:
        """
        Return validation errors.

        Empty list means the ROI is valid.
        """

        errors = []

        points = roi.get_points()

        # -------------------------------------------------
        # MINIMUM POINTS
        # -------------------------------------------------

        if len(points) < 3:

            errors.append(
                f"ROI '{roi.roi_id}' must contain "
                f"at least 3 points."
            )

            return errors

        # -------------------------------------------------
        # POINT VALIDATION
        # -------------------------------------------------

        for index, point in enumerate(points):

            if not self.validate_point(point):

                x, y = point

                errors.append(
                    f"ROI '{roi.roi_id}' point "
                    f"{index} ({x}, {y}) is outside "
                    f"the frame "
                    f"{self.width}x{self.height}."
                )

        return errors

    # -----------------------------------------------------
    # VALIDATE MULTIPLE ROIS
    # -----------------------------------------------------

    def validate_rois(
        self,
        rois: List[ROI],
    ) -> List[str]:

        errors = []

        for roi in rois:

            errors.extend(
                self.validate_roi(roi)
            )

        return errors
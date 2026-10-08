from dataclasses import dataclass, field
from typing import List, Optional, Tuple


Point = Tuple[int, int]


@dataclass
class ROI:
    """
    Region of interest for one camera.

    Points are stored as:
        [(x1, y1), (x2, y2), ...]
    """

    roi_id: str
    points: List[Point] = field(
        default_factory=list
    )

    def get_points(self) -> List[Point]:
        return list(self.points)


@dataclass
class CameraConfig:
    """
    Configuration for one traffic camera.

    A camera can optionally belong to:
        - a junction
        - an approach
        - an ROI

    Nothing assumes that every junction has
    the same number of cameras.
    """

    camera_id: str

    source: str

    junction_id: Optional[str] = None

    approach_id: Optional[str] = None

    rois: List[ROI] = field(
        default_factory=list
    )

    def add_roi(self, roi: ROI):
        """
        Add an ROI to this camera.
        """

        self.rois.append(roi)

    def get_rois(self) -> List[ROI]:
        """
        Return all ROIs belonging to this camera.
        """

        return list(self.rois)

    def get_metadata(self) -> dict:
        """
        Return camera metadata.
        """

        return {
            "camera_id": self.camera_id,
            "source": self.source,
            "junction_id": self.junction_id,
            "approach_id": self.approach_id,
            "roi_count": len(self.rois),
        }
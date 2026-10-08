import cv2
import numpy as np


class ROIManager:

    def __init__(self, polygons=None):
        self.polygons = polygons or {}

    def add_roi(self, name, points):
        self.polygons[name] = np.array(
            points,
            dtype=np.int32
        )

    def point_inside(self, point, roi_name):

        if roi_name not in self.polygons:
            return False

        polygon = self.polygons[roi_name]

        result = cv2.pointPolygonTest(
            polygon,
            point,
            False
        )

        return result >= 0

    def draw(self, frame):

        for name, polygon in self.polygons.items():

            cv2.polylines(
                frame,
                [polygon],
                True,
                (255, 255, 0),
                2
            )

            x, y = polygon[0]

            cv2.putText(
                frame,
                name,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2
            )

        return frame
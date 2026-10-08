import math
from typing import Optional, Tuple

import sumolib

from src.perception.junction.movement import (
    Movement,
)


class MovementAnalyzer:
    """
    Determines movement characteristics from SUMO geometry.

    The analyzer does NOT assume:

        - NORTH
        - SOUTH
        - EAST
        - WEST
        - fixed junctions
        - fixed approaches
        - fixed lane counts
        - fixed movement IDs

    Movement classification is derived from geometry.
    """

    # --------------------------------------------------------
    # ANGLE THRESHOLDS
    # --------------------------------------------------------

    STRAIGHT_THRESHOLD = 30.0

    UTURN_THRESHOLD = 150.0

    RIGHT_THRESHOLD_MIN = 30.0

    RIGHT_THRESHOLD_MAX = 150.0

    LEFT_THRESHOLD_MIN = -150.0

    LEFT_THRESHOLD_MAX = -30.0

    def __init__(self, network_file: str):

        self.network_file = network_file

        self.net = sumolib.net.readNet(
            network_file
        )

    # ========================================================
    # ANGLE NORMALIZATION
    # ========================================================

    @staticmethod
    def normalize_angle(
        angle: float,
    ) -> float:
        """
        Normalize angle to:

            [-180, 180]
        """

        while angle > 180.0:
            angle -= 360.0

        while angle <= -180.0:
            angle += 360.0

        return angle

    # ========================================================
    # GEOMETRY HEADING
    # ========================================================

    @staticmethod
    def calculate_heading(
        points: Tuple[Tuple[float, float], ...],
        from_start: bool = True,
    ) -> Optional[float]:
        """
        Calculate heading from the first/last useful
        pair of geometry points.

        SUMO geometry uses Cartesian coordinates.
        """

        if len(points) < 2:
            return None

        if from_start:

            x1, y1 = points[0]
            x2, y2 = points[1]

        else:

            x1, y1 = points[-2]
            x2, y2 = points[-1]

        dx = x2 - x1
        dy = y2 - y1

        if abs(dx) < 1e-9 and abs(dy) < 1e-9:
            return None

        angle = math.degrees(
            math.atan2(
                dy,
                dx,
            )
        )

        return angle

    # ========================================================
    # LANE HEADING
    # ========================================================

    def get_lane_heading(
        self,
        lane,
        from_start: bool = True,
    ) -> Optional[float]:

        shape = lane.getShape()

        if not shape:
            return None

        return self.calculate_heading(
            tuple(shape),
            from_start=from_start,
        )

    # ========================================================
    # MOVEMENT TYPE
    # ========================================================

    def classify_movement(
        self,
        heading_change: float,
    ) -> str:
        """
        Classify a movement from heading change.

        Positive = one turning direction.
        Negative = opposite turning direction.

        We deliberately do not call these directions
        NORTH/SOUTH/EAST/WEST.

        Classification is based only on geometry.
        """

        absolute_change = abs(
            heading_change
        )

        # ----------------------------------------------------
        # STRAIGHT
        # ----------------------------------------------------

        if (
            absolute_change
            <= self.STRAIGHT_THRESHOLD
        ):

            return "straight"

        # ----------------------------------------------------
        # U-TURN
        # ----------------------------------------------------

        if (
            absolute_change
            >= self.UTURN_THRESHOLD
        ):

            return "u_turn"

        # ----------------------------------------------------
        # TURN
        # ----------------------------------------------------

        if (
            heading_change > 0
        ):

            return "turn_positive"

        return "turn_negative"

    # ========================================================
    # ANALYZE CONNECTION
    # ========================================================

    def analyze_connection(
        self,
        junction_id: str,
        movement_id: str,
        from_lane_id: str,
        to_lane_id: str,
        from_edge_id: str,
        to_edge_id: str,
    ) -> Movement:

        from_lane = self.net.getLane(
            from_lane_id
        )

        to_lane = self.net.getLane(
            to_lane_id
        )

        # ----------------------------------------------------
        # INCOMING HEADING
        # ----------------------------------------------------

        heading_in = self.get_lane_heading(
            from_lane,
            from_start=False,
        )

        # ----------------------------------------------------
        # OUTGOING HEADING
        # ----------------------------------------------------

        heading_out = self.get_lane_heading(
            to_lane,
            from_start=True,
        )

        # ----------------------------------------------------
        # HEADING DIFFERENCE
        # ----------------------------------------------------

        heading_change = None

        movement_type = "unknown"

        if (
            heading_in is not None
            and heading_out is not None
        ):

            heading_change = (
                heading_out
                - heading_in
            )

            heading_change = (
                self.normalize_angle(
                    heading_change
                )
            )

            movement_type = (
                self.classify_movement(
                    heading_change
                )
            )

        return Movement(
            movement_id=movement_id,
            junction_id=junction_id,
            from_lane_id=from_lane_id,
            to_lane_id=to_lane_id,
            from_edge_id=from_edge_id,
            to_edge_id=to_edge_id,
            heading_in=heading_in,
            heading_out=heading_out,
            heading_change=heading_change,
            movement_type=movement_type,
        )
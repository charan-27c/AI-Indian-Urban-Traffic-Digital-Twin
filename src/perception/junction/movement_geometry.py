from dataclasses import dataclass
from math import atan2, degrees
from typing import List, Tuple


Point = Tuple[float, float]


@dataclass
class MovementGeometry:
    movement_id: str
    from_lane_id: str
    to_lane_id: str

    turn_angle: float
    movement_type: str
    confidence: float


class MovementGeometryInferencer:
    """
    Infers movement semantics directly from SUMO lane geometry.

    No fixed:
        - junction IDs
        - compass directions
        - approach count
        - lane count
        - movement definitions

    are required.
    """

    STRAIGHT_THRESHOLD = 30.0
    UTURN_THRESHOLD = 150.0

    def __init__(
        self,
        straight_threshold: float = STRAIGHT_THRESHOLD,
        uturn_threshold: float = UTURN_THRESHOLD,
    ):
        self.straight_threshold = straight_threshold
        self.uturn_threshold = uturn_threshold

    # ========================================================
    # GEOMETRY HELPERS
    # ========================================================

    @staticmethod
    def _heading(
        start: Point,
        end: Point,
    ) -> float:

        dx = end[0] - start[0]
        dy = end[1] - start[1]

        return degrees(
            atan2(dy, dx)
        )

    @staticmethod
    def _normalize_angle(
        angle: float,
    ) -> float:

        while angle > 180.0:
            angle -= 360.0

        while angle < -180.0:
            angle += 360.0

        return angle

    # ========================================================
    # TURN ANGLE
    # ========================================================

    def calculate_turn_angle(
        self,
        incoming_shape: List[Point],
        outgoing_shape: List[Point],
    ) -> float:

        if len(incoming_shape) < 2:
            raise ValueError(
                "Incoming lane geometry requires "
                "at least two points."
            )

        if len(outgoing_shape) < 2:
            raise ValueError(
                "Outgoing lane geometry requires "
                "at least two points."
            )

        incoming_heading = self._heading(
            incoming_shape[-2],
            incoming_shape[-1],
        )

        outgoing_heading = self._heading(
            outgoing_shape[0],
            outgoing_shape[1],
        )

        return self._normalize_angle(
            outgoing_heading - incoming_heading
        )

    # ========================================================
    # MOVEMENT TYPE
    # ========================================================

    def classify_angle(
        self,
        angle: float,
    ) -> str:

        magnitude = abs(angle)

        if magnitude <= self.straight_threshold:
            return "straight"

        if magnitude >= self.uturn_threshold:
            return "u_turn"

        if angle > 0:
            return "left"

        return "right"

    # ========================================================
    # CONFIDENCE
    # ========================================================

    def calculate_confidence(
        self,
        angle: float,
    ) -> float:

        magnitude = abs(angle)

        if magnitude <= 10:
            return 1.0

        if magnitude <= self.straight_threshold:
            return 0.8

        if magnitude >= self.uturn_threshold:
            return 0.95

        if 70 <= magnitude <= 110:
            return 0.95

        return 0.75

    # ========================================================
    # INFER
    # ========================================================

    def infer(
        self,
        movement_id: str,
        from_lane_id: str,
        to_lane_id: str,
        incoming_shape: List[Point],
        outgoing_shape: List[Point],
    ) -> MovementGeometry:

        angle = self.calculate_turn_angle(
            incoming_shape,
            outgoing_shape,
        )

        movement_type = self.classify_angle(
            angle
        )

        confidence = self.calculate_confidence(
            angle
        )

        return MovementGeometry(
            movement_id=movement_id,
            from_lane_id=from_lane_id,
            to_lane_id=to_lane_id,
            turn_angle=angle,
            movement_type=movement_type,
            confidence=confidence,
        )
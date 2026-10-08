from __future__ import annotations

import math
from typing import Optional, Tuple

from .lane_geometry import LaneGeometry


# ============================================================
# MOVEMENT TYPES
# ============================================================

LEFT = "LEFT"
RIGHT = "RIGHT"
STRAIGHT = "STRAIGHT"
U_TURN = "U_TURN"
UNKNOWN = "UNKNOWN"


# ============================================================
# GEOMETRY MOVEMENT INFERENCE
# ============================================================

class GeometryMovementInference:
    """
    Infers movement semantics from SUMO lane geometry.

    No fixed junction layout is assumed.

    Supported:
        LEFT
        RIGHT
        STRAIGHT
        U_TURN
        UNKNOWN
    """

    def __init__(
        self,
        straight_threshold: float = 30.0,
        u_turn_threshold: float = 135.0,
    ):
        self.straight_threshold = (
            float(straight_threshold)
        )

        self.u_turn_threshold = (
            float(u_turn_threshold)
        )

    # ========================================================
    # ANGLE NORMALIZATION
    # ========================================================

    @staticmethod
    def normalize_angle(
        angle: float,
    ) -> float:
        """
        Normalize angle to [-180, 180].
        """

        while angle > 180.0:
            angle -= 360.0

        while angle <= -180.0:
            angle += 360.0

        return angle

    # ========================================================
    # HEADING
    # ========================================================

    @staticmethod
    def heading(
        point_a: Tuple[float, float],
        point_b: Tuple[float, float],
    ) -> float:
        """
        Calculate heading from point_a to point_b.

        Returns degrees in the mathematical coordinate system.
        """

        dx = point_b[0] - point_a[0]
        dy = point_b[1] - point_a[1]

        if abs(dx) < 1e-12 and abs(dy) < 1e-12:
            return 0.0

        return math.degrees(
            math.atan2(dy, dx)
        )

    # ========================================================
    # LANE ENTRY HEADING
    # ========================================================

    @classmethod
    def entry_heading(
        cls,
        geometry: LaneGeometry,
    ) -> Optional[float]:
        """
        Determine the direction in which traffic approaches
        the junction.

        Uses the first two geometry points.
        """

        shape = geometry.shape

        if len(shape) < 2:
            return None

        return cls.heading(
            shape[0],
            shape[1],
        )

    # ========================================================
    # LANE EXIT HEADING
    # ========================================================

    @classmethod
    def exit_heading(
        cls,
        geometry: LaneGeometry,
    ) -> Optional[float]:
        """
        Determine the direction in which traffic leaves
        the junction.

        Uses the final two geometry points.
        """

        shape = geometry.shape

        if len(shape) < 2:
            return None

        return cls.heading(
            shape[-2],
            shape[-1],
        )

    # ========================================================
    # TURN ANGLE
    # ========================================================

    @classmethod
    def turn_angle(
        cls,
        incoming: LaneGeometry,
        outgoing: LaneGeometry,
    ) -> Optional[float]:
        """
        Calculate the signed turn angle.

        Positive:
            counter-clockwise

        Negative:
            clockwise
        """

        incoming_heading = cls.entry_heading(
            incoming
        )

        outgoing_heading = cls.exit_heading(
            outgoing
        )

        if (
            incoming_heading is None
            or outgoing_heading is None
        ):
            return None

        return cls.normalize_angle(
            outgoing_heading
            - incoming_heading
        )

    # ========================================================
    # CLASSIFICATION
    # ========================================================

    def classify_angle(
        self,
        angle: Optional[float],
    ) -> str:
        """
        Convert turn angle into movement semantics.
        """

        if angle is None:
            return UNKNOWN

        absolute_angle = abs(angle)

        # ----------------------------------------------------
        # U-TURN
        # ----------------------------------------------------

        if absolute_angle >= self.u_turn_threshold:
            return U_TURN

        # ----------------------------------------------------
        # STRAIGHT
        # ----------------------------------------------------

        if absolute_angle <= self.straight_threshold:
            return STRAIGHT

        # ----------------------------------------------------
        # LEFT / RIGHT
        # ----------------------------------------------------

        if angle > 0:
            return LEFT

        return RIGHT

    # ========================================================
    # INFER
    # ========================================================

    def infer(
        self,
        incoming: LaneGeometry,
        outgoing: LaneGeometry,
    ) -> str:
        """
        Infer movement type from two lane geometries.
        """

        angle = self.turn_angle(
            incoming,
            outgoing,
        )

        return self.classify_angle(
            angle
        )

    # ========================================================
    # INFER WITH DETAILS
    # ========================================================

    def infer_with_details(
        self,
        incoming: LaneGeometry,
        outgoing: LaneGeometry,
    ) -> dict:
        """
        Return movement classification together with
        geometry information.

        Useful for debugging and the future dashboard.
        """

        incoming_heading = self.entry_heading(
            incoming
        )

        outgoing_heading = self.exit_heading(
            outgoing
        )

        angle = self.turn_angle(
            incoming,
            outgoing,
        )

        movement = self.classify_angle(
            angle
        )

        return {
            "movement": movement,
            "incoming_lane_id": incoming.lane_id,
            "outgoing_lane_id": outgoing.lane_id,
            "incoming_heading": incoming_heading,
            "outgoing_heading": outgoing_heading,
            "turn_angle": angle,
        }
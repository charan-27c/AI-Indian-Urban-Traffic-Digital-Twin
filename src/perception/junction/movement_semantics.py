from dataclasses import dataclass, field
from typing import Dict, Optional
import math

from src.perception.junction.topology import (
    JunctionTopology,
    Movement,
)


@dataclass
class MovementSemantic:

    movement_id: str

    from_lane_id: str
    to_lane_id: str

    from_approach_id: str
    to_edge_id: str

    movement_type: str = "unknown"

    turning_angle: Optional[float] = None

    signal_group: Optional[str] = None

    metadata: Dict = field(
        default_factory=dict
    )


class MovementSemanticAnalyzer:
    """
    Runtime movement semantic analyzer.

    Movement classification is based on geometry,
    not fixed direction names.
    """

    STRAIGHT_THRESHOLD = 25.0

    LEFT_THRESHOLD = 135.0

    RIGHT_THRESHOLD = -25.0

    UTURN_THRESHOLD = 150.0

    def analyze(
        self,
        topology: JunctionTopology,
    ) -> Dict[str, MovementSemantic]:

        result = {}

        for movement in (
            topology.get_movements()
        ):

            angle = (
                self._get_turning_angle(
                    movement,
                    topology,
                )
            )

            movement_type = (
                self._classify_angle(angle)
            )

            semantic = MovementSemantic(

                movement_id=(
                    movement.movement_id
                ),

                from_lane_id=(
                    movement.from_lane_id
                ),

                to_lane_id=(
                    movement.to_lane_id
                ),

                from_approach_id=(
                    movement.from_approach_id
                ),

                to_edge_id=(
                    movement.to_edge_id
                ),

                movement_type=movement_type,

                turning_angle=angle,
            )

            result[
                movement.movement_id
            ] = semantic

        return result

    # ========================================================
    # TURNING ANGLE
    # ========================================================

    def _get_turning_angle(
        self,
        movement: Movement,
        topology: JunctionTopology,
    ) -> Optional[float]:

        from_lane = self._find_lane(
            movement.from_lane_id,
            topology,
        )

        if from_lane is None:
            return None

        # The outgoing lane may belong to another junction,
        # so first try to obtain its geometry from the movement
        # metadata if available.
        to_lane = self._find_lane(
            movement.to_lane_id,
            topology,
        )

        if to_lane is None:
            return None

        from_geometry = from_lane.geometry
        to_geometry = to_lane.geometry

        if len(from_geometry) < 2:
            return None

        if len(to_geometry) < 2:
            return None

        # --------------------------------------------------------
        # Incoming heading
        # --------------------------------------------------------

        p1 = from_geometry[-2]
        p2 = from_geometry[-1]

        incoming_angle = math.degrees(
            math.atan2(
                p2[1] - p1[1],
                p2[0] - p1[0],
            )
        )

        # --------------------------------------------------------
        # Outgoing heading
        # --------------------------------------------------------

        p3 = to_geometry[0]
        p4 = to_geometry[1]

        outgoing_angle = math.degrees(
            math.atan2(
                p4[1] - p3[1],
                p4[0] - p3[0],
            )
        )

        # --------------------------------------------------------
        # Signed turn angle
        # --------------------------------------------------------

        angle = (
            outgoing_angle
            - incoming_angle
        )

        return self._normalize_angle(
            angle
        )

    # ========================================================
    # CLASSIFICATION
    # ========================================================

    def _classify_angle(
        self,
        angle: Optional[float],
    ) -> str:

        if angle is None:
            return "unknown"

        normalized = self._normalize_angle(
            angle
        )

        absolute = abs(normalized)

        # Straight
        if absolute <= 25.0:
            return "straight"

        # U-turn
        if absolute >= 150.0:
            return "u_turn"

        # Left
        if normalized > 25.0:
            return "left"

        # Right
        if normalized < -25.0:
            return "right"

        return "unknown"

    # ========================================================
    # FIND LANE
    # ========================================================

    def _find_lane(
        self,
        lane_id: str,
        topology: JunctionTopology,
    ):

        for approach in (
            topology.get_approaches()
        ):

            for lane in (
                approach.get_lanes()
            ):

                if lane.lane_id == lane_id:
                    return lane

        return None

    # ========================================================
    # NORMALIZE ANGLE
    # ========================================================

    @staticmethod
    def _normalize_angle(
        angle: float,
    ) -> float:

        while angle > 180.0:
            angle -= 360.0

        while angle <= -180.0:
            angle += 360.0

        return angle
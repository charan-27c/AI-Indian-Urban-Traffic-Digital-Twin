from dataclasses import dataclass
from typing import Dict, List, Tuple


# ============================================================
# MOVEMENT CONFLICT
# ============================================================

@dataclass(frozen=True)
class MovementConflict:
    """
    Describes the relationship between two movements.

    relationship:
        conflict      -> movements should not run together
        compatible    -> movements may run together
        unknown       -> insufficient information
    """

    movement_a: str
    movement_b: str
    relationship: str
    confidence: float = 0.0


# ============================================================
# CONFLICT MATRIX
# ============================================================

class MovementConflictMatrix:
    """
    Runtime conflict graph for dynamically discovered movements.

    No fixed:
        - junction IDs
        - movement IDs
        - approach count
        - lane count
        - LEFT/RIGHT/STRAIGHT assumptions
    """

    def __init__(self):

        self._relationships: Dict[
            Tuple[str, str],
            MovementConflict,
        ] = {}

    # --------------------------------------------------------
    # NORMALIZE PAIR
    # --------------------------------------------------------

    @staticmethod
    def _pair(
        movement_a: str,
        movement_b: str,
    ) -> Tuple[str, str]:

        if movement_a <= movement_b:
            return movement_a, movement_b

        return movement_b, movement_a

    # --------------------------------------------------------
    # REGISTER
    # --------------------------------------------------------

    def register(
        self,
        conflict: MovementConflict,
    ) -> None:

        if not conflict.movement_a:
            raise ValueError(
                "movement_a cannot be empty."
            )

        if not conflict.movement_b:
            raise ValueError(
                "movement_b cannot be empty."
            )

        if (
            conflict.movement_a
            == conflict.movement_b
        ):
            raise ValueError(
                "A movement cannot conflict "
                "with itself."
            )

        valid_relationships = {
            "conflict",
            "compatible",
            "unknown",
        }

        if (
            conflict.relationship
            not in valid_relationships
        ):
            raise ValueError(
                "Invalid relationship: "
                f"{conflict.relationship}"
            )

        pair = self._pair(
            conflict.movement_a,
            conflict.movement_b,
        )

        normalized = MovementConflict(
            movement_a=pair[0],
            movement_b=pair[1],
            relationship=conflict.relationship,
            confidence=conflict.confidence,
        )

        self._relationships[pair] = normalized

    # --------------------------------------------------------
    # GET RELATIONSHIP
    # --------------------------------------------------------

    def get(
        self,
        movement_a: str,
        movement_b: str,
    ) -> MovementConflict:

        pair = self._pair(
            movement_a,
            movement_b,
        )

        if pair not in self._relationships:
            raise KeyError(
                f"No relationship registered "
                f"between '{movement_a}' and "
                f"'{movement_b}'."
            )

        return self._relationships[pair]

    # --------------------------------------------------------
    # CONTAINS
    # --------------------------------------------------------

    def contains(
        self,
        movement_a: str,
        movement_b: str,
    ) -> bool:

        pair = self._pair(
            movement_a,
            movement_b,
        )

        return pair in self._relationships

    # --------------------------------------------------------
    # CONFLICTS
    # --------------------------------------------------------

    def conflicts_with(
        self,
        movement_id: str,
    ) -> List[str]:

        result = []

        for relationship in (
            self._relationships.values()
        ):

            if (
                relationship.relationship
                != "conflict"
            ):
                continue

            if (
                relationship.movement_a
                == movement_id
            ):
                result.append(
                    relationship.movement_b
                )

            elif (
                relationship.movement_b
                == movement_id
            ):
                result.append(
                    relationship.movement_a
                )

        return sorted(result)

    # --------------------------------------------------------
    # COMPATIBLE
    # --------------------------------------------------------

    def compatible_with(
        self,
        movement_id: str,
    ) -> List[str]:

        result = []

        for relationship in (
            self._relationships.values()
        ):

            if (
                relationship.relationship
                != "compatible"
            ):
                continue

            if (
                relationship.movement_a
                == movement_id
            ):
                result.append(
                    relationship.movement_b
                )

            elif (
                relationship.movement_b
                == movement_id
            ):
                result.append(
                    relationship.movement_a
                )

        return sorted(result)

    # --------------------------------------------------------
    # GET ALL
    # --------------------------------------------------------

    def get_all(
        self,
    ) -> Dict[
        Tuple[str, str],
        MovementConflict,
    ]:

        return dict(
            self._relationships
        )

    # --------------------------------------------------------
    # COUNT
    # --------------------------------------------------------

    def count(self) -> int:

        return len(
            self._relationships
        )

    # --------------------------------------------------------
    # CLEAR
    # --------------------------------------------------------

    def clear(self) -> None:

        self._relationships.clear()


# ============================================================
# MOVEMENT CONFLICT ANALYZER
# ============================================================

class MovementConflictAnalyzer:
    """
    Builds a conflict matrix for dynamically discovered
    movements.

    IMPORTANT:

    This first version uses movement semantics as a
    topology-level safety abstraction.

    It does NOT assume fixed J1/J2/J3/J4 definitions.
    """

    # --------------------------------------------------------
    # ANALYZE
    # --------------------------------------------------------

    def analyze(
        self,
        movements: list,
    ) -> MovementConflictMatrix:

        matrix = MovementConflictMatrix()

        for index, movement_a in enumerate(
            movements
        ):

            for movement_b in movements[
                index + 1:
            ]:

                relationship = (
                    self._classify_relationship(
                        movement_a,
                        movement_b,
                    )
                )

                matrix.register(
                    MovementConflict(
                        movement_a=(
                            movement_a.movement_id
                        ),
                        movement_b=(
                            movement_b.movement_id
                        ),
                        relationship=relationship,
                        confidence=1.0,
                    )
                )

        return matrix

    # --------------------------------------------------------
    # RELATIONSHIP
    # --------------------------------------------------------

    def _classify_relationship(
        self,
        movement_a,
        movement_b,
    ) -> str:

        # ----------------------------------------------------
        # Same incoming lane
        #
        # Two different movements from the same lane
        # cannot safely receive green simultaneously.
        # ----------------------------------------------------

        if (
            movement_a.from_lane_id
            == movement_b.from_lane_id
        ):
            return "conflict"

        # ----------------------------------------------------
        # Same incoming approach
        #
        # Different lanes may have independent movements,
        # but without explicit lane-level signal information
        # we remain conservative.
        # ----------------------------------------------------

        if (
            movement_a.from_approach_id
            == movement_b.from_approach_id
        ):
            return "conflict"

        # ----------------------------------------------------
        # Same outgoing lane
        #
        # Two movements merging into the same outgoing lane
        # are treated as conflicting.
        # ----------------------------------------------------

        if (
            movement_a.to_lane_id
            == movement_b.to_lane_id
        ):
            return "conflict"

        # ----------------------------------------------------
        # Different approaches and different outgoing lanes.
        #
        # At this topology layer we cannot yet prove that
        # their paths intersect.
        #
        # Therefore mark UNKNOWN rather than incorrectly
        # declaring them compatible.
        # ----------------------------------------------------

        return "unknown"
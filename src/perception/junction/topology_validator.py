from dataclasses import dataclass, field
from typing import List, Set

from src.perception.junction.topology import (
    JunctionTopology,
)


@dataclass
class TopologyValidationResult:
    """
    Result of validating one or more dynamically
    discovered junction topologies.
    """

    valid: bool = True

    errors: List[str] = field(
        default_factory=list
    )

    warnings: List[str] = field(
        default_factory=list
    )

    checked_junctions: int = 0

    checked_approaches: int = 0

    checked_lanes: int = 0

    checked_connections: int = 0

    checked_movements: int = 0

    def add_error(self, message: str):
        self.valid = False
        self.errors.append(message)

    def add_warning(self, message: str):
        self.warnings.append(message)

    def get_summary(self) -> dict:
        return {
            "valid": self.valid,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "checked_junctions":
                self.checked_junctions,
            "checked_approaches":
                self.checked_approaches,
            "checked_lanes":
                self.checked_lanes,
            "checked_connections":
                self.checked_connections,
            "checked_movements":
                self.checked_movements,
        }


class TopologyValidator:
    """
    Advanced runtime topology validator.

    This validator deliberately makes NO assumptions about:

        - number of junctions
        - number of approaches
        - compass directions
        - number of lanes
        - number of movements
        - LEFT / STRAIGHT / RIGHT
        - fixed junction IDs

    Everything is validated from discovered topology.
    """

    # ========================================================
    # VALIDATE ALL JUNCTIONS
    # ========================================================

    def validate_all(
        self,
        topologies: dict[str, JunctionTopology],
    ) -> TopologyValidationResult:

        result = TopologyValidationResult()

        if not topologies:
            result.add_error(
                "No junction topology was discovered."
            )
            return result

        result.checked_junctions = len(
            topologies
        )

        for junction_id, topology in (
            topologies.items()
        ):

            self.validate_junction(
                topology,
                result,
            )

        return result

    # ========================================================
    # VALIDATE ONE JUNCTION
    # ========================================================

    def validate_junction(
        self,
        topology: JunctionTopology,
        result: TopologyValidationResult,
    ):

        junction_id = topology.junction_id

        approaches = topology.get_approaches()

        connections = topology.get_connections()

        # ----------------------------------------------------
        # JUNCTION ID
        # ----------------------------------------------------

        if not junction_id:
            result.add_error(
                "Discovered junction has empty ID."
            )

        # ----------------------------------------------------
        # APPROACH VALIDATION
        # ----------------------------------------------------

        result.checked_approaches += len(
            approaches
        )

        if not approaches:

            result.add_warning(
                f"Junction '{junction_id}' "
                f"has no incoming approaches."
            )

        # Track globally within this junction.
        lane_ids: Set[str] = set()

        # ----------------------------------------------------
        # APPROACHES
        # ----------------------------------------------------

        for approach in approaches:

            approach_id = (
                approach.approach_id
            )

            if not approach_id:

                result.add_error(
                    f"Junction '{junction_id}' "
                    f"contains an approach with "
                    f"an empty ID."
                )

                continue

            lanes = approach.get_lanes()

            # ------------------------------------------------
            # LANES
            # ------------------------------------------------

            result.checked_lanes += len(
                lanes
            )

            if not lanes:

                result.add_warning(
                    f"Approach '{approach_id}' "
                    f"in junction '{junction_id}' "
                    f"has no lanes."
                )

            for lane in lanes:

                lane_id = lane.lane_id

                # Empty lane ID
                if not lane_id:

                    result.add_error(
                        f"Approach '{approach_id}' "
                        f"in junction '{junction_id}' "
                        f"contains a lane with "
                        f"an empty ID."
                    )

                    continue

                # Duplicate lane
                if lane_id in lane_ids:

                    result.add_error(
                        f"Duplicate lane '{lane_id}' "
                        f"found in junction "
                        f"'{junction_id}'."
                    )

                lane_ids.add(
                    lane_id
                )

                # Lane must belong to approach
                if (
                    lane.approach_id
                    != approach_id
                ):

                    result.add_error(
                        f"Lane '{lane_id}' "
                        f"claims approach "
                        f"'{lane.approach_id}' "
                        f"but is stored under "
                        f"'{approach_id}'."
                    )

                # Lane must have edge
                if not lane.edge_id:

                    result.add_error(
                        f"Lane '{lane_id}' "
                        f"in junction "
                        f"'{junction_id}' "
                        f"has no edge ID."
                    )

        # ----------------------------------------------------
        # CONNECTION VALIDATION
        # ----------------------------------------------------

        result.checked_connections += len(
            connections
        )

        if not connections:

            result.add_warning(
                f"Junction '{junction_id}' "
                f"has no lane connections."
            )

        connection_ids: Set[str] = set()

        for connection in connections:

            connection_id = (
                connection.connection_id
            )

            # ------------------------------------------------
            # CONNECTION ID
            # ------------------------------------------------

            if not connection_id:

                result.add_error(
                    f"Junction '{junction_id}' "
                    f"contains a connection "
                    f"with an empty ID."
                )

                continue

            if connection_id in connection_ids:

                result.add_error(
                    f"Duplicate connection "
                    f"'{connection_id}' "
                    f"found in junction "
                    f"'{junction_id}'."
                )

            connection_ids.add(
                connection_id
            )

            # ------------------------------------------------
            # SOURCE LANE
            # ------------------------------------------------

            if not connection.from_lane_id:

                result.add_error(
                    f"Connection "
                    f"'{connection_id}' "
                    f"has no source lane."
                )

            elif (
                connection.from_lane_id
                not in lane_ids
            ):

                result.add_error(
                    f"Connection "
                    f"'{connection_id}' "
                    f"references unknown "
                    f"source lane "
                    f"'{connection.from_lane_id}'."
                )

            # ------------------------------------------------
            # DESTINATION LANE
            # ------------------------------------------------

            if not connection.to_lane_id:

                result.add_error(
                    f"Connection "
                    f"'{connection_id}' "
                    f"has no destination lane."
                )

            # Destination lanes may belong to
            # another junction, so they are NOT
            # required to exist in this junction.

            # ------------------------------------------------
            # SOURCE EDGE
            # ------------------------------------------------

            if not connection.from_edge_id:

                result.add_error(
                    f"Connection "
                    f"'{connection_id}' "
                    f"has no source edge."
                )

            # ------------------------------------------------
            # DESTINATION EDGE
            # ------------------------------------------------

            if not connection.to_edge_id:

                result.add_error(
                    f"Connection "
                    f"'{connection_id}' "
                    f"has no destination edge."
                )

        # ----------------------------------------------------
        # MOVEMENT COUNT
        # ----------------------------------------------------
        #
        # At this architecture level:
        #
        #     movement = lane-to-lane connection
        #
        # A richer movement model can later add:
        #
        #     route
        #     direction
        #     turn classification
        #     conflict group
        #
        # without changing topology discovery.
        # ----------------------------------------------------

        result.checked_movements += len(
            connections
        )

        # ----------------------------------------------------
        # ORPHAN APPROACH CHECK
        # ----------------------------------------------------

        connected_source_lanes = {
            connection.from_lane_id
            for connection in connections
            if connection.from_lane_id
        }

        for lane_id in lane_ids:

            if lane_id not in (
                connected_source_lanes
            ):

                result.add_warning(
                    f"Lane '{lane_id}' "
                    f"in junction "
                    f"'{junction_id}' "
                    f"has no outgoing "
                    f"movement connection."
                )

    # ========================================================
    # STRICT VALIDATION
    # ========================================================

    def validate_or_raise(
        self,
        topologies: dict[str, JunctionTopology],
    ) -> TopologyValidationResult:

        result = self.validate_all(
            topologies
        )

        if not result.valid:

            message = [
                "Dynamic topology validation failed."
            ]

            for error in result.errors:

                message.append(
                    f"  ERROR: {error}"
                )

            raise ValueError(
                "\n".join(message)
            )

        return result
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


# ============================================================
# LANE
# ============================================================

@dataclass
class Lane:

    lane_id: str

    approach_id: str

    edge_id: str

    geometry: List[
        Tuple[float, float]
    ] = field(
        default_factory=list
    )


# ============================================================
# APPROACH
# ============================================================

@dataclass
class Approach:

    approach_id: str

    name: str

    lanes: List[Lane] = field(
        default_factory=list
    )

    def add_lane(
        self,
        lane: Lane,
    ) -> None:

        self.lanes.append(
            lane
        )

    def get_lanes(
        self,
    ) -> List[Lane]:

        return list(
            self.lanes
        )


# ============================================================
# CONNECTION
# ============================================================

@dataclass
class Connection:

    connection_id: str

    from_lane_id: str

    to_lane_id: str

    from_edge_id: str

    to_edge_id: str


# ============================================================
# MOVEMENT
# ============================================================

@dataclass
class Movement:

    movement_id: str

    junction_id: str

    connection_id: str

    from_lane_id: str

    to_lane_id: str

    from_edge_id: str

    to_edge_id: str

    from_approach_id: str

    @property
    def approach_id(self) -> str:
        """
        Backward-compatible alias.

        Some older modules may use movement.approach_id.
        """

        return self.from_approach_id


# ============================================================
# JUNCTION TOPOLOGY
# ============================================================

@dataclass
class JunctionTopology:

    junction_id: str

    approaches: Dict[
        str,
        Approach
    ] = field(
        default_factory=dict
    )

    connections: Dict[
        str,
        Connection
    ] = field(
        default_factory=dict
    )

    movements: Dict[
        str,
        Movement
    ] = field(
        default_factory=dict
    )

    # ========================================================
    # APPROACHES
    # ========================================================

    def add_approach(
        self,
        approach: Approach,
    ) -> None:

        self.approaches[
            approach.approach_id
        ] = approach

    def get_approaches(
        self,
    ) -> List[Approach]:

        return list(
            self.approaches.values()
        )

    def get_approach(
        self,
        approach_id: str,
    ) -> Approach:

        if approach_id not in self.approaches:

            raise KeyError(
                f"Approach '{approach_id}' "
                f"not found in junction "
                f"'{self.junction_id}'."
            )

        return self.approaches[
            approach_id
        ]

    def approach_count(self) -> int:

        return len(
            self.approaches
        )

    # ========================================================
    # LANES
    # ========================================================

    def lane_count(self) -> int:

        return sum(
            len(
                approach.lanes
            )
            for approach
            in self.approaches.values()
        )

    # ========================================================
    # CONNECTIONS
    # ========================================================

    def add_connection(
        self,
        connection: Connection,
    ) -> None:

        self.connections[
            connection.connection_id
        ] = connection

    def get_connections(
        self,
    ) -> List[Connection]:

        return list(
            self.connections.values()
        )

    def connection_count(self) -> int:

        return len(
            self.connections
        )

    # ========================================================
    # MOVEMENTS
    # ========================================================

    def add_movement(
        self,
        movement: Movement,
    ) -> None:

        self.movements[
            movement.movement_id
        ] = movement

    def get_movements(
        self,
    ) -> List[Movement]:

        return list(
            self.movements.values()
        )

    def get_movement(
        self,
        movement_id: str,
    ) -> Movement:

        if movement_id not in self.movements:

            raise KeyError(
                f"Movement '{movement_id}' "
                f"not found in junction "
                f"'{self.junction_id}'."
            )

        return self.movements[
            movement_id
        ]

    def movement_count(self) -> int:

        return len(
            self.movements
        )

    # ========================================================
    # SUMMARY
    # ========================================================

    def get_summary(self) -> dict:

        return {
            "junction_id":
                self.junction_id,

            "approach_count":
                self.approach_count(),

            "lane_count":
                self.lane_count(),

            "connection_count":
                self.connection_count(),

            "movement_count":
                self.movement_count(),

            "approaches": [

                {
                    "approach_id":
                        approach.approach_id,

                    "name":
                        approach.name,

                    "lane_count":
                        len(
                            approach.lanes
                        ),

                    "lanes": [

                        lane.lane_id

                        for lane
                        in approach.lanes
                    ],
                }

                for approach
                in self.get_approaches()
            ],
        }
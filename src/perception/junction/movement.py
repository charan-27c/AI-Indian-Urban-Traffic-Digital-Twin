from dataclasses import dataclass
from typing import Optional


@dataclass
class Movement:
    """
    Dynamically discovered traffic movement.

    A movement is created from an actual lane-to-lane
    connection discovered from the SUMO network.

    No junction-specific assumptions are made.
    """

    movement_id: str

    junction_id: str

    from_lane_id: str
    to_lane_id: str

    from_edge_id: str
    to_edge_id: str

    heading_in: Optional[float] = None
    heading_out: Optional[float] = None
    heading_change: Optional[float] = None

    movement_type: str = "unknown"

    def get_metadata(self) -> dict:
        return {
            "movement_id": self.movement_id,
            "junction_id": self.junction_id,
            "from_lane_id": self.from_lane_id,
            "to_lane_id": self.to_lane_id,
            "from_edge_id": self.from_edge_id,
            "to_edge_id": self.to_edge_id,
            "heading_in": self.heading_in,
            "heading_out": self.heading_out,
            "heading_change": self.heading_change,
            "movement_type": self.movement_type,
        }
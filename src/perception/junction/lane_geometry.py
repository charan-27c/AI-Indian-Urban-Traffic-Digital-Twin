from dataclasses import dataclass
from typing import Dict, List, Tuple


# ============================================================
# LANE GEOMETRY
# ============================================================

@dataclass
class LaneGeometry:
    """
    Runtime geometry of one SUMO lane.
    """

    lane_id: str
    edge_id: str

    shape: List[
        Tuple[float, float]
    ]

    length: float = 0.0
    width: float = 0.0

    @property
    def start_point(self):
        if not self.shape:
            return None

        return self.shape[0]

    @property
    def end_point(self):
        if not self.shape:
            return None

        return self.shape[-1]

    @property
    def point_count(self) -> int:
        return len(self.shape)


# ============================================================
# REGISTRY
# ============================================================

class LaneGeometryRegistry:

    def __init__(self):

        self._geometries: Dict[
            str,
            LaneGeometry,
        ] = {}

    # --------------------------------------------------------
    # REGISTER
    # --------------------------------------------------------

    def register(
        self,
        geometry: LaneGeometry,
    ) -> None:

        if not geometry.lane_id:

            raise ValueError(
                "Lane geometry must have a lane_id."
            )

        self._geometries[
            geometry.lane_id
        ] = geometry

    # --------------------------------------------------------
    # GET
    # --------------------------------------------------------

    def get(
        self,
        lane_id: str,
    ) -> LaneGeometry:

        if lane_id not in self._geometries:

            raise KeyError(
                f"Lane geometry not found: {lane_id}"
            )

        return self._geometries[
            lane_id
        ]

    # --------------------------------------------------------
    # GET ALL
    # --------------------------------------------------------

    def get_all(
        self,
    ) -> Dict[str, LaneGeometry]:

        return dict(
            self._geometries
        )

    # --------------------------------------------------------
    # COUNT
    # --------------------------------------------------------

    def count(self) -> int:

        return len(
            self._geometries
        )

    # --------------------------------------------------------
    # CONTAINS
    # --------------------------------------------------------

    def contains(
        self,
        lane_id: str,
    ) -> bool:

        return lane_id in self._geometries

    # --------------------------------------------------------
    # CLEAR
    # --------------------------------------------------------

    def clear(self) -> None:

        self._geometries.clear()

    # --------------------------------------------------------
    # ITERATION
    # --------------------------------------------------------

    def __iter__(self):

        return iter(
            self._geometries.values()
        )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    def summary(self) -> dict:

        return {
            "lane_count":
                self.count(),

            "lanes":
                list(
                    self._geometries.keys()
                ),
        }
from pathlib import Path
from typing import Dict, List, Optional

import sumolib

from .topology_models import (
    Lane,
    Approach,
    Connection,
    Movement,
    JunctionTopology,
)

from .lane_geometry import (
    LaneGeometry,
    LaneGeometryRegistry,
)
from .movement_geometry import (
    MovementGeometryInferencer,
)

class SumoTopology:
    """
    Dynamic SUMO topology discovery.

    SUMO is treated as the authoritative runtime topology.

    This class does NOT assume:

        - fixed junction IDs
        - fixed number of junctions
        - four approaches
        - four directions
        - fixed lane counts
        - fixed movements
        - fixed signal phases

    Runtime topology is discovered from the SUMO .net.xml file.

    Geometry is also discovered automatically during construction.
    """

    # ========================================================
    # INITIALIZATION
    # ========================================================

    def __init__(
        self,
        network_path: str,
    ) -> None:

        self.network_path = Path(
            network_path
        ).resolve()

        if not self.network_path.exists():
            raise FileNotFoundError(
                "SUMO network file not found: "
                f"{self.network_path}"
            )

        if self.network_path.suffixes[-2:] != [
            ".net",
            ".xml",
        ]:
            raise ValueError(
                "Expected a SUMO .net.xml network file: "
                f"{self.network_path}"
            )

        # ----------------------------------------------------
        # LOAD SUMO NETWORK
        # ----------------------------------------------------

        self.net = sumolib.net.readNet(
            str(self.network_path)
        )

        # ----------------------------------------------------
        # GEOMETRY REGISTRY
        # ----------------------------------------------------

        self.geometry_registry = (
            LaneGeometryRegistry()
        )
        self.movement_geometry_inferencer = (
             MovementGeometryInferencer()
      )

        # ----------------------------------------------------
        # RUNTIME TOPOLOGY CACHE
        # ----------------------------------------------------

        self._junctions: Dict[
            str,
            JunctionTopology,
        ] = {}

        # ----------------------------------------------------
        # DISCOVER GEOMETRY IMMEDIATELY
        # ----------------------------------------------------

        self._discover_lane_geometry()

    # ========================================================
    # LANE GEOMETRY
    # ========================================================

    def _discover_lane_geometry(
        self,
    ) -> None:
        """
        Discover geometry for every SUMO lane.

        This is intentionally performed during __init__ so
        callers can immediately access:

            loader.geometry_registry

        without first calling discover_all_junctions().
        """

        self.geometry_registry.clear()

        for edge in self.net.getEdges():

            edge_id = edge.getID()

            # Ignore internal SUMO edges only if they do not
            # expose normal lane geometry.
            for lane in edge.getLanes():

                lane_id = lane.getID()

                # ------------------------------------------------
                # SUMO lane shape
                # ------------------------------------------------

                shape = lane.getShape()

                if shape is None:
                    shape = []

                normalized_shape = [
                    (
                        float(point[0]),
                        float(point[1]),
                    )
                    for point in shape
                ]

                # ------------------------------------------------
                # Length
                # ------------------------------------------------

                try:
                    length = float(
                        lane.getLength()
                    )
                except (
                    AttributeError,
                    TypeError,
                    ValueError,
                ):
                    length = 0.0

                # ------------------------------------------------
                # Width
                # ------------------------------------------------

                try:
                    width = float(
                        lane.getWidth()
                    )
                except (
                    AttributeError,
                    TypeError,
                    ValueError,
                ):
                    width = 0.0

                geometry = LaneGeometry(
                    lane_id=lane_id,
                    edge_id=edge_id,
                    shape=normalized_shape,
                    length=length,
                    width=width,
                )

                self.geometry_registry.register(
                    geometry
                )

    # ========================================================
    # GET LANE GEOMETRY
    # ========================================================

    def get_lane_geometry(
        self,
        lane_id: str,
    ) -> LaneGeometry:

        return self.geometry_registry.get(
            lane_id
        )

    # ========================================================
    # DISCOVER ALL JUNCTIONS
    # ========================================================

    def discover_all_junctions(
        self,
    ) -> Dict[str, JunctionTopology]:
        """
        Discover all valid runtime junctions.

        SUMO junctions are inspected dynamically.

        Junctions that have no useful incoming lanes are
        ignored.
        """

        self._junctions.clear()

        for junction in self.net.getNodes():

            junction_id = junction.getID()

            # ------------------------------------------------
            # Only real network junctions
            # ------------------------------------------------

            if not self._is_real_junction(
                junction
            ):
                continue

            try:

                topology = (
                    self.discover_junction(
                        junction_id
                    )
                )

                if topology is None:
                    continue

                if topology.approach_count() == 0:
                    continue

                if topology.lane_count() == 0:
                    continue

                self._junctions[
                    junction_id
                ] = topology

            except Exception as exc:

                print(
                    f"[WARNING] Failed to discover "
                    f"junction '{junction_id}': "
                    f"{exc}"
                )

        return dict(
            self._junctions
        )

    # ========================================================
    # VALID JUNCTION
    # ========================================================

    def _is_real_junction(
        self,
        junction,
    ) -> bool:
        """
        Determine whether a SUMO node should be treated
        as a traffic junction.

        Internal connector nodes such as:

            A1_IN
            B1_IN

        are not considered useful junctions unless they
        actually contain incoming lanes.
        """

        incoming_edges = (
            junction.getIncoming()
        )

        if not incoming_edges:
            return False

        return True

    # ========================================================
    # DISCOVER ONE JUNCTION
    # ========================================================

    def discover_junction(
        self,
        junction_id: str,
    ) -> Optional[JunctionTopology]:
        """
        Discover one junction dynamically.
        """

        junction = self.net.getNode(
            junction_id
        )

        if junction is None:
            return None

        topology = JunctionTopology(
            junction_id=junction_id
        )

        # ----------------------------------------------------
        # DISCOVER INCOMING EDGES
        # ----------------------------------------------------

        incoming_edges = (
            junction.getIncoming()
        )

        if not incoming_edges:
            return topology

        # ----------------------------------------------------
        # APPROACHES
        # ----------------------------------------------------

        for edge in incoming_edges:

            edge_id = edge.getID()

            approach = Approach(
                approach_id=edge_id,
                name=edge_id,
            )

            # ------------------------------------------------
            # LANES
            # ------------------------------------------------

            for lane in edge.getLanes():

                lane_id = lane.getID()

                geometry = (
                    self._get_geometry_or_none(
                        lane_id
                    )
                )

                lane_model = Lane(
                    lane_id=lane_id,
                    approach_id=edge_id,
                    edge_id=edge_id,
                    geometry=(
                        geometry.shape
                        if geometry
                        else []
                    ),
                )

                approach.add_lane(
                    lane_model
                )

            # Only register useful approaches.
            if approach.lanes:

                topology.add_approach(
                    approach
                )

        # ----------------------------------------------------
        # CONNECTIONS
        # ----------------------------------------------------

        connection_index = 0

        for from_edge in incoming_edges:

            from_edge_id = (
                from_edge.getID()
            )

            for from_lane in (
                from_edge.getLanes()
            ):

                from_lane_id = (
                    from_lane.getID()
                )

                # ------------------------------------------------
                # SUMO API:
                #
                # lane.getOutgoing()
                #
                # is the safest way to obtain actual
                # outgoing connections for a lane.
                # ------------------------------------------------

                try:
                    outgoing_connections = (
                        from_lane.getOutgoing()
                    )

                except AttributeError:

                    outgoing_connections = []

                for connection in (
                    outgoing_connections
                ):

                    try:

                        to_lane = (
                            connection.getToLane()
                        )

                    except AttributeError:

                        continue

                    if to_lane is None:
                        continue

                    to_lane_id = (
                        to_lane.getID()
                    )

                    to_edge = (
                        to_lane.getEdge()
                    )

                    if to_edge is None:
                        continue

                    to_edge_id = (
                        to_edge.getID()
                    )

                    connection_id = (
                        f"{junction_id}_C"
                        f"{connection_index}"
                    )

                    connection_index += 1

                    connection_model = Connection(
                        connection_id=connection_id,
                        from_lane_id=from_lane_id,
                        to_lane_id=to_lane_id,
                        from_edge_id=from_edge_id,
                        to_edge_id=to_edge_id,
                    )

                    topology.add_connection(
                        connection_model
                    )

                    # ------------------------------------------------
                    # MOVEMENT
                    # ------------------------------------------------

                    movement_id = (
                        f"{junction_id}_M"
                        f"{topology.movement_count()}"
                    )

                    movement_model = Movement(
                        movement_id=movement_id,
                        junction_id=junction_id,
                        connection_id=connection_id,
                        from_lane_id=from_lane_id,
                        to_lane_id=to_lane_id,
                        from_edge_id=from_edge_id,
                        to_edge_id=to_edge_id,
                        from_approach_id=from_edge_id,
                    )

                    topology.add_movement(
                        movement_model
                    )

        return topology

    # ========================================================
    # GEOMETRY LOOKUP
    # ========================================================

    def _get_geometry_or_none(
        self,
        lane_id: str,
    ) -> Optional[LaneGeometry]:

        if not self.geometry_registry.contains(
            lane_id
        ):
            return None

        return self.geometry_registry.get(
            lane_id
        )

    # ========================================================
    # GET CURRENT JUNCTIONS
    # ========================================================

    def get_junctions(
        self,
    ) -> Dict[str, JunctionTopology]:

        return dict(
            self._junctions
        )

    # ========================================================
    # GET ONE JUNCTION
    # ========================================================

    def get_junction(
        self,
        junction_id: str,
    ) -> JunctionTopology:

        if junction_id not in self._junctions:

            raise KeyError(
                f"Junction '{junction_id}' "
                "has not been discovered."
            )

        return self._junctions[
            junction_id
        ]

    # ========================================================
    # SUMMARY
    # ========================================================

    def get_summary(
        self,
    ) -> dict:

        return {
            "network_file":
                str(self.network_path),

            "junction_count":
                len(self._junctions),

            "geometry_lane_count":
                self.geometry_registry.count(),

            "junctions": [
                topology.get_summary()
                for topology
                in self._junctions.values()
            ],
        }
from dataclasses import dataclass, field
from typing import Dict, List, Set

from src.perception.junction.topology import (
    JunctionTopology,
)


# ============================================================
# GRAPH NODES
# ============================================================

@dataclass
class GraphNode:
    """
    Generic runtime topology graph node.

    Supported node types include:

        junction
        approach
        lane
        movement
    """

    node_id: str
    node_type: str
    attributes: Dict = field(
        default_factory=dict
    )


# ============================================================
# GRAPH EDGES
# ============================================================

@dataclass
class GraphEdge:
    """
    Directed relationship between two topology nodes.
    """

    edge_id: str
    from_node: str
    to_node: str
    relation: str
    attributes: Dict = field(
        default_factory=dict
    )


# ============================================================
# RUNTIME TOPOLOGY GRAPH
# ============================================================

class RuntimeTopologyGraph:
    """
    Runtime graph generated from discovered SUMO topology.

    The graph does NOT assume:

        - fixed junction count
        - four directions
        - four approaches
        - fixed lane count
        - fixed movement count
        - LEFT / STRAIGHT / RIGHT
        - fixed junction names

    Everything is discovered dynamically.
    """

    def __init__(self):

        self.nodes: Dict[str, GraphNode] = {}

        self.edges: Dict[str, GraphEdge] = {}

        self._outgoing: Dict[
            str,
            Set[str]
        ] = {}

        self._incoming: Dict[
            str,
            Set[str]
        ] = {}

    # ========================================================
    # NODE MANAGEMENT
    # ========================================================

    def add_node(
        self,
        node: GraphNode,
    ):

        self.nodes[node.node_id] = node

        self._outgoing.setdefault(
            node.node_id,
            set(),
        )

        self._incoming.setdefault(
            node.node_id,
            set(),
        )

    # ========================================================
    # EDGE MANAGEMENT
    # ========================================================

    def add_edge(
        self,
        edge: GraphEdge,
    ):

        if edge.from_node not in self.nodes:
            raise KeyError(
                f"Source node '{edge.from_node}' "
                f"does not exist."
            )

        if edge.to_node not in self.nodes:
            raise KeyError(
                f"Target node '{edge.to_node}' "
                f"does not exist."
            )

        self.edges[edge.edge_id] = edge

        self._outgoing[
            edge.from_node
        ].add(edge.edge_id)

        self._incoming[
            edge.to_node
        ].add(edge.edge_id)

    # ========================================================
    # NODE QUERIES
    # ========================================================

    def get_node(
        self,
        node_id: str,
    ) -> GraphNode:

        if node_id not in self.nodes:
            raise KeyError(
                f"Graph node '{node_id}' not found."
            )

        return self.nodes[node_id]

    def get_nodes_by_type(
        self,
        node_type: str,
    ) -> List[GraphNode]:

        return [
            node
            for node in self.nodes.values()
            if node.node_type == node_type
        ]

    # ========================================================
    # EDGE QUERIES
    # ========================================================

    def get_outgoing_edges(
        self,
        node_id: str,
    ) -> List[GraphEdge]:

        return [
            self.edges[edge_id]
            for edge_id in self._outgoing.get(
                node_id,
                set(),
            )
        ]

    def get_incoming_edges(
        self,
        node_id: str,
    ) -> List[GraphEdge]:

        return [
            self.edges[edge_id]
            for edge_id in self._incoming.get(
                node_id,
                set(),
            )
        ]

    # ========================================================
    # SUMMARY
    # ========================================================

    def get_summary(self) -> dict:

        return {
            "node_count": len(
                self.nodes
            ),

            "edge_count": len(
                self.edges
            ),

            "junctions": len(
                self.get_nodes_by_type(
                    "junction"
                )
            ),

            "approaches": len(
                self.get_nodes_by_type(
                    "approach"
                )
            ),

            "lanes": len(
                self.get_nodes_by_type(
                    "lane"
                )
            ),

            "movements": len(
                self.get_nodes_by_type(
                    "movement"
                )
            ),
        }


# ============================================================
# GRAPH BUILDER
# ============================================================

class TopologyGraphBuilder:
    """
    Builds a RuntimeTopologyGraph from discovered
    JunctionTopology objects.

    The input is runtime topology.

    No road structure is hard-coded.
    """

    def __init__(self):

        self.graph = RuntimeTopologyGraph()

    # ========================================================
    # BUILD ALL JUNCTIONS
    # ========================================================

    def build(
        self,
        topologies: Dict[
            str,
            JunctionTopology
        ],
    ) -> RuntimeTopologyGraph:

        for junction_id, topology in (
            topologies.items()
        ):

            self._add_junction(
                topology
            )

        return self.graph

    # ========================================================
    # JUNCTION
    # ========================================================

    def _add_junction(
        self,
        topology: JunctionTopology,
    ):

        junction_node_id = (
            f"junction:{topology.junction_id}"
        )

        self.graph.add_node(
            GraphNode(
                node_id=junction_node_id,
                node_type="junction",
                attributes={
                    "junction_id":
                        topology.junction_id,
                    "approach_count":
                        topology.approach_count(),
                    "lane_count":
                        topology.lane_count(),
                    "connection_count":
                        topology.connection_count(),
                },
            )
        )

        # ----------------------------------------------------
        # APPROACHES
        # ----------------------------------------------------

        for approach in (
            topology.get_approaches()
        ):

            approach_node_id = (
                f"approach:"
                f"{topology.junction_id}:"
                f"{approach.approach_id}"
            )

            self.graph.add_node(
                GraphNode(
                    node_id=approach_node_id,
                    node_type="approach",
                    attributes={
                        "junction_id":
                            topology.junction_id,

                        "approach_id":
                            approach.approach_id,

                        "name":
                            approach.name,

                        "lane_count":
                            len(approach.lanes),
                    },
                )
            )

            # Approach belongs to junction.

            self.graph.add_edge(
                GraphEdge(
                    edge_id=(
                        f"contains:"
                        f"{junction_node_id}:"
                        f"{approach_node_id}"
                    ),
                    from_node=junction_node_id,
                    to_node=approach_node_id,
                    relation="contains",
                )
            )

            # ------------------------------------------------
            # LANES
            # ------------------------------------------------

            for lane in approach.get_lanes():

                lane_node_id = (
                    f"lane:{lane.lane_id}"
                )

                self.graph.add_node(
                    GraphNode(
                        node_id=lane_node_id,
                        node_type="lane",
                        attributes={
                            "junction_id":
                                topology.junction_id,

                            "approach_id":
                                approach.approach_id,

                            "lane_id":
                                lane.lane_id,

                            "edge_id":
                                lane.edge_id,
                        },
                    )
                )

                self.graph.add_edge(
                    GraphEdge(
                        edge_id=(
                            f"contains:"
                            f"{approach_node_id}:"
                            f"{lane_node_id}"
                        ),
                        from_node=approach_node_id,
                        to_node=lane_node_id,
                        relation="contains",
                    )
                )

        # ----------------------------------------------------
        # MOVEMENTS
        # ----------------------------------------------------

        for connection in (
            topology.get_connections()
        ):

            movement_node_id = (
                f"movement:"
                f"{topology.junction_id}:"
                f"{connection.connection_id}"
            )

            self.graph.add_node(
                GraphNode(
                    node_id=movement_node_id,
                    node_type="movement",
                    attributes={
                        "junction_id":
                            topology.junction_id,

                        "connection_id":
                            connection.connection_id,

                        "from_lane_id":
                            connection.from_lane_id,

                        "to_lane_id":
                            connection.to_lane_id,

                        "from_edge_id":
                            connection.from_edge_id,

                        "to_edge_id":
                            connection.to_edge_id,
                    },
                )
            )

            # ------------------------------------------------
            # MOVEMENT → SOURCE LANE
            # ------------------------------------------------

            source_lane_id = (
                f"lane:"
                f"{connection.from_lane_id}"
            )

            if source_lane_id in self.graph.nodes:

                self.graph.add_edge(
                    GraphEdge(
                        edge_id=(
                            f"movement_source:"
                            f"{movement_node_id}"
                        ),
                        from_node=source_lane_id,
                        to_node=movement_node_id,
                        relation="movement_source",
                    )
                )

            # ------------------------------------------------
            # MOVEMENT → TARGET LANE
            # ------------------------------------------------

            target_lane_id = (
                f"lane:"
                f"{connection.to_lane_id}"
            )

            # Target lanes can belong to another
            # junction and therefore may not exist
            # in this local topology.

            if target_lane_id in self.graph.nodes:

                self.graph.add_edge(
                    GraphEdge(
                        edge_id=(
                            f"movement_target:"
                            f"{movement_node_id}"
                        ),
                        from_node=movement_node_id,
                        to_node=target_lane_id,
                        relation="movement_target",
                    )
                )

    # ========================================================
    # VALIDATION
    # ========================================================

    def validate(self) -> List[str]:

        errors = []

        for node in (
            self.graph.get_nodes_by_type(
                "movement"
            )
        ):

            source_lane = node.attributes.get(
                "from_lane_id"
            )

            target_lane = node.attributes.get(
                "to_lane_id"
            )

            if not source_lane:
                errors.append(
                    f"Movement '{node.node_id}' "
                    f"has no source lane."
                )

            if not target_lane:
                errors.append(
                    f"Movement '{node.node_id}' "
                    f"has no target lane."
                )

        return errors
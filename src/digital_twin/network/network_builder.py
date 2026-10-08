from pathlib import Path
from typing import Dict, List, Any

import yaml


class NetworkBuilder:
    """
    Loads and validates the dynamic network definition.

    The network definition is configuration-driven.

    This class does NOT assume:

        - fixed junction count
        - fixed junction IDs
        - four approaches
        - NORTH / SOUTH / EAST / WEST
        - fixed lane count
        - fixed movement count
        - fixed movement names
        - fixed signal phases

    Physical topology is represented by:

        nodes
        edges

    Runtime topology will later be discovered from
    the generated SUMO .net.xml network.
    """

    # ========================================================
    # INITIALIZATION
    # ========================================================

    def __init__(self, config_path: str):

        self.config_path = Path(
            config_path
        )

        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Network configuration not found: "
                f"{self.config_path}"
            )

        with open(
            self.config_path,
            "r",
            encoding="utf-8",
        ) as file:

            self.config = yaml.safe_load(
                file
            )

        if not self.config:
            raise ValueError(
                "Network configuration is empty."
            )

        self.network = self.config.get(
            "network",
            {},
        )

        if not self.network:
            raise ValueError(
                "Missing 'network' section "
                "in network configuration."
            )

        self.junctions = self.network.get(
            "junctions",
            {},
        )

        self.nodes = self.network.get(
            "nodes",
            [],
        )

        self.edges = self.network.get(
            "edges",
            [],
        )

        self.generation = self.network.get(
            "generation",
            {},
        )

    # ========================================================
    # NETWORK
    # ========================================================

    def get_network_name(self) -> str:
        """
        Return the configured network name.
        """

        return self.network.get(
            "name",
            "unnamed_network",
        )

    # ========================================================
    # JUNCTIONS
    # ========================================================

    def get_junctions(self) -> Dict[str, Dict[str, Any]]:
        """
        Return all configured junction definitions.

        No fixed junction IDs are assumed.
        """

        return dict(
            self.junctions
        )

    def get_junction(
        self,
        junction_id: str,
    ) -> Dict[str, Any]:
        """
        Return one junction definition.
        """

        if junction_id not in self.junctions:
            raise KeyError(
                f"Junction '{junction_id}' "
                f"not found."
            )

        return self.junctions[
            junction_id
        ]

    def get_junction_count(self) -> int:
        """
        Return the number of configured junctions.
        """

        return len(
            self.junctions
        )

    # ========================================================
    # APPROACHES
    # ========================================================

    def get_approaches(
        self,
        junction_id: str,
    ) -> List[str]:
        """
        Return approaches belonging to one junction.

        The number of approaches is completely dynamic.
        """

        junction = self.get_junction(
            junction_id
        )

        approaches = junction.get(
            "approaches",
            [],
        )

        return list(
            approaches
        )

    def get_approach_count(
        self,
        junction_id: str,
    ) -> int:
        """
        Return the number of approaches
        belonging to a junction.
        """

        return len(
            self.get_approaches(
                junction_id
            )
        )

    # ========================================================
    # PHYSICAL NODES
    # ========================================================

    def get_nodes(self) -> List[Dict[str, Any]]:
        """
        Return all physical network nodes.

        Nodes are used to generate the SUMO network.
        """

        return list(
            self.nodes
        )

    def get_node_ids(self) -> List[str]:
        """
        Return all physical node IDs.
        """

        return [
            node["id"]
            for node in self.nodes
            if "id" in node
        ]

    def get_node(
        self,
        node_id: str,
    ) -> Dict[str, Any]:
        """
        Return one physical node.
        """

        for node in self.nodes:

            if node.get("id") == node_id:
                return dict(node)

        raise KeyError(
            f"Node '{node_id}' not found."
        )

    def get_node_count(self) -> int:
        """
        Return the number of physical nodes.
        """

        return len(
            self.nodes
        )

    # ========================================================
    # PHYSICAL EDGES
    # ========================================================

    def get_edges(self) -> List[Dict[str, Any]]:
        """
        Return all physical network edges.

        Edges represent roads.

        They are NOT movement definitions.
        """

        return list(
            self.edges
        )

    def get_edge_ids(self) -> List[str]:
        """
        Return all physical edge IDs.
        """

        return [
            edge["id"]
            for edge in self.edges
            if "id" in edge
        ]

    def get_edge(
        self,
        edge_id: str,
    ) -> Dict[str, Any]:
        """
        Return one physical edge.
        """

        for edge in self.edges:

            if edge.get("id") == edge_id:
                return dict(edge)

        raise KeyError(
            f"Edge '{edge_id}' not found."
        )

    def get_edge_count(self) -> int:
        """
        Return the number of physical edges.
        """

        return len(
            self.edges
        )

    # ========================================================
    # GENERATION CONFIGURATION
    # ========================================================

    def get_generation_config(self) -> Dict[str, Any]:
        """
        Return SUMO network generation configuration.
        """

        return dict(
            self.generation
        )

    def get_generation_output_directory(self) -> str:
        """
        Return the SUMO network output directory.
        """

        output = self.generation.get(
            "output",
            {},
        )

        return output.get(
            "directory",
            "simulation/networks",
        )

    def get_generation_network_filename(self) -> str:
        """
        Return the generated SUMO network filename.
        """

        output = self.generation.get(
            "output",
            {},
        )

        return output.get(
            "network_file",
            "ai_traffic.net.xml",
        )

    # ========================================================
    # STRUCTURAL VALIDATION
    # ========================================================

    def validate(self) -> bool:
        """
        Validate the complete network definition.

        Validation is structural.

        No assumptions are made about:

            - junction count
            - approach count
            - lane count
            - movement count
        """

        # ----------------------------------------------------
        # NETWORK NAME
        # ----------------------------------------------------

        if not self.get_network_name():
            raise ValueError(
                "Network name cannot be empty."
            )

        # ----------------------------------------------------
        # JUNCTIONS
        # ----------------------------------------------------

        if not isinstance(
            self.junctions,
            dict,
        ):
            raise ValueError(
                "'junctions' must be a mapping."
            )

        if not self.junctions:
            raise ValueError(
                "Network contains no junctions."
            )

        junction_ids = set(
            self.junctions.keys()
        )

        # ----------------------------------------------------
        # JUNCTION STRUCTURE
        # ----------------------------------------------------

        for (
            junction_id,
            junction,
        ) in self.junctions.items():

            if not isinstance(
                junction,
                dict,
            ):
                raise ValueError(
                    f"Invalid definition for "
                    f"junction '{junction_id}'."
                )

            # -----------------------------------------------
            # JUNCTION NODE
            # -----------------------------------------------

            node_id = junction.get(
                "node"
            )

            if node_id is not None:

                if not isinstance(
                    node_id,
                    str,
                ):
                    raise ValueError(
                        f"Node reference for "
                        f"junction '{junction_id}' "
                        f"must be a string."
                    )

            # -----------------------------------------------
            # APPROACHES
            # -----------------------------------------------

            approaches = junction.get(
                "approaches",
                [],
            )

            if not isinstance(
                approaches,
                list,
            ):
                raise ValueError(
                    f"Approaches for junction "
                    f"'{junction_id}' must be "
                    f"a list."
                )

            if not approaches:
                raise ValueError(
                    f"Junction '{junction_id}' "
                    f"has no approaches."
                )

            # -----------------------------------------------
            # DUPLICATE APPROACHES
            # -----------------------------------------------

            if len(approaches) != len(
                set(approaches)
            ):
                raise ValueError(
                    f"Junction '{junction_id}' "
                    f"contains duplicate approaches."
                )

            # -----------------------------------------------
            # APPROACH ID VALIDATION
            # -----------------------------------------------

            for approach in approaches:

                if not isinstance(
                    approach,
                    str,
                ):
                    raise ValueError(
                        f"Invalid approach in "
                        f"junction '{junction_id}'."
                    )

                if not approach.strip():
                    raise ValueError(
                        f"Empty approach ID in "
                        f"junction '{junction_id}'."
                    )

        # ----------------------------------------------------
        # PHYSICAL NODES
        # ----------------------------------------------------

        if not isinstance(
            self.nodes,
            list,
        ):
            raise ValueError(
                "'nodes' must be a list."
            )

        if not self.nodes:
            raise ValueError(
                "Network contains no "
                "physical nodes."
            )

        node_ids = set()

        for node in self.nodes:

            if not isinstance(
                node,
                dict,
            ):
                raise ValueError(
                    "Every node must be "
                    "a mapping."
                )

            node_id = node.get(
                "id"
            )

            if not node_id:
                raise ValueError(
                    "Every node must have "
                    "an 'id'."
                )

            if node_id in node_ids:
                raise ValueError(
                    f"Duplicate node ID: "
                    f"'{node_id}'."
                )

            node_ids.add(
                node_id
            )

            # Coordinates are required for
            # physical network generation.

            if "x" not in node:
                raise ValueError(
                    f"Node '{node_id}' "
                    f"is missing 'x'."
                )

            if "y" not in node:
                raise ValueError(
                    f"Node '{node_id}' "
                    f"is missing 'y'."
                )

        # ----------------------------------------------------
        # JUNCTION NODE REFERENCES
        # ----------------------------------------------------

        for (
            junction_id,
            junction,
        ) in self.junctions.items():

            node_id = junction.get(
                "node"
            )

            if node_id is not None:

                if node_id not in node_ids:
                    raise ValueError(
                        f"Junction "
                        f"'{junction_id}' "
                        f"references unknown "
                        f"node '{node_id}'."
                    )

        # ----------------------------------------------------
        # PHYSICAL EDGES
        # ----------------------------------------------------

        if not isinstance(
            self.edges,
            list,
        ):
            raise ValueError(
                "'edges' must be a list."
            )

        if not self.edges:
            raise ValueError(
                "Network contains no "
                "physical edges."
            )

        edge_ids = set()

        for edge in self.edges:

            if not isinstance(
                edge,
                dict,
            ):
                raise ValueError(
                    "Every edge must be "
                    "a mapping."
                )

            edge_id = edge.get(
                "id"
            )

            if not edge_id:
                raise ValueError(
                    "Every edge must have "
                    "an 'id'."
                )

            if edge_id in edge_ids:
                raise ValueError(
                    f"Duplicate edge ID: "
                    f"'{edge_id}'."
                )

            edge_ids.add(
                edge_id
            )

            # -----------------------------------------------
            # SOURCE NODE
            # -----------------------------------------------

            from_node = edge.get(
                "from"
            )

            if from_node not in node_ids:
                raise ValueError(
                    f"Edge '{edge_id}' "
                    f"references unknown "
                    f"source node "
                    f"'{from_node}'."
                )

            # -----------------------------------------------
            # DESTINATION NODE
            # -----------------------------------------------

            to_node = edge.get(
                "to"
            )

            if to_node not in node_ids:
                raise ValueError(
                    f"Edge '{edge_id}' "
                    f"references unknown "
                    f"destination node "
                    f"'{to_node}'."
                )

            # -----------------------------------------------
            # SELF LOOP
            # -----------------------------------------------

            if from_node == to_node:
                raise ValueError(
                    f"Edge '{edge_id}' "
                    f"cannot connect a node "
                    f"to itself."
                )

            # -----------------------------------------------
            # LANES
            # -----------------------------------------------

            lanes = edge.get(
                "lanes",
                1,
            )

            if not isinstance(
                lanes,
                int,
            ):
                raise ValueError(
                    f"Lane count for edge "
                    f"'{edge_id}' must be "
                    f"an integer."
                )

            if lanes < 1:
                raise ValueError(
                    f"Edge '{edge_id}' "
                    f"must have at least "
                    f"one lane."
                )

            # -----------------------------------------------
            # SPEED
            # -----------------------------------------------

            speed = edge.get(
                "speed",
                13.89,
            )

            if speed <= 0:
                raise ValueError(
                    f"Speed for edge "
                    f"'{edge_id}' must "
                    f"be greater than zero."
                )

        # ----------------------------------------------------
        # GENERATION CONFIGURATION
        # ----------------------------------------------------

        if not isinstance(
            self.generation,
            dict,
        ):
            raise ValueError(
                "'generation' must be a mapping."
            )

        output = self.generation.get(
            "output",
            {},
        )

        if not isinstance(
            output,
            dict,
        ):
            raise ValueError(
                "'generation.output' must "
                "be a mapping."
            )

        output_directory = output.get(
            "directory"
        )

        if not output_directory:
            raise ValueError(
                "Generation output directory "
                "is required."
            )

        network_filename = output.get(
            "network_file"
        )

        if not network_filename:
            raise ValueError(
                "Generation network filename "
                "is required."
            )

        return True

    # ========================================================
    # SUMMARY
    # ========================================================

    def get_summary(self) -> dict:
        """
        Return a generic network summary.

        This summary does not assume a fixed
        topology.
        """

        return {
            "name":
                self.get_network_name(),

            "junction_count":
                self.get_junction_count(),

            "node_count":
                self.get_node_count(),

            "edge_count":
                self.get_edge_count(),

            "junctions": {
                junction_id: {
                    "type":
                        junction.get(
                            "type",
                            "unknown",
                        ),

                    "node":
                        junction.get(
                            "node"
                        ),

                    "approach_count":
                        self.get_approach_count(
                            junction_id
                        ),

                    "approaches":
                        self.get_approaches(
                            junction_id
                        ),
                }

                for (
                    junction_id,
                    junction
                ) in self.junctions.items()
            },

            "nodes":
                self.get_node_ids(),

            "edges":
                self.get_edge_ids(),

            "generation":
                self.get_generation_config(),
        }
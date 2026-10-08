from pathlib import Path
import subprocess
from typing import Dict, List, Optional


class SumoNetworkGenerator:
    """
    Generates a SUMO .net.xml network from dynamic
    physical network definitions.

    Architecture:

        network.yaml
             |
             v
        NetworkBuilder
             |
             v
        nodes + edges
             |
             v
        .nod.xml + .edg.xml
             |
             v
        SUMO netconvert
             |
             v
        .net.xml
             |
             v
        SumoTopology
             |
             v
        runtime topology

    This class does NOT assume:

        - fixed junction count
        - fixed junction IDs
        - four approaches
        - cardinal directions
        - fixed lane counts
        - fixed movements
        - LEFT / RIGHT / STRAIGHT

    SUMO is the authoritative runtime topology.
    """

    # ========================================================
    # INITIALIZATION
    # ========================================================

    def __init__(
        self,
        output_directory: str,
        netconvert_binary: str = "netconvert",
    ):
        """
        Initialize the SUMO network generator.

        Parameters
        ----------
        output_directory:
            Directory where generated SUMO files are stored.

        netconvert_binary:
            SUMO netconvert executable.

            Normally:
                netconvert

            Or an absolute path if SUMO is not in PATH.
        """

        self.output_directory = Path(
            output_directory
        )

        self.output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.netconvert_binary = (
            netconvert_binary
        )

    # ========================================================
    # NORMALIZE EDGE LANES
    # ========================================================

    @staticmethod
    def _get_lane_count(
        edge: Dict,
    ) -> int:
        """
        Read lane count from an edge definition.

        Preferred configuration field:

            lanes

        Legacy compatibility:

            numLanes
        """

        if "lanes" in edge:

            lanes = edge["lanes"]

        elif "numLanes" in edge:

            lanes = edge["numLanes"]

        else:

            lanes = 1

        try:

            lanes = int(lanes)

        except (
            TypeError,
            ValueError,
        ) as exc:

            raise ValueError(
                f"Invalid lane count for "
                f"edge '{edge.get('id')}'."
            ) from exc

        if lanes < 1:

            raise ValueError(
                f"Edge '{edge.get('id')}' "
                f"must have at least one lane."
            )

        return lanes

    # ========================================================
    # NORMALIZE SPEED
    # ========================================================

    @staticmethod
    def _get_speed(
        edge: Dict,
    ) -> float:
        """
        Return edge speed in m/s.
        """

        speed = edge.get(
            "speed",
            13.89,
        )

        try:

            speed = float(speed)

        except (
            TypeError,
            ValueError,
        ) as exc:

            raise ValueError(
                f"Invalid speed for "
                f"edge '{edge.get('id')}'."
            ) from exc

        if speed <= 0:

            raise ValueError(
                f"Speed for edge "
                f"'{edge.get('id')}' "
                f"must be greater than zero."
            )

        return speed

    # ========================================================
    # VALIDATE NODES
    # ========================================================

    def validate_nodes(
        self,
        nodes: List[Dict],
    ) -> None:
        """
        Validate physical node definitions.
        """

        if not nodes:

            raise ValueError(
                "Cannot generate SUMO network "
                "without nodes."
            )

        node_ids = set()

        for node in nodes:

            node_id = node.get(
                "id"
            )

            if not node_id:

                raise ValueError(
                    "Every node must have an ID."
                )

            if node_id in node_ids:

                raise ValueError(
                    f"Duplicate node ID: "
                    f"'{node_id}'."
                )

            node_ids.add(
                node_id
            )

            if "x" not in node:

                raise ValueError(
                    f"Node '{node_id}' "
                    f"is missing x coordinate."
                )

            if "y" not in node:

                raise ValueError(
                    f"Node '{node_id}' "
                    f"is missing y coordinate."
                )

    # ========================================================
    # VALIDATE EDGES
    # ========================================================

    def validate_edges(
        self,
        nodes: List[Dict],
        edges: List[Dict],
    ) -> None:
        """
        Validate physical edge definitions.
        """

        if not edges:

            raise ValueError(
                "Cannot generate SUMO network "
                "without edges."
            )

        node_ids = {
            node["id"]
            for node in nodes
        }

        edge_ids = set()

        for edge in edges:

            edge_id = edge.get(
                "id"
            )

            if not edge_id:

                raise ValueError(
                    "Every edge must have an ID."
                )

            if edge_id in edge_ids:

                raise ValueError(
                    f"Duplicate edge ID: "
                    f"'{edge_id}'."
                )

            edge_ids.add(
                edge_id
            )

            from_node = edge.get(
                "from"
            )

            to_node = edge.get(
                "to"
            )

            if from_node not in node_ids:

                raise ValueError(
                    f"Edge '{edge_id}' "
                    f"references unknown "
                    f"source node "
                    f"'{from_node}'."
                )

            if to_node not in node_ids:

                raise ValueError(
                    f"Edge '{edge_id}' "
                    f"references unknown "
                    f"destination node "
                    f"'{to_node}'."
                )

            if from_node == to_node:

                raise ValueError(
                    f"Edge '{edge_id}' "
                    f"cannot connect a node "
                    f"to itself."
                )

            self._get_lane_count(
                edge
            )

            self._get_speed(
                edge
            )

    # ========================================================
    # WRITE NODE FILE
    # ========================================================

    def write_nodes(
        self,
        nodes: List[Dict],
        filename: str = "network.nod.xml",
    ) -> Path:
        """
        Write SUMO node XML.
        """

        self.validate_nodes(
            nodes
        )

        output = (
            self.output_directory
            / filename
        )

        lines = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<nodes version="1.0">',
        ]

        for node in nodes:

            node_id = node["id"]

            x = node["x"]
            y = node["y"]

            node_type = node.get(
                "type",
                "priority",
            )

            lines.append(
                f'    <node '
                f'id="{node_id}" '
                f'x="{x}" '
                f'y="{y}" '
                f'type="{node_type}"/>'
            )

        lines.append(
            "</nodes>"
        )

        output.write_text(
            "\n".join(lines),
            encoding="utf-8",
        )

        return output

    # ========================================================
    # WRITE EDGE FILE
    # ========================================================

    def write_edges(
        self,
        edges: List[Dict],
        nodes: Optional[List[Dict]] = None,
        filename: str = "network.edg.xml",
    ) -> Path:
        """
        Write SUMO edge XML.

        Configuration uses:

            lanes

        SUMO XML uses:

            numLanes
        """

        if nodes is not None:

            self.validate_edges(
                nodes,
                edges,
            )

        elif not edges:

            raise ValueError(
                "Cannot write an empty edge file."
            )

        output = (
            self.output_directory
            / filename
        )

        lines = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<edges version="1.0">',
        ]

        for edge in edges:

            edge_id = edge["id"]

            from_node = edge["from"]

            to_node = edge["to"]

            lanes = self._get_lane_count(
                edge
            )

            speed = self._get_speed(
                edge
            )

            lines.append(
                f'    <edge '
                f'id="{edge_id}" '
                f'from="{from_node}" '
                f'to="{to_node}" '
                f'numLanes="{lanes}" '
                f'speed="{speed}"/>'
            )

        lines.append(
            "</edges>"
        )

        output.write_text(
            "\n".join(lines),
            encoding="utf-8",
        )

        return output

    # ========================================================
    # RUN NETCONVERT
    # ========================================================

    def _run_netconvert(
        self,
        node_file: Path,
        edge_file: Path,
        network_file: Path,
    ) -> subprocess.CompletedProcess:
        """
        Execute SUMO netconvert.
        """

        command = [
            self.netconvert_binary,

            "--node-files",
            str(node_file),

            "--edge-files",
            str(edge_file),

            "--output-file",
            str(network_file),
        ]

        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
        )

        if result.returncode != 0:

            raise RuntimeError(
                "SUMO netconvert failed.\n\n"
                f"Command:\n"
                f"{' '.join(command)}\n\n"
                f"STDOUT:\n"
                f"{result.stdout}\n\n"
                f"STDERR:\n"
                f"{result.stderr}"
            )

        return result

    # ========================================================
    # GENERATE NETWORK
    # ========================================================

    def generate(
        self,
        nodes: List[Dict],
        edges: List[Dict],
        network_filename: str = (
            "ai_traffic.net.xml"
        ),
    ) -> Path:
        """
        Generate the SUMO .net.xml network.

        Pipeline:

            validate
                ↓
            write nodes
                ↓
            write edges
                ↓
            netconvert
                ↓
            verify output
        """

        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        self.validate_nodes(
            nodes
        )

        self.validate_edges(
            nodes,
            edges,
        )

        # ----------------------------------------------------
        # WRITE INPUT FILES
        # ----------------------------------------------------

        node_file = self.write_nodes(
            nodes
        )

        edge_file = self.write_edges(
            edges,
            nodes=nodes,
        )

        # ----------------------------------------------------
        # OUTPUT NETWORK
        # ----------------------------------------------------

        network_file = (
            self.output_directory
            / network_filename
        )

        # ----------------------------------------------------
        # REMOVE OLD NETWORK
        # ----------------------------------------------------

        if network_file.exists():

            network_file.unlink()

        # ----------------------------------------------------
        # RUN NETCONVERT
        # ----------------------------------------------------

        self._run_netconvert(
            node_file=node_file,
            edge_file=edge_file,
            network_file=network_file,
        )

        # ----------------------------------------------------
        # VERIFY
        # ----------------------------------------------------

        if not network_file.exists():

            raise FileNotFoundError(
                "SUMO network was not generated: "
                f"{network_file}"
            )

        if network_file.stat().st_size == 0:

            raise RuntimeError(
                "SUMO generated an empty "
                f"network file: {network_file}"
            )

        return network_file

    # ========================================================
    # GENERATE FROM NETWORK BUILDER
    # ========================================================

    def generate_from_builder(
        self,
        builder,
    ) -> Path:
        """
        Generate a SUMO network directly from
        NetworkBuilder.

        This is the preferred integration path.

        Example:

            builder = NetworkBuilder(
                "config/network.yaml"
            )

            generator.generate_from_builder(
                builder
            )
        """

        # ----------------------------------------------------
        # VALIDATE CONFIGURATION
        # ----------------------------------------------------

        builder.validate()

        # ----------------------------------------------------
        # GET PHYSICAL TOPOLOGY
        # ----------------------------------------------------

        nodes = builder.get_nodes()

        edges = builder.get_edges()

        # ----------------------------------------------------
        # GET OUTPUT CONFIGURATION
        # ----------------------------------------------------

        output_directory = (
            builder.get_generation_output_directory()
        )

        network_filename = (
            builder.get_generation_network_filename()
        )

        # ----------------------------------------------------
        # USE BUILDER OUTPUT DIRECTORY
        # ----------------------------------------------------

        configured_directory = Path(
            output_directory
        )

        if configured_directory != (
            self.output_directory
        ):

            self.output_directory = (
                configured_directory
            )

            self.output_directory.mkdir(
                parents=True,
                exist_ok=True,
            )

        # ----------------------------------------------------
        # GENERATE
        # ----------------------------------------------------

        return self.generate(
            nodes=nodes,
            edges=edges,
            network_filename=network_filename,
        )
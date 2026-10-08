from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import xml.etree.ElementTree as ET

import sumolib


@dataclass(frozen=True)
class RouteDefinition:
    """
    Represents one valid SUMO traffic route.
    """

    route_id: str
    edges: tuple[str, ...]


class DynamicRouteGenerator:
    """
    Dynamically generates SUMO routes from a SUMO network.

    Design principles
    -----------------
    - No hardcoded junction IDs.
    - No hardcoded edge IDs.
    - No hardcoded lane counts.
    - No hardcoded movement counts.
    - Traffic lights are discovered from the SUMO network.
    - Entry edges are discovered automatically.
    - Exit edges are discovered automatically.
    - Routes terminate when they reach a network exit.
    - Feedback/circulation edges are not followed after an exit.
    - Repeated edges are prevented.
    - Maximum route length prevents pathological paths.
    """

    def __init__(
        self,
        network_file: str | Path,
        max_route_edges: int = 8,
    ) -> None:
        self.network_file = Path(network_file)
        self.max_route_edges = max_route_edges

        if not self.network_file.exists():
            raise FileNotFoundError(
                f"SUMO network not found: {self.network_file}"
            )

        if self.network_file.suffix.lower() != ".xml":
            raise ValueError(
                "SUMO network file must be an XML file."
            )

        self.net = sumolib.net.readNet(
            str(self.network_file)
        )

    # ------------------------------------------------------------------
    # NETWORK DISCOVERY
    # ------------------------------------------------------------------

    def edges(self):
        """
        Return all non-internal SUMO edges.

        Internal edges such as :J1_0 are excluded because they
        represent junction geometry rather than normal route edges.
        """
        return [
            edge
            for edge in self.net.getEdges()
            if not edge.getID().startswith(":")
        ]

    def traffic_lights(self) -> list[str]:
        """
        Dynamically discover all SUMO traffic-light controllers.
        """
        return sorted(
            tls.getID()
            for tls in self.net.getTrafficLights()
        )

    def traffic_light_nodes(self):
        """
        Dynamically discover the SUMO nodes controlled by traffic lights.
        """
        traffic_light_ids = set(
            self.traffic_lights()
        )

        return [
            node
            for node in self.net.getNodes()
            if node.getID() in traffic_light_ids
        ]

    # ------------------------------------------------------------------
    # EDGE DISCOVERY
    # ------------------------------------------------------------------

    def _incoming_routable_count(self) -> dict[str, int]:
        """
        Count routable incoming edges for every routable edge.

        Internal SUMO edges are ignored.
        """
        routable_edges = self.edges()

        incoming_count = {
            edge.getID(): 0
            for edge in routable_edges
        }

        for edge in routable_edges:
            for outgoing in edge.getOutgoing():

                outgoing_id = outgoing.getID()

                if outgoing_id.startswith(":"):
                    continue

                if outgoing_id in incoming_count:
                    incoming_count[outgoing_id] += 1

        return incoming_count

    def source_edges(self):
        """
        Dynamically discover network-entry edges.

        An edge is considered a source when no other routable edge
        points into it.

        This avoids hardcoding A1_EDGE, A2_EDGE, etc.
        """
        incoming_count = self._incoming_routable_count()

        return [
            edge
            for edge in self.edges()
            if incoming_count[edge.getID()] == 0
        ]

    def exit_edges(self):
        """
        Dynamically discover network-exit edges.

        An exit edge starts at a signalized junction and terminates
        outside the signalized junction network.

        This is important because the current network contains
        circulation edges such as:

            J1_EXIT -> A4_EDGE
            J2_EXIT -> B1_EDGE
            J3_EXIT -> C1_EDGE

        Those are valid network connections, but once traffic reaches
        an exit edge we should terminate a normal traffic-demand route
        rather than immediately send the vehicle back into the network.
        """
        traffic_light_ids = set(
            self.traffic_lights()
        )

        exits = []

        for edge in self.edges():

            from_node = edge.getFromNode()
            to_node = edge.getToNode()

            from_id = from_node.getID()
            to_id = to_node.getID()

            # Starts at a signalized junction...
            if from_id not in traffic_light_ids:
                continue

            # ...and leaves the signalized-junction network.
            if to_id in traffic_light_ids:
                continue

            exits.append(edge)

        return exits

    def _outgoing_edges(self, edge):
        """
        Return only normal, non-internal outgoing edges.
        """
        return [
            outgoing
            for outgoing in edge.getOutgoing()
            if not outgoing.getID().startswith(":")
        ]

    # ------------------------------------------------------------------
    # PATH DISCOVERY
    # ------------------------------------------------------------------

    def _find_paths(
        self,
        start,
        exit_ids: set[str],
    ) -> list[tuple[str, ...]]:
        """
        Discover bounded source-to-exit paths.

        Important behavior:

        If the current edge is an exit edge, the route immediately
        terminates.

        Therefore this will NOT generate:

            A1_EDGE
            -> J1_EXIT
            -> A4_EDGE
            -> J1_J2

        Instead it generates:

            A1_EDGE
            -> J1_EXIT

        Repeated edges are also forbidden.
        """

        results: list[tuple[str, ...]] = []

        def dfs(
            current,
            path: list[str],
            visited: set[str],
        ) -> None:

            current_id = current.getID()

            # ----------------------------------------------------------
            # EXIT CONDITION
            # ----------------------------------------------------------

            if current_id in exit_ids:

                # A route needs at least one transition.
                if len(path) >= 2:
                    results.append(
                        tuple(path)
                    )

                # VERY IMPORTANT:
                # Do not continue through the exit edge.
                return

            # ----------------------------------------------------------
            # MAXIMUM ROUTE LENGTH
            # ----------------------------------------------------------

            if len(path) >= self.max_route_edges:
                return

            # ----------------------------------------------------------
            # FOLLOW VALID OUTGOING EDGES
            # ----------------------------------------------------------

            for next_edge in self._outgoing_edges(
                current
            ):
                next_id = next_edge.getID()

                # Prevent loops.
                if next_id in visited:
                    continue

                dfs(
                    next_edge,
                    path + [next_id],
                    visited | {next_id},
                )

        dfs(
            start,
            [start.getID()],
            {start.getID()},
        )

        return results

    def discover_routes(self) -> list[RouteDefinition]:
        """
        Discover all bounded source-to-exit routes.
        """

        sources = self.source_edges()

        exit_ids = {
            edge.getID()
            for edge in self.exit_edges()
        }

        routes: list[RouteDefinition] = []

        route_number = 0

        for source in sources:

            paths = self._find_paths(
                source,
                exit_ids,
            )

            for path in paths:

                route_number += 1

                routes.append(
                    RouteDefinition(
                        route_id=(
                            f"route_{route_number:04d}"
                        ),
                        edges=path,
                    )
                )

        return routes

    # ------------------------------------------------------------------
    # ROUTE VALIDATION
    # ------------------------------------------------------------------

    def validate_route(
        self,
        route: RouteDefinition,
    ) -> bool:
        """
        Validate a discovered route.

        Validation checks:

        1. Route has at least two edges.
        2. No duplicate edges.
        3. Every edge exists in the network.
        4. Every consecutive pair is connected.
        5. Final edge is a discovered exit.
        """

        if len(route.edges) < 2:
            return False

        # No repeated edge.
        if len(route.edges) != len(
            set(route.edges)
        ):
            return False

        network_edge_ids = {
            edge.getID()
            for edge in self.edges()
        }

        # Every route edge must exist.
        if not all(
            edge_id in network_edge_ids
            for edge_id in route.edges
        ):
            return False

        edge_map = {
            edge.getID(): edge
            for edge in self.edges()
        }

        # Verify graph connectivity.
        for current_id, next_id in zip(
            route.edges,
            route.edges[1:],
        ):

            current_edge = edge_map[current_id]

            outgoing_ids = {
                outgoing.getID()
                for outgoing in self._outgoing_edges(
                    current_edge
                )
            }

            if next_id not in outgoing_ids:
                return False

        # Final edge must be an exit.
        exit_ids = {
            edge.getID()
            for edge in self.exit_edges()
        }

        if route.edges[-1] not in exit_ids:
            return False

        return True

    def validate_routes(
        self,
        routes: Iterable[RouteDefinition],
    ) -> list[RouteDefinition]:
        """
        Return only valid routes.
        """

        return [
            route
            for route in routes
            if self.validate_route(route)
        ]

    # ------------------------------------------------------------------
    # SUMO ROUTE FILE
    # ------------------------------------------------------------------

    def generate(
        self,
        output_file: str | Path,
        routes: Iterable[RouteDefinition] | None = None,
    ) -> list[RouteDefinition]:
        """
        Generate a SUMO .rou.xml route file.
        """

        output_file = Path(
            output_file
        )

        output_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        # Discover routes if none supplied.
        if routes is None:
            routes = self.discover_routes()

        routes = list(routes)

        # Validate before writing.
        routes = self.validate_routes(
            routes
        )

        # --------------------------------------------------------------
        # XML ROOT
        # --------------------------------------------------------------

        root = ET.Element(
            "routes"
        )

        # --------------------------------------------------------------
        # BASE INDIAN VEHICLE TYPE
        # --------------------------------------------------------------

        ET.SubElement(
            root,
            "vType",
            {
                "id": "indian_vehicle",
                "vClass": "passenger",
                "accel": "2.6",
                "decel": "4.5",
                "sigma": "0.5",
                "length": "4.5",
                "minGap": "2.0",
                "maxSpeed": "13.89",
            },
        )

        # --------------------------------------------------------------
        # ROUTES
        # --------------------------------------------------------------

        for route in routes:

            ET.SubElement(
                root,
                "route",
                {
                    "id": route.route_id,
                    "edges": " ".join(
                        route.edges
                    ),
                },
            )

        # --------------------------------------------------------------
        # WRITE XML
        # --------------------------------------------------------------

        tree = ET.ElementTree(
            root
        )

        ET.indent(
            tree,
            space="    ",
        )

        tree.write(
            output_file,
            encoding="UTF-8",
            xml_declaration=True,
        )

        return routes
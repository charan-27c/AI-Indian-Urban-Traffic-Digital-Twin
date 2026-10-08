from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Iterable
import random
import xml.etree.ElementTree as ET

from .route_generator import (
    DynamicRouteGenerator,
    RouteDefinition,
)


class TrafficScenario(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HEAVY = "heavy"
    PEAK = "peak"
    EMERGENCY = "emergency"


@dataclass(frozen=True)
class VehicleTypeDefinition:
    type_id: str
    vclass: str
    accel: float
    decel: float
    sigma: float
    length: float
    min_gap: float
    max_speed: float
    color: str | None = None


@dataclass(frozen=True)
class DemandProfile:
    """
    Traffic-demand configuration for one scenario.

    vehicles_per_hour is the approximate demand generated per
    selected route for normal traffic.

    emergency_vehicles is the number of emergency vehicles
    injected into the scenario.
    """

    scenario: TrafficScenario
    vehicles_per_hour: int
    emergency_vehicles: int
    begin: int = 0
    end: int = 1800


class TrafficDemandGenerator:
    """
    Generate SUMO traffic demand from dynamically discovered routes.

    Architecture:

        SUMO network
             |
             v
        Route discovery
             |
             v
        Valid RouteDefinition objects
             |
             v
        Traffic demand model
             |
             +---- LOW
             +---- MEDIUM
             +---- HEAVY
             +---- PEAK
             +---- EMERGENCY
             |
             v
        scenario.rou.xml

    No junction IDs or edge IDs are hardcoded.
    """

    DEFAULT_PROFILES = {
        TrafficScenario.LOW: DemandProfile(
            scenario=TrafficScenario.LOW,
            vehicles_per_hour=180,
            emergency_vehicles=0,
        ),
        TrafficScenario.MEDIUM: DemandProfile(
            scenario=TrafficScenario.MEDIUM,
            vehicles_per_hour=450,
            emergency_vehicles=0,
        ),
        TrafficScenario.HEAVY: DemandProfile(
            scenario=TrafficScenario.HEAVY,
            vehicles_per_hour=900,
            emergency_vehicles=0,
        ),
        TrafficScenario.PEAK: DemandProfile(
            scenario=TrafficScenario.PEAK,
            vehicles_per_hour=1400,
            emergency_vehicles=0,
        ),
        TrafficScenario.EMERGENCY: DemandProfile(
            scenario=TrafficScenario.EMERGENCY,
            vehicles_per_hour=900,
            emergency_vehicles=8,
        ),
    }

    def __init__(
        self,
        network_file: str | Path,
        seed: int = 42,
        max_route_edges: int = 8,
    ) -> None:
        self.network_file = Path(network_file)
        self.seed = seed

        self.route_generator = DynamicRouteGenerator(
            network_file=self.network_file,
            max_route_edges=max_route_edges,
        )

        self.random = random.Random(seed)

    # ------------------------------------------------------------------
    # VEHICLE TYPES
    # ------------------------------------------------------------------

    def vehicle_types(
        self,
    ) -> list[VehicleTypeDefinition]:
        """
        Indian-oriented SUMO vehicle classes.

        These are simulation classes. Your trained YOLO classes remain
        part of the perception layer and can later be mapped into these
        simulation categories.
        """

        return [
            VehicleTypeDefinition(
                type_id="car",
                vclass="passenger",
                accel=2.6,
                decel=4.5,
                sigma=0.5,
                length=4.5,
                min_gap=2.0,
                max_speed=13.89,
            ),
            VehicleTypeDefinition(
                type_id="motorcycle",
                vclass="motorcycle",
                accel=3.0,
                decel=5.0,
                sigma=0.7,
                length=2.2,
                min_gap=1.0,
                max_speed=15.0,
            ),
            VehicleTypeDefinition(
                type_id="scooter",
                vclass="motorcycle",
                accel=2.8,
                decel=5.0,
                sigma=0.7,
                length=2.0,
                min_gap=1.0,
                max_speed=13.89,
            ),
            VehicleTypeDefinition(
                type_id="auto_rickshaw",
                vclass="passenger",
                accel=2.2,
                decel=4.0,
                sigma=0.6,
                length=3.0,
                min_gap=1.5,
                max_speed=11.11,
            ),
            VehicleTypeDefinition(
                type_id="bus",
                vclass="bus",
                accel=1.2,
                decel=3.5,
                sigma=0.4,
                length=12.0,
                min_gap=3.0,
                max_speed=11.11,
            ),
            VehicleTypeDefinition(
                type_id="truck",
                vclass="truck",
                accel=1.4,
                decel=3.5,
                sigma=0.5,
                length=10.0,
                min_gap=3.0,
                max_speed=11.11,
            ),
            VehicleTypeDefinition(
                type_id="van",
                vclass="delivery",
                accel=2.0,
                decel=4.0,
                sigma=0.5,
                length=5.5,
                min_gap=2.0,
                max_speed=12.50,
            ),
            VehicleTypeDefinition(
                type_id="bicycle",
                vclass="bicycle",
                accel=1.5,
                decel=2.5,
                sigma=0.8,
                length=1.8,
                min_gap=0.8,
                max_speed=6.0,
            ),
            VehicleTypeDefinition(
                type_id="ambulance",
                vclass="emergency",
                accel=3.5,
                decel=5.5,
                sigma=0.3,
                length=5.5,
                min_gap=2.0,
                max_speed=16.67,
            ),
            VehicleTypeDefinition(
                type_id="fire_engine",
                vclass="emergency",
                accel=2.5,
                decel=5.0,
                sigma=0.3,
                length=8.0,
                min_gap=2.5,
                max_speed=15.0,
            ),
            VehicleTypeDefinition(
                type_id="police_vehicle",
                vclass="emergency",
                accel=3.2,
                decel=5.5,
                sigma=0.3,
                length=5.0,
                min_gap=2.0,
                max_speed=16.67,
            ),
        ]

    # ------------------------------------------------------------------
    # DEMAND PROFILES
    # ------------------------------------------------------------------

    def profile(
        self,
        scenario: TrafficScenario | str,
    ) -> DemandProfile:
        if isinstance(scenario, str):
            scenario = TrafficScenario(
                scenario.lower()
            )

        return self.DEFAULT_PROFILES[
            scenario
        ]

    # ------------------------------------------------------------------
    # ROUTE DISCOVERY
    # ------------------------------------------------------------------

    def discover_routes(self) -> list[RouteDefinition]:
        """
        Obtain valid routes from the dynamic route generator.
        """
        return self.route_generator.discover_routes()

    # ------------------------------------------------------------------
    # TRAFFIC MIX
    # ------------------------------------------------------------------

    def traffic_mix(
        self,
        scenario: TrafficScenario,
    ) -> dict[str, float]:
        """
        Indian urban traffic composition.

        Values represent probabilities and sum to 1.
        """

        if scenario == TrafficScenario.LOW:
            return {
                "car": 0.35,
                "motorcycle": 0.22,
                "scooter": 0.18,
                "auto_rickshaw": 0.10,
                "bus": 0.04,
                "truck": 0.04,
                "van": 0.03,
                "bicycle": 0.04,
            }

        if scenario == TrafficScenario.MEDIUM:
            return {
                "car": 0.32,
                "motorcycle": 0.22,
                "scooter": 0.18,
                "auto_rickshaw": 0.10,
                "bus": 0.05,
                "truck": 0.05,
                "van": 0.05,
                "bicycle": 0.03,
            }

        if scenario == TrafficScenario.HEAVY:
            return {
                "car": 0.28,
                "motorcycle": 0.24,
                "scooter": 0.20,
                "auto_rickshaw": 0.10,
                "bus": 0.06,
                "truck": 0.05,
                "van": 0.05,
                "bicycle": 0.02,
            }

        # Peak traffic.
        return {
            "car": 0.30,
            "motorcycle": 0.24,
            "scooter": 0.19,
            "auto_rickshaw": 0.10,
            "bus": 0.06,
            "truck": 0.04,
            "van": 0.05,
            "bicycle": 0.02,
        }

    def _choose_vehicle_type(
        self,
        mix: dict[str, float],
    ) -> str:
        vehicle_types = list(mix.keys())
        probabilities = list(mix.values())

        return self.random.choices(
            vehicle_types,
            weights=probabilities,
            k=1,
        )[0]

    # ------------------------------------------------------------------
    # ROUTE SELECTION
    # ------------------------------------------------------------------

    def _route_weights(
        self,
        routes: list[RouteDefinition],
    ) -> list[float]:
        """
        Prefer longer network routes slightly.

        This creates more traffic through downstream junctions
        without hardcoding those junctions.
        """

        weights = []

        for route in routes:
            length = len(route.edges)

            # Minimum weight = 1.
            # Longer routes receive slightly more demand.
            weight = 1.0 + (
                0.35 * max(
                    0,
                    length - 2,
                )
            )

            weights.append(weight)

        return weights

    # ------------------------------------------------------------------
    # XML HELPERS
    # ------------------------------------------------------------------

    def _add_vehicle_types(
        self,
        root: ET.Element,
    ) -> None:

        for vehicle_type in self.vehicle_types():

            attributes = {
                "id": vehicle_type.type_id,
                "vClass": vehicle_type.vclass,
                "accel": str(
                    vehicle_type.accel
                ),
                "decel": str(
                    vehicle_type.decel
                ),
                "sigma": str(
                    vehicle_type.sigma
                ),
                "length": str(
                    vehicle_type.length
                ),
                "minGap": str(
                    vehicle_type.min_gap
                ),
                "maxSpeed": str(
                    vehicle_type.max_speed
                ),
            }

            if vehicle_type.color:
                attributes["color"] = (
                    vehicle_type.color
                )

            ET.SubElement(
                root,
                "vType",
                attributes,
            )

    def _add_routes(
        self,
        root: ET.Element,
        routes: Iterable[RouteDefinition],
    ) -> None:

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

    # ------------------------------------------------------------------
    # NORMAL DEMAND
    # ------------------------------------------------------------------

    def _add_normal_demand(
        self,
        root: ET.Element,
        routes: list[RouteDefinition],
        profile: DemandProfile,
    ) -> None:

        if not routes:
            raise ValueError(
                "No valid routes available."
            )

        mix = self.traffic_mix(
            profile.scenario
        )

        route_weights = self._route_weights(
            routes
        )

        # Use one flow per route.
        for index, route in enumerate(routes):

            # Deterministic route-specific variation.
            variation = self.random.uniform(
                0.75,
                1.25,
            )

            route_rate = max(
                1,
                round(
                    profile.vehicles_per_hour
                    * variation
                ),
            )

            # Give each route its own vehicle type.
            vehicle_type = self._choose_vehicle_type(
                mix
            )

            ET.SubElement(
                root,
                "flow",
                {
                    "id": (
                        f"{profile.scenario.value}"
                        f"_flow_{index:04d}"
                    ),
                    "type": vehicle_type,
                    "route": route.route_id,
                    "begin": str(
                        profile.begin
                    ),
                    "end": str(
                        profile.end
                    ),
                    "vehsPerHour": str(
                        route_rate
                    ),
                    "departLane": "best",
                    "departSpeed": "max",
                },
            )

    # ------------------------------------------------------------------
    # EMERGENCY DEMAND
    # ------------------------------------------------------------------

    def _add_emergency_demand(
        self,
        root: ET.Element,
        routes: list[RouteDefinition],
        profile: DemandProfile,
    ) -> None:

        if profile.emergency_vehicles <= 0:
            return

        if not routes:
            raise ValueError(
                "Cannot generate emergency traffic without routes."
            )

        emergency_types = [
            "ambulance",
            "fire_engine",
            "police_vehicle",
        ]

        route_weights = self._route_weights(
            routes
        )

        for index in range(
            profile.emergency_vehicles
        ):

            route = self.random.choices(
                routes,
                weights=route_weights,
                k=1,
            )[0]

            vehicle_type = (
                emergency_types[
                    index % len(
                        emergency_types
                    )
                ]
            )

            # Spread emergency vehicles through
            # the simulation rather than inserting all
            # at time zero.
            time_span = max(
                1,
                profile.end - profile.begin,
            )

            depart_time = (
                profile.begin
                + self.random.randint(
                    0,
                    time_span - 1,
                )
            )

            ET.SubElement(
                root,
                "vehicle",
                {
                    "id": (
                        f"emergency_{index + 1:03d}"
                    ),
                    "type": vehicle_type,
                    "route": route.route_id,
                    "depart": str(
                        depart_time
                    ),
                    "departLane": "best",
                    "departSpeed": "max",
                },
            )

    # ------------------------------------------------------------------
    # GENERATE ONE SCENARIO
    # ------------------------------------------------------------------

    def generate_scenario(
        self,
        scenario: TrafficScenario | str,
        output_file: str | Path,
    ) -> list[RouteDefinition]:
        """
        Generate one complete SUMO route/demand file.

        Example:

            low.rou.xml
            medium.rou.xml
            heavy.rou.xml
            peak.rou.xml
            emergency.rou.xml
        """

        if isinstance(scenario, str):
            scenario = TrafficScenario(
                scenario.lower()
            )

        profile = self.profile(
            scenario
        )

        routes = self.discover_routes()

        routes = self.route_generator.validate_routes(
            routes
        )

        if not routes:
            raise ValueError(
                "No valid routes discovered from the SUMO network."
            )

        output_file = Path(
            output_file
        )

        output_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        root = ET.Element(
            "routes"
        )

        # Vehicle types.
        self._add_vehicle_types(
            root
        )

        # Route definitions.
        self._add_routes(
            root,
            routes,
        )

        # Normal traffic.
        self._add_normal_demand(
            root,
            routes,
            profile,
        )

        # Emergency traffic.
        self._add_emergency_demand(
            root,
            routes,
            profile,
        )

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

    # ------------------------------------------------------------------
    # GENERATE ALL SCENARIOS
    # ------------------------------------------------------------------

    def generate_all(
        self,
        output_directory: str | Path,
    ) -> dict[
        TrafficScenario,
        Path,
    ]:
        """
        Generate all standard traffic scenarios.
        """

        output_directory = Path(
            output_directory
        )

        output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        generated = {}

        for scenario in TrafficScenario:

            output_file = (
                output_directory
                / f"{scenario.value}.rou.xml"
            )

            self.generate_scenario(
                scenario=scenario,
                output_file=output_file,
            )

            generated[
                scenario
            ] = output_file

        return generated
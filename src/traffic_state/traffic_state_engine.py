from typing import Dict, Iterable
from src.incidents.topology_impact import (
    IncidentTopologyImpactManager,
)

from src.incidents.pressure_impact import (
    IncidentPressureAdjustment,
)

from .models import (
    VehicleState,
    LaneTrafficState,
    ApproachTrafficState,
    MovementTrafficState,
    JunctionTrafficState,
    NetworkTrafficState,
)


# ============================================================
# TRAFFIC STATE ENGINE
# ============================================================

class TrafficStateEngine:
    """
    Dynamic real-time traffic state engine.

    Runtime topology is supplied externally by:
        - SumoTopology
        - RuntimeTopologyManager

    No fixed:
        - junction IDs
        - approach IDs
        - lane IDs
        - movement IDs
        - directions
        - lane counts

    are hard-coded in this engine.
    """

    def __init__(
        self,
        queue_speed_threshold: float = 1.0,
        free_flow_speed: float = 13.89,
    ):

        # ----------------------------------------------------
        # Configuration
        # ----------------------------------------------------

        self.queue_speed_threshold = (
            queue_speed_threshold
        )

        self.free_flow_speed = (
            free_flow_speed
        )

        # ----------------------------------------------------
        # Latest tracked vehicles
        # ----------------------------------------------------

        self.vehicles: Dict[
            str,
            VehicleState
        ] = {}

        # ----------------------------------------------------
        # Current network state
        # ----------------------------------------------------

        self.state = NetworkTrafficState()

        # ----------------------------------------------------
        # Runtime topology
        # ----------------------------------------------------

        self._topology = None

        # ----------------------------------------------------
        # Dynamic topology indexes
        #
        # lane -> approach
        # lane -> junction
        # lane -> movements
        # movement -> junction
        # ----------------------------------------------------

        self.lane_to_approach: Dict[
            str,
            str
        ] = {}

        self.lane_to_junction: Dict[
            str,
            str
        ] = {}

        self.lane_to_movements: Dict[
            str,
            list[str]
        ] = {}

        self.movement_to_junction: Dict[
            str,
            str
        ] = {}

    # ========================================================
    # TOPOLOGY
    # ========================================================

    def set_topology(
        self,
        topology,
    ) -> None:
        """
        Attach the runtime SUMO topology.

        The topology is discovered dynamically.

        No junction or movement IDs are hard-coded.
        """

        self._topology = topology

        self._build_topology_indexes()

    # ========================================================
    # TOPOLOGY INDEXES
    # ========================================================

    def _build_topology_indexes(
        self,
    ) -> None:

        self.lane_to_approach.clear()

        self.lane_to_junction.clear()

        self.lane_to_movements.clear()

        self.movement_to_junction.clear()

        if self._topology is None:
            return

        # ----------------------------------------------------
        # Discover junctions
        # ----------------------------------------------------

        if hasattr(
            self._topology,
            "discover_all_junctions",
        ):

            junctions = (
                self._topology
                .discover_all_junctions()
            )

        elif hasattr(
            self._topology,
            "get_junctions",
        ):

            junctions = (
                self._topology
                .get_junctions()
            )

        elif hasattr(
            self._topology,
            "junctions",
        ):

            junctions = (
                self._topology.junctions
            )

        else:

            raise AttributeError(
                "Runtime topology must provide "
                "discover_all_junctions(), "
                "get_junctions(), or junctions."
            )

        # ----------------------------------------------------
        # Normalize junction collection
        # ----------------------------------------------------

        if isinstance(
            junctions,
            dict,
        ):

            junction_items = (
                junctions.items()
            )

        else:

            junction_items = (
                (
                    junction.junction_id,
                    junction,
                )
                for junction in junctions
            )

        # ====================================================
        # JUNCTION
        # ====================================================

        for junction_id, junction in (
            junction_items
        ):

            # ------------------------------------------------
            # Approaches
            # ------------------------------------------------

            if hasattr(
                junction,
                "get_approaches",
            ):

                approaches = (
                    junction.get_approaches()
                )

            elif hasattr(
                junction,
                "approaches",
            ):

                approaches = (
                    junction.approaches.values()
                    if isinstance(
                        junction.approaches,
                        dict,
                    )
                    else junction.approaches
                )

            else:

                approaches = []

            # ------------------------------------------------
            # Lanes
            # ------------------------------------------------

            for approach in approaches:

                approach_id = (
                    approach.approach_id
                )

                if hasattr(
                    approach,
                    "get_lanes",
                ):

                    lanes = (
                        approach.get_lanes()
                    )

                elif hasattr(
                    approach,
                    "lanes",
                ):

                    lanes = (
                        approach.lanes.values()
                        if isinstance(
                            approach.lanes,
                            dict,
                        )
                        else approach.lanes
                    )

                else:

                    lanes = []

                for lane in lanes:

                    lane_id = (
                        lane.lane_id
                    )

                    self.lane_to_approach[
                        lane_id
                    ] = approach_id

                    self.lane_to_junction[
                        lane_id
                    ] = junction_id

            # ------------------------------------------------
            # Movements
            # ------------------------------------------------

            if hasattr(
                junction,
                "get_movements",
            ):

                movements = (
                    junction.get_movements()
                )

            elif hasattr(
                junction,
                "movements",
            ):

                movements = (
                    junction.movements.values()
                    if isinstance(
                        junction.movements,
                        dict,
                    )
                    else junction.movements
                )

            else:

                movements = []

            for movement in movements:

                movement_id = (
                    movement.movement_id
                )

                from_lane_id = (
                    movement.from_lane_id
                )

                self.movement_to_junction[
                    movement_id
                ] = junction_id

                self.lane_to_movements.setdefault(
                    from_lane_id,
                    [],
                ).append(
                    movement_id
                )

    # ========================================================
    # DYNAMIC MOVEMENT LOOKUP
    # ========================================================

    def _find_movement_for_lane(
        self,
        lane_id: str,
    ):
        """
        Find the dynamically discovered movement
        that starts from the supplied lane.

        Example:

            A1_EDGE_0
                 |
                 v
              J1_M1

        The relationship is discovered from
        runtime topology rather than hard-coded.
        """

        if self._topology is None:
            return None

        # ----------------------------------------------------
        # Preferred dynamic discovery
        # ----------------------------------------------------

        if hasattr(
            self._topology,
            "discover_all_junctions",
        ):

            junctions = (
                self._topology
                .discover_all_junctions()
            )

        # ----------------------------------------------------
        # Compatibility fallback
        # ----------------------------------------------------

        elif hasattr(
            self._topology,
            "get_junctions",
        ):

            junctions = (
                self._topology
                .get_junctions()
            )

        elif hasattr(
            self._topology,
            "junctions",
        ):

            junctions = (
                self._topology.junctions
            )

        else:

            return None

        # ----------------------------------------------------
        # Normalize junction collection
        # ----------------------------------------------------

        if isinstance(
            junctions,
            dict,
        ):

            junction_values = (
                junctions.values()
            )

        else:

            junction_values = junctions

        # ----------------------------------------------------
        # Search movements
        # ----------------------------------------------------

        for junction in junction_values:

            if hasattr(
                junction,
                "get_movements",
            ):

                movements = (
                    junction.get_movements()
                )

            elif hasattr(
                junction,
                "movements",
            ):

                movements = (
                    junction.movements.values()
                    if isinstance(
                        junction.movements,
                        dict,
                    )
                    else junction.movements
                )

            else:

                continue

            for movement in movements:

                if (
                    movement.from_lane_id
                    == lane_id
                ):

                    return movement

        return None

    # ========================================================
    # VEHICLES
    # ========================================================

    def update_vehicle(
        self,
        vehicle: VehicleState,
    ) -> None:

        self.vehicles[
            vehicle.vehicle_id
        ] = vehicle

    # ========================================================

    def remove_vehicle(
        self,
        vehicle_id: str,
    ) -> None:

        self.vehicles.pop(
            vehicle_id,
            None,
        )

    # ========================================================

    def clear_vehicles(
        self,
    ) -> None:

        self.vehicles.clear()

    # ========================================================
    # QUEUE
    # ========================================================

    def is_queued(
        self,
        vehicle: VehicleState,
    ) -> bool:

        return (
            vehicle.stopped
            or
            vehicle.speed
            <= self.queue_speed_threshold
        )

    # ========================================================
    # AVERAGE SPEED
    # ========================================================

    def calculate_average_speed(
        self,
        vehicles: Iterable[
            VehicleState
        ],
    ) -> float:

        vehicles = list(
            vehicles
        )

        if not vehicles:
            return 0.0

        return (
            sum(
                vehicle.speed
                for vehicle in vehicles
            )
            /
            len(vehicles)
        )

    # ========================================================
    # PRESSURE
    # ========================================================

    def calculate_pressure(
        self,
        vehicle_count: int,
        queue_length: int,
    ) -> float:

        return (
            float(vehicle_count)
            +
            2.0 * float(queue_length)
        )

    # ========================================================
    # VEHICLES BY LANE
    # ========================================================

    def _vehicles_by_lane(
        self,
    ) -> Dict[
        str,
        list[VehicleState]
    ]:

        result: Dict[
            str,
            list[VehicleState]
        ] = {}

        for vehicle in (
            self.vehicles.values()
        ):

            result.setdefault(
                vehicle.lane_id,
                [],
            ).append(
                vehicle
            )

        return result

    # ========================================================
    # BUILD LANE STATES
    # ========================================================

    def _build_lane_states(
        self,
    ) -> Dict[
        str,
        LaneTrafficState
    ]:

        lane_states: Dict[
            str,
            LaneTrafficState
        ] = {}

        vehicles_by_lane = (
            self._vehicles_by_lane()
        )

        # ----------------------------------------------------
        # Create states for every discovered topology lane.
        # ----------------------------------------------------

        for lane_id in (
            self.lane_to_approach.keys()
        ):

            vehicles = (
                vehicles_by_lane.get(
                    lane_id,
                    [],
                )
            )

            # ------------------------------------------------
            # Edge
            # ------------------------------------------------

            edge_id = ""

            if vehicles:

                edge_id = (
                    vehicles[0].edge_id
                )

            # ------------------------------------------------
            # Queue
            # ------------------------------------------------

            queue_length = sum(
                1
                for vehicle
                in vehicles
                if self.is_queued(
                    vehicle
                )
            )

            # ------------------------------------------------
            # Speed
            # ------------------------------------------------

            average_speed = (
                self.calculate_average_speed(
                    vehicles
                )
            )

            # ------------------------------------------------
            # Waiting
            # ------------------------------------------------

            waiting_time = sum(
                vehicle.waiting_time
                for vehicle in vehicles
            )

            # ------------------------------------------------
            # Delay
            # ------------------------------------------------

            delay = sum(
                max(
                    0.0,
                    self.free_flow_speed
                    -
                    vehicle.speed,
                )
                for vehicle
                in vehicles
            )

            # ------------------------------------------------
            # Pressure
            # ------------------------------------------------

            pressure = (
                self.calculate_pressure(
                    len(vehicles),
                    queue_length,
                )
            )

            lane_states[
                lane_id
            ] = LaneTrafficState(
                lane_id=lane_id,
                edge_id=edge_id,
                vehicle_count=len(
                    vehicles
                ),
                queue_length=queue_length,
                average_speed=average_speed,
                waiting_time=waiting_time,
                delay=delay,
                pressure=pressure,
            )

        return lane_states

    # ========================================================
    # BUILD JUNCTION STATES
    # ========================================================

    def _build_junction_states(
        self,
        lane_states: Dict[
            str,
            LaneTrafficState
        ],
    ) -> Dict[
        str,
        JunctionTrafficState
    ]:

        junction_states: Dict[
            str,
            JunctionTrafficState
        ] = {}

        # ----------------------------------------------------
        # Create all discovered junctions dynamically.
        # ----------------------------------------------------

        for lane_id, junction_id in (
            self.lane_to_junction.items()
        ):

            if junction_id not in (
                junction_states
            ):

                junction_states[
                    junction_id
                ] = JunctionTrafficState(
                    junction_id=junction_id
                )

            junction = (
                junction_states[
                    junction_id
                ]
            )

            lane_state = (
                lane_states[lane_id]
            )

            junction.lanes[
                lane_id
            ] = lane_state

            junction.vehicle_count += (
                lane_state.vehicle_count
            )

            junction.queue_length += (
                lane_state.queue_length
            )

            junction.waiting_time += (
                lane_state.waiting_time
            )

            junction.delay += (
                lane_state.delay
            )

            junction.pressure += (
                lane_state.pressure
            )

        # ----------------------------------------------------
        # Calculate junction average speeds.
        # ----------------------------------------------------

        for junction in (
            junction_states.values()
        ):

            total_vehicles = (
                junction.vehicle_count
            )

            if total_vehicles > 0:

                weighted_speed = sum(
                    lane.average_speed
                    *
                    lane.vehicle_count
                    for lane
                    in junction.lanes.values()
                )

                junction.average_speed = (
                    weighted_speed
                    /
                    total_vehicles
                )

        return junction_states

    # ========================================================
    # BUILD APPROACH STATES
    # ========================================================

    def _build_approach_states(
        self,
        junction_states: Dict[
            str,
            JunctionTrafficState
        ],
    ) -> None:

        for junction_id, junction in (
            junction_states.items()
        ):

            for lane_id, lane_state in (
                junction.lanes.items()
            ):

                approach_id = (
                    self.lane_to_approach.get(
                        lane_id
                    )
                )

                if approach_id is None:
                    continue

                if approach_id not in (
                    junction.approaches
                ):

                    junction.approaches[
                        approach_id
                    ] = ApproachTrafficState(
                        approach_id=approach_id
                    )

                approach = (
                    junction.approaches[
                        approach_id
                    ]
                )

                approach.vehicle_count += (
                    lane_state.vehicle_count
                )

                approach.queue_length += (
                    lane_state.queue_length
                )

                approach.waiting_time += (
                    lane_state.waiting_time
                )

                approach.delay += (
                    lane_state.delay
                )

                approach.pressure += (
                    lane_state.pressure
                )

                approach.lanes.add(
                    lane_id
                )

            # ------------------------------------------------
            # Calculate weighted approach speed.
            # ------------------------------------------------

            for approach in (
                junction.approaches.values()
            ):

                if approach.vehicle_count <= 0:
                    approach.average_speed = 0.0
                    continue

                weighted_speed = 0.0

                for lane_id in approach.lanes:

                    lane_state = (
                        junction.lanes.get(
                            lane_id
                        )
                    )

                    if lane_state is None:
                        continue

                    weighted_speed += (
                        lane_state.average_speed
                        *
                        lane_state.vehicle_count
                    )

                approach.average_speed = (
                    weighted_speed
                    /
                    approach.vehicle_count
                )

    # ========================================================
    # BUILD MOVEMENT STATES
    # ========================================================

    def _build_movement_states(
        self,
    ) -> Dict[
        str,
        MovementTrafficState
    ]:

        movement_states = {}

        # ----------------------------------------------------
        # Group vehicles by dynamically discovered movement.
        # ----------------------------------------------------

        for vehicle in (
            self.vehicles.values()
        ):

            movement = (
                self._find_movement_for_lane(
                    vehicle.lane_id
                )
            )

            if movement is None:
                continue

            movement_id = (
                movement.movement_id
            )

            if movement_id not in (
                movement_states
            ):

                movement_states[
                    movement_id
                ] = {
                    "movement": movement,
                    "vehicles": [],
                }

            movement_states[
                movement_id
            ][
                "vehicles"
            ].append(
                vehicle
            )

        # ----------------------------------------------------
        # Build aggregated movement states.
        # ----------------------------------------------------

        result: Dict[
            str,
            MovementTrafficState
        ] = {}

        for movement_id, data in (
            movement_states.items()
        ):

            movement = (
                data["movement"]
            )

            vehicles = (
                data["vehicles"]
            )

            count = len(
                vehicles
            )

            queue = sum(
                1
                for vehicle
                in vehicles
                if self.is_queued(
                    vehicle
                )
            )

            average_speed = (
                sum(
                    vehicle.speed
                    for vehicle
                    in vehicles
                )
                /
                count
                if count
                else 0.0
            )

            waiting_time = sum(
                vehicle.waiting_time
                for vehicle
                in vehicles
            )

            delay = sum(
                max(
                    0.0,
                    self.free_flow_speed
                    -
                    vehicle.speed,
                )
                for vehicle
                in vehicles
            )

            pressure = (
                queue
                +
                waiting_time
            )

            lanes = {
                vehicle.lane_id
                for vehicle
                in vehicles
                if vehicle.lane_id
            }

            result[
                movement_id
            ] = MovementTrafficState(
                movement_id=movement_id,
                junction_id=(
                    movement.junction_id
                ),
                vehicle_count=count,
                queue_length=queue,
                average_speed=average_speed,
                waiting_time=waiting_time,
                delay=delay,
                pressure=pressure,
                lanes=lanes,
            )

        return result

    # ========================================================
    # BUILD COMPLETE STATE
    # ========================================================

    def build_state(
        self,
        timestamp: float,
    ) -> NetworkTrafficState:

        # ----------------------------------------------------
        # Lane states
        # ----------------------------------------------------

        lane_states = (
            self._build_lane_states()
        )

        # ----------------------------------------------------
        # Junction states
        # ----------------------------------------------------

        junction_states = (
            self._build_junction_states(
                lane_states
            )
        )

        # ----------------------------------------------------
        # Approach states
        # ----------------------------------------------------

        self._build_approach_states(
            junction_states
        )

        # ----------------------------------------------------
        # Movement states
        # ----------------------------------------------------

        movement_states = (
            self._build_movement_states()
        )

        # ----------------------------------------------------
        # Attach movement states to junctions
        # ----------------------------------------------------

        for movement_state in (
            movement_states.values()
        ):

            junction_id = (
                movement_state.junction_id
            )

            if junction_id not in (
                junction_states
            ):
                continue

            junction_states[
                junction_id
            ].movements[
                movement_state.movement_id
            ] = movement_state

        # ----------------------------------------------------
        # Network aggregation
        # ----------------------------------------------------

        network = (
            NetworkTrafficState(
                timestamp=timestamp
            )
        )

        network.junctions = (
            junction_states
        )

        # ----------------------------------------------------
        # Network vehicle count
        # ----------------------------------------------------

        network.vehicle_count = sum(
            junction.vehicle_count
            for junction
            in junction_states.values()
        )

        # ----------------------------------------------------
        # Network queue
        # ----------------------------------------------------

        network.queue_length = sum(
            junction.queue_length
            for junction
            in junction_states.values()
        )

        # ----------------------------------------------------
        # Network waiting time
        # ----------------------------------------------------

        network.waiting_time = sum(
            junction.waiting_time
            for junction
            in junction_states.values()
        )

        # ----------------------------------------------------
        # Network delay
        # ----------------------------------------------------

        network.delay = sum(
            junction.delay
            for junction
            in junction_states.values()
        )

        # ----------------------------------------------------
        # Network pressure
        # ----------------------------------------------------

        network.pressure = sum(
            junction.pressure
            for junction
            in junction_states.values()
        )

        # ----------------------------------------------------
        # Network average speed
        # ----------------------------------------------------

        if network.vehicle_count > 0:

            network.average_speed = (
                sum(
                    junction.average_speed
                    *
                    junction.vehicle_count
                    for junction
                    in junction_states.values()
                )
                /
                network.vehicle_count
            )

        # ----------------------------------------------------
        # Save current state
        # ----------------------------------------------------

        self.state = network

        return network

    # ========================================================
    # CURRENT STATE
    # ========================================================

    def get_state(
        self,
    ) -> NetworkTrafficState:

        return self.state
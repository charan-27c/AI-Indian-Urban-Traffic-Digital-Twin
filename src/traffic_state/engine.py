from typing import Dict, Iterable, Optional
from src.incidents.topology_impact import IncidentTopologyImpactManager
from src.incidents.state_impact import IncidentStateImpactResolver

from .models import (
    VehicleState,
    LaneTrafficState,
    ApproachTrafficState,
    MovementTrafficState,
    JunctionTrafficState,
    NetworkTrafficState,
)


class TrafficStateEngine:
    """
    Dynamic real-time traffic state engine.

    Runtime topology is supplied externally by SumoTopology
    / RuntimeTopologyManager.

    No fixed:
        - junction IDs
        - approach IDs
        - lane IDs
        - movement IDs
        - directions
        - lane counts
    """

    def __init__(
        self,
        queue_speed_threshold: float = 1.0,
        free_flow_speed: float = 13.89,
    ):
      

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

        self.topology = None
        self.incident_impact_manager = IncidentTopologyImpactManager()

        self.incident_state_resolver = IncidentStateImpactResolver(
            self.incident_impact_manager
      )
        # ----------------------------------------------------
        # Lane -> approach
        # Lane -> junction
        # Lane -> movement
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
        Register runtime topology.

        The topology object is expected to provide
        discovered junctions and their approaches,
        lanes and movements.
        """

        self.topology = topology

        self._build_topology_indexes()

    def set_incident_impact_manager(
        self,
        incident_impact_manager: IncidentTopologyImpactManager,
    ) -> None:
        """
        Attach runtime incident topology impact manager.

        Incident impacts temporarily block lanes/movements
        without modifying the base topology.
        """
        self.incident_impact_manager = incident_impact_manager

    def _lane_is_blocked(
        self,
        lane_id: str,
    ) -> bool:
        if self.incident_impact_manager is None:
            return False
        return self.incident_impact_manager.is_lane_blocked(
            lane_id
        )

    def _movement_is_blocked(
        self,
        movement_id: str,
        source_lanes: Iterable[str],
    ) -> bool:
        if self.incident_impact_manager is None:
            return False
        # Explicit movement blockage
        if self.incident_impact_manager.is_movement_blocked(
            movement_id
        ):
            return True
        # Safety rule:
        # if any source lane is blocked, this movement
        # cannot be served.
        for lane_id in source_lanes:
            if self._lane_is_blocked(lane_id):
                return True
        return False

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

        if self.topology is None:
            return

        # ----------------------------------------------------
        # Support either:
        #
        # topology.junctions
        #
        # or:
        #
        # topology.get_junctions()
        # ----------------------------------------------------

        if hasattr(
            self.topology,
            "get_junctions",
        ):

            junctions = (
                self.topology.get_junctions()
            )

        elif hasattr(
            self.topology,
            "junctions",
        ):

            junctions = (
                self.topology.junctions
            )

        else:

            raise AttributeError(
                "Runtime topology must provide "
                "get_junctions() or junctions."
            )

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
                for junction
                in junctions
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

            else:

                approaches = (
                    junction.approaches.values()
                )

            for approach in approaches:

                approach_id = (
                    approach.approach_id
                )

                # --------------------------------------------
                # Lanes
                # --------------------------------------------

                if hasattr(
                    approach,
                    "get_lanes",
                ):

                    lanes = (
                        approach.get_lanes()
                    )

                else:

                    lanes = approach.lanes

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

            else:

                movements = (
                    junction.movements.values()
                )

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

            edge_id = ""

            if vehicles:
                edge_id = (
                    vehicles[0].edge_id
                )

            queue_length = sum(
                1
                for vehicle
                in vehicles
                if self.is_queued(vehicle)
            )

            average_speed = (
                self.calculate_average_speed(
                    vehicles
                )
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
                self.calculate_pressure(
                    len(vehicles),
                    queue_length,
                )
            )
            blocked = self._lane_is_blocked(lane_id)

            effective_pressure = (
                pressure * 2.0 + 100.0
                if blocked
                else pressure
            )

            available = not blocked

            lane_states[lane_id] = (
                LaneTrafficState(
                    lane_id=lane_id,
                    edge_id=edge_id,
                    vehicle_count=len(vehicles),
                    queue_length=queue_length,
                    average_speed=average_speed,
                    waiting_time=waiting_time,
                    delay=delay,
                    pressure=pressure,
                    blocked=blocked,
                    effective_pressure=effective_pressure,
                    available=available
                )
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

            if junction_id not in junction_states:

                junction_states[
                    junction_id
                ] = JunctionTrafficState(
                    junction_id=junction_id
                )

            junction = (
                junction_states[junction_id]
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

                # Weighted speed later.
                if (
                    lane_state.vehicle_count
                    > 0
                ):

                    old_count = (
                        approach.vehicle_count
                        -
                        lane_state.vehicle_count
                    )

                    if (
                        old_count
                        +
                        lane_state.vehicle_count
                        > 0
                    ):

                        previous_weight = (
                            approach.average_speed
                            *
                            old_count
                        )

                        current_weight = (
                            lane_state.average_speed
                            *
                            lane_state.vehicle_count
                        )

                        approach.average_speed = (
                            (
                                previous_weight
                                +
                                current_weight
                            )
                            /
                            (
                                old_count
                                +
                                lane_state.vehicle_count
                            )
                        )

    # ========================================================
    # BUILD MOVEMENT STATES
    # ========================================================

    def _build_movement_states(
        self,
        junction_states: Dict[
            str,
            JunctionTrafficState
        ],
    ) -> None:

        for (
            movement_id,
            junction_id,
        ) in (
            self.movement_to_junction.items()
        ):

            junction = (
                junction_states.get(
                    junction_id
                )
            )

            if junction is None:
                continue

            # --------------------------------------------
            # Find source lane(s)
            # --------------------------------------------

            source_lanes = [
                lane_id
                for lane_id, movements
                in self.lane_to_movements.items()
                if movement_id in movements
            ]
            if not source_lanes:
                continue        

            # ----------------------------------------------------
            # Aggregate all source lanes
            # ----------------------------------------------------

            total_vehicle_count = 0
            total_queue_length = 0

            total_waiting_time = 0.0
            total_delay = 0.0

            weighted_speed = 0.0

            active_lane_ids = set()

            for lane_id in source_lanes:

                lane_state = (
                    junction.lanes.get(
                    lane_id
                )
            )

                if lane_state is None:
                    continue

                # Movement-level state is initially
                # derived from the lane state.
                total_vehicle_count += (
                    lane_state.vehicle_count
                )

                total_queue_length += (
                    lane_state.queue_length
                )

                total_waiting_time += (
                    lane_state.waiting_time
                )

                total_delay += (
                    lane_state.delay
                )
                weighted_speed += (
                     lane_state.average_speed
                    * lane_state.vehicle_count
                )
                if lane_state.vehicle_count > 0:

                    active_lane_ids.add(
                    lane_id
                )
               
                if total_vehicle_count > 0:

                    average_speed = (
                    weighted_speed
                / total_vehicle_count
               )

                else:

                   average_speed = 0.0 

                pressure = (
                    self.calculate_pressure(
                        total_vehicle_count,
                        total_queue_length,
                    )
                )
                blocked = self._movement_is_blocked(
                    movement_id,
                    source_lanes,
                )

                effective_pressure = (
                    pressure * 2.0 + 100.0
                    if blocked
                    else pressure
              )

                available = not blocked

                junction.movements[
                    movement_id
                ] = MovementTrafficState(
                    movement_id=movement_id,
                    junction_id=junction_id,
                    vehicle_count=total_vehicle_count,
                    queue_length=total_queue_length,
                    average_speed=average_speed,
                    waiting_time=total_waiting_time,
                    delay=total_delay,
                    pressure=pressure,
                    lanes=set(
                       active_lane_ids
                    ),
                    blocked=blocked,
                    effective_pressure=effective_pressure,
                    available=available
                )

    # ========================================================
    # BUILD COMPLETE STATE
    # ========================================================

    def build_state(
        self,
        timestamp: float,
    ) -> NetworkTrafficState:

        lane_states = (
            self._build_lane_states()
        )

        junction_states = (
            self._build_junction_states(
                lane_states
            )
        )

        self._build_approach_states(
            junction_states
        )

        self._build_movement_states(
            junction_states
        )

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

        network.vehicle_count = sum(
            junction.vehicle_count
            for junction
            in junction_states.values()
        )

        network.queue_length = sum(
            junction.queue_length
            for junction
            in junction_states.values()
        )

        network.waiting_time = sum(
            junction.waiting_time
            for junction
            in junction_states.values()
        )

        network.delay = sum(
            junction.delay
            for junction
            in junction_states.values()
        )

        network.pressure = sum(
            junction.pressure
            for junction
            in junction_states.values()
        )

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

        self.state = network

        return network

    # ========================================================
    # CURRENT STATE
    # ========================================================

    def get_state(
        self,
    ) -> NetworkTrafficState:

        return self.state

from dataclasses import dataclass, field
from typing import Dict, List
from src.traffic_state.traffic_state_engine import (
    TrafficStateEngine,
    VehicleState,
)


def test_traffic_state_engine():

    engine = TrafficStateEngine()

    engine.update_vehicle(
        VehicleState(
            vehicle_id="vehicle_1",
            vehicle_class="car",
            lane_id="A1_EDGE_0",
            edge_id="A1_EDGE",
            speed=0.0,
            waiting_time=5.0,
            stopped=True,
        )
    )

    engine.update_vehicle(
        VehicleState(
            vehicle_id="vehicle_2",
            vehicle_class="motorcycle",
            lane_id="A1_EDGE_0",
            edge_id="A1_EDGE",
            speed=8.0,
            waiting_time=1.0,
            stopped=False,
        )
    )

    state = engine.build_state(
        timestamp=10.0
    )

    assert state.vehicle_count == 2
    assert state.queue_length == 1
    assert state.average_speed == 4.0
    assert state.waiting_time == 6.0
    assert state.delay == 6.0
    assert state.density == 2.0
    assert state.occupancy == 0.5
    assert state.flow_rate == 2.0
    assert state.pressure == 7.0


# ============================================================
# VEHICLE STATE
# ============================================================

@dataclass
class VehicleState:
    """
    Runtime state of one detected/tracked vehicle.
    """

    vehicle_id: str
    vehicle_class: str

    lane_id: str
    edge_id: str

    speed: float
    waiting_time: float

    stopped: bool = False


# ============================================================
# JUNCTION TRAFFIC STATE
# ============================================================

@dataclass
class JunctionTrafficState:
    """
    Aggregated traffic state for one junction.

    Junction IDs are supplied dynamically by the topology
    system. No fixed junction names are assumed.
    """

    junction_id: str

    vehicle_count: int = 0
    queue_length: int = 0

    average_speed: float = 0.0
    waiting_time: float = 0.0

    pressure: float = 0.0


# ============================================================
# NETWORK TRAFFIC STATE
# ============================================================

@dataclass
class NetworkTrafficState:
    """
    Complete real-time network traffic state.
    """

    timestamp: float

    vehicle_count: int = 0
    queue_length: int = 0

    average_speed: float = 0.0

    density: float = 0.0
    occupancy: float = 0.0

    waiting_time: float = 0.0
    delay: float = 0.0

    flow_rate: float = 0.0
    pressure: float = 0.0

    junctions: Dict[
        str,
        JunctionTrafficState
    ] = field(default_factory=dict)


# ============================================================
# TRAFFIC STATE ENGINE
# ============================================================

class TrafficStateEngine:
    """
    Real-time traffic state aggregation engine.

    Pipeline:

        VehicleState
             |
             v
        Vehicle Registry
             |
             v
        Aggregation
             |
             +--> vehicle count
             +--> queue length
             +--> average speed
             +--> waiting time
             +--> density
             +--> occupancy
             +--> delay
             +--> flow
             +--> pressure
             |
             v
        NetworkTrafficState

    Important:

    This engine does NOT assume:
        - fixed junction IDs
        - fixed lanes
        - four approaches
        - four directions
        - fixed vehicle classes

    Topology information is supplied separately by the
    runtime SUMO topology system.
    """

    def __init__(self):

        # ----------------------------------------------------
        # ACTIVE VEHICLE REGISTRY
        # ----------------------------------------------------

        self._vehicles: Dict[
            str,
            VehicleState
        ] = {}

    # ========================================================
    # UPDATE VEHICLE
    # ========================================================

    def update_vehicle(
        self,
        vehicle: VehicleState,
    ) -> None:
        """
        Add a new vehicle or update an existing vehicle.

        The latest observation replaces the previous state.
        """

        if not isinstance(
            vehicle,
            VehicleState
        ):
            raise TypeError(
                "update_vehicle() expects "
                "a VehicleState object."
            )

        if not vehicle.vehicle_id:
            raise ValueError(
                "Vehicle ID cannot be empty."
            )

        self._vehicles[
            vehicle.vehicle_id
        ] = vehicle

    # ========================================================
    # UPDATE MULTIPLE VEHICLES
    # ========================================================

    def update_vehicles(
        self,
        vehicles: List[VehicleState],
    ) -> None:
        """
        Update multiple vehicle states at once.
        """

        for vehicle in vehicles:

            self.update_vehicle(
                vehicle
            )

    # ========================================================
    # REMOVE VEHICLE
    # ========================================================

    def remove_vehicle(
        self,
        vehicle_id: str,
    ) -> None:
        """
        Remove a vehicle that has left the active
        observation/network area.
        """

        self._vehicles.pop(
            vehicle_id,
            None,
        )

    # ========================================================
    # CLEAR VEHICLES
    # ========================================================

    def clear(self) -> None:
        """
        Remove all active vehicles.
        """

        self._vehicles.clear()

    # ========================================================
    # VEHICLE COUNT
    # ========================================================

    def vehicle_count(self) -> int:
        """
        Return number of currently active vehicles.
        """

        return len(
            self._vehicles
        )

    # ========================================================
    # GET VEHICLES
    # ========================================================

    def get_vehicles(
        self,
    ) -> Dict[str, VehicleState]:
        """
        Return a copy of the active vehicle registry.
        """

        return dict(
            self._vehicles
        )

    # ========================================================
    # QUEUE LENGTH
    # ========================================================

    def _calculate_queue_length(
        self,
        vehicles: List[VehicleState],
    ) -> int:

        return sum(
            1
            for vehicle in vehicles
            if vehicle.stopped
        )

    # ========================================================
    # AVERAGE SPEED
    # ========================================================

    def _calculate_average_speed(
        self,
        vehicles: List[VehicleState],
    ) -> float:

        if not vehicles:
            return 0.0

        return (
            sum(
                vehicle.speed
                for vehicle in vehicles
            )
            / len(vehicles)
        )

    # ========================================================
    # WAITING TIME
    # ========================================================

    def _calculate_waiting_time(
        self,
        vehicles: List[VehicleState],
    ) -> float:

        return sum(
            max(
                0.0,
                float(vehicle.waiting_time)
            )
            for vehicle in vehicles
        )

    # ========================================================
    # PRESSURE
    # ========================================================

    def _calculate_pressure(
        self,
        queue_length: int,
        waiting_time: float,
    ) -> float:
        """
        Prototype traffic pressure.

        Later this will be extended with:

            - lane pressure
            - movement pressure
            - predicted pressure
            - downstream pressure
            - emergency priority
            - starvation priority
        """

        return (
            float(queue_length)
            + float(waiting_time)
        )

    # ========================================================
    # BUILD NETWORK STATE
    # ========================================================

    def build_state(
        self,
        timestamp: float,
    ) -> NetworkTrafficState:
        """
        Build the current network traffic state.
        """

        # ----------------------------------------------------
        # SNAPSHOT
        # ----------------------------------------------------

        vehicles = list(
            self._vehicles.values()
        )

        # ----------------------------------------------------
        # EMPTY NETWORK
        # ----------------------------------------------------

        if not vehicles:

            return NetworkTrafficState(
                timestamp=float(timestamp)
            )

        # ----------------------------------------------------
        # VEHICLE COUNT
        # ----------------------------------------------------

        vehicle_count = len(
            vehicles
        )

        # ----------------------------------------------------
        # QUEUE
        # ----------------------------------------------------

        queue_length = (
            self._calculate_queue_length(
                vehicles
            )
        )

        # ----------------------------------------------------
        # AVERAGE SPEED
        # ----------------------------------------------------

        average_speed = (
            self._calculate_average_speed(
                vehicles
            )
        )

        # ----------------------------------------------------
        # WAITING TIME
        # ----------------------------------------------------

        waiting_time = (
            self._calculate_waiting_time(
                vehicles
            )
        )

        # ----------------------------------------------------
        # PRESSURE
        # ----------------------------------------------------

        pressure = (
            self._calculate_pressure(
                queue_length,
                waiting_time,
            )
        )

        # ----------------------------------------------------
        # DENSITY
        #
        # Prototype representation.
        #
        # Later:
        #   vehicles / road length
        # ----------------------------------------------------

        density = float(
            vehicle_count
        )

        # ----------------------------------------------------
        # OCCUPANCY
        # ----------------------------------------------------

        occupancy = (
            queue_length
            / vehicle_count
            if vehicle_count > 0
            else 0.0
        )

        # ----------------------------------------------------
        # DELAY
        #
        # At this stage waiting time is used as the
        # delay approximation.
        # ----------------------------------------------------

        delay = waiting_time

        # ----------------------------------------------------
        # FLOW RATE
        #
        # Temporal flow calculation will be added later.
        # For now this represents the active vehicle
        # observation count.
        # ----------------------------------------------------

        flow_rate = float(
            vehicle_count
        )

        # ----------------------------------------------------
        # JUNCTION STATES
        #
        # Junction assignment requires runtime topology.
        #
        # Therefore we intentionally do not guess junction
        # IDs from lane names.
        # ----------------------------------------------------

        junctions: Dict[
            str,
            JunctionTrafficState
        ] = {}

        # ----------------------------------------------------
        # FINAL STATE
        # ----------------------------------------------------

        return NetworkTrafficState(

            timestamp=float(
                timestamp
            ),

            vehicle_count=vehicle_count,

            queue_length=queue_length,

            average_speed=average_speed,

            density=density,

            occupancy=occupancy,

            waiting_time=waiting_time,

            delay=delay,

            flow_rate=flow_rate,

            pressure=pressure,

            junctions=junctions,
        )
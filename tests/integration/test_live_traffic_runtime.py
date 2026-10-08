from __future__ import annotations

from pathlib import Path

from traffic_state.engine import (
    TrafficStateEngine,
)

from simulation.live_state_bridge import (
    LiveSUMOVehicleStateAdapter,
)

from simulation.live_traffic_runtime import (
    LiveTrafficRuntime,
)

from simulation.live_topology_runtime import (
    LiveSUMOTopologyRuntime,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

NETWORK_FILE = (
    PROJECT_ROOT
    / "simulation"
    / "networks"
    / "ai_traffic.net.xml"
)


class FakeTraCIManager:

    def __init__(self):

        self._vehicles = {

            "car_1": {
                "vehicle_class": "car",
                "lane_id": "A1_EDGE_0",
                "edge_id": "A1_EDGE",
                "speed": 5.0,
                "waiting_time": 2.0,
            },

            "ambulance_1": {
                "vehicle_class": "ambulance",
                "lane_id": "A1_EDGE_1",
                "edge_id": "A1_EDGE",
                "speed": 0.0,
                "waiting_time": 8.0,
            },
        }

        self.simulation_time = 10.0

    def vehicle_ids(self):

        return list(
            self._vehicles.keys()
        )

    def vehicle_state(
        self,
        vehicle_id,
    ):

        return self._vehicles[
            vehicle_id
        ]


def build_engine():

    engine = TrafficStateEngine()

    topology_runtime = (
        LiveSUMOTopologyRuntime(
            network_file=NETWORK_FILE,
            traffic_state_engine=engine,
        )
    )

    topology_runtime.apply_to_engine()

    return engine


def test_vehicle_adapter_converts_live_vehicle():

    adapter = (
        LiveSUMOVehicleStateAdapter()
    )

    state = adapter.convert(
        "car_1",
        {
            "vehicle_class": "car",
            "lane_id": "A1_EDGE_0",
            "edge_id": "A1_EDGE",
            "speed": 5.0,
            "waiting_time": 2.0,
        },
    )

    assert state.vehicle_id == "car_1"
    assert state.vehicle_class == "car"
    assert state.lane_id == "A1_EDGE_0"
    assert state.edge_id == "A1_EDGE"
    assert state.speed == 5.0
    assert state.waiting_time == 2.0
    assert state.stopped is False


def test_stopped_vehicle_is_detected():

    adapter = (
        LiveSUMOVehicleStateAdapter()
    )

    state = adapter.convert(
        "ambulance_1",
        {
            "vehicle_class": "ambulance",
            "lane_id": "A1_EDGE_1",
            "edge_id": "A1_EDGE",
            "speed": 0.0,
            "waiting_time": 8.0,
        },
    )

    assert state.stopped is True


def test_snapshot_conversion():

    adapter = (
        LiveSUMOVehicleStateAdapter()
    )

    snapshot = {

        "car_1": {
            "vehicle_class": "car",
            "lane_id": "A1_EDGE_0",
            "edge_id": "A1_EDGE",
            "speed": 5.0,
            "waiting_time": 2.0,
        },

        "bus_1": {
            "vehicle_class": "bus",
            "lane_id": "A2_EDGE_0",
            "edge_id": "A2_EDGE",
            "speed": 0.0,
            "waiting_time": 4.0,
        },
    }

    vehicles = (
        adapter.convert_snapshot(
            snapshot
        )
    )

    assert len(vehicles) == 2

    assert {
        vehicle.vehicle_id
        for vehicle in vehicles
    } == {
        "car_1",
        "bus_1",
    }


def test_live_runtime_updates_engine():

    manager = FakeTraCIManager()

    engine = build_engine()

    runtime = LiveTrafficRuntime(
        traci_manager=manager,
        traffic_state_engine=engine,
    )

    state = runtime.update_state()

    assert len(engine.vehicles) == 2

    assert state.vehicle_count == 2

    assert state.timestamp == 10.0


def test_live_runtime_removes_departed_vehicle():

    manager = FakeTraCIManager()

    engine = build_engine()

    runtime = LiveTrafficRuntime(
        traci_manager=manager,
        traffic_state_engine=engine,
    )

    runtime.update_state()

    assert len(engine.vehicles) == 2

    del manager._vehicles[
        "car_1"
    ]

    manager.simulation_time = 11.0

    state = runtime.update_state()

    assert len(engine.vehicles) == 1

    assert "car_1" not in engine.vehicles

    assert "ambulance_1" in engine.vehicles

    assert state.vehicle_count == 1

    assert state.timestamp == 11.0
from pathlib import Path

import pytest

from src.simulation.traci_manager import (
    TraCIConnectionError,
    TraCIManager,
)


class FakeSimulation:
    def __init__(self):
        self.time = 0.0

    def getTime(self):
        return self.time

    def simulationStep(self, target_time):
        self.time = float(target_time)


class FakeTrafficLight:
    def __init__(self):
        self.states = {
            "tls_1": "GrGr",
            "tls_2": "rGrG",
        }

    def getIDList(self):
        return tuple(self.states.keys())

    def getControlledLinks(self, tls_id):
        return (
            (
                (
                    "incoming_0",
                    "outgoing_0",
                    ""
                ),
            ),
        )

    def getRedYellowGreenState(self, tls_id):
        return self.states[tls_id]

    def setRedYellowGreenState(
        self,
        tls_id,
        state,
    ):
        self.states[tls_id] = state


class FakeVehicle:
    def getIDList(self):
        return ("veh_1",)

    def getSpeed(self, vehicle_id):
        return 8.5

    def getLaneID(self, vehicle_id):
        return "lane_1"

    def getRoadID(self, vehicle_id):
        return "edge_1"

    def getPosition(self, vehicle_id):
        return (10.0, 20.0)


class FakeTraCI:
    def __init__(self):
        self.simulation = FakeSimulation()
        self.trafficlight = FakeTrafficLight()
        self.vehicle = FakeVehicle()
        self.started = False
        self.closed = False

    def start(self, command, label=None):
        self.started = True

    def switch(self, label):
        self.started = True
    def simulationStep(self, target_time=None):
        if target_time is None:
            target_time = self.simulation.time + 1.0

        self.simulation.simulationStep(target_time)

    def close(self):
        self.closed = True


@pytest.fixture
def fake_traci():
    return FakeTraCI()


@pytest.fixture
def config_file(tmp_path):
    config = tmp_path / "test.sumocfg"
    config.write_text(
        "<configuration></configuration>",
        encoding="utf-8",
    )
    return config


def test_config_validation(config_file):
    manager = TraCIManager(
        config_file,
        traci_module=FakeTraCI(),
    )

    assert manager.validate_config() == config_file.resolve()


def test_missing_config_raises(tmp_path):
    manager = TraCIManager(
        tmp_path / "missing.sumocfg",
        traci_module=FakeTraCI(),
    )

    with pytest.raises(FileNotFoundError):
        manager.validate_config()


def test_invalid_config_extension(tmp_path):
    config = tmp_path / "config.xml"
    config.write_text(
        "<configuration/>",
        encoding="utf-8",
    )

    manager = TraCIManager(
        config,
        traci_module=FakeTraCI(),
    )

    with pytest.raises(ValueError):
        manager.validate_config()


def test_requires_connection(config_file):
    manager = TraCIManager(
        config_file,
        traci_module=FakeTraCI(),
    )

    with pytest.raises(TraCIConnectionError):
        manager.simulation_time()


def test_start(config_file, fake_traci, monkeypatch):
    manager = TraCIManager(
        config_file,
        traci_module=fake_traci,
    )

    monkeypatch.setattr(
        manager,
        "find_sumo_binary",
        lambda: "sumo.exe",
    )

    manager.start()

    assert manager.connected
    assert fake_traci.started


def test_simulation_step(
    config_file,
    fake_traci,
    monkeypatch,
):
    manager = TraCIManager(
        config_file,
        traci_module=fake_traci,
        step_length=1.0,
    )

    monkeypatch.setattr(
        manager,
        "find_sumo_binary",
        lambda: "sumo.exe",
    )

    manager.start()

    assert manager.simulation_time() == 0.0

    now = manager.step()

    assert now == 1.0


def test_dynamic_traffic_light_discovery(
    config_file,
    fake_traci,
    monkeypatch,
):
    manager = TraCIManager(
        config_file,
        traci_module=fake_traci,
    )

    monkeypatch.setattr(
        manager,
        "find_sumo_binary",
        lambda: "sumo.exe",
    )

    manager.start()

    ids = manager.traffic_light_ids()

    assert ids == (
        "tls_1",
        "tls_2",
    )


def test_controlled_links(
    config_file,
    fake_traci,
    monkeypatch,
):
    manager = TraCIManager(
        config_file,
        traci_module=fake_traci,
    )

    monkeypatch.setattr(
        manager,
        "find_sumo_binary",
        lambda: "sumo.exe",
    )

    manager.start()

    links = manager.controlled_links(
        "tls_1"
    )

    assert links


def test_signal_state_and_update(
    config_file,
    fake_traci,
    monkeypatch,
):
    manager = TraCIManager(
        config_file,
        traci_module=fake_traci,
    )

    monkeypatch.setattr(
        manager,
        "find_sumo_binary",
        lambda: "sumo.exe",
    )

    manager.start()

    assert manager.signal_state(
        "tls_1"
    ) == "GrGr"

    manager.set_signal_state(
        "tls_1",
        "rrrr",
    )

    assert manager.signal_state(
        "tls_1"
    ) == "rrrr"


def test_vehicle_discovery(
    config_file,
    fake_traci,
    monkeypatch,
):
    manager = TraCIManager(
        config_file,
        traci_module=fake_traci,
    )

    monkeypatch.setattr(
        manager,
        "find_sumo_binary",
        lambda: "sumo.exe",
    )

    manager.start()

    assert manager.vehicle_ids() == (
        "veh_1",
    )

    state = manager.vehicle_state(
        "veh_1"
    )

    assert state["vehicle_id"] == "veh_1"
    assert state["speed"] == 8.5
    assert state["lane_id"] == "lane_1"


def test_close_is_idempotent(
    config_file,
    fake_traci,
    monkeypatch,
):
    manager = TraCIManager(
        config_file,
        traci_module=fake_traci,
    )

    monkeypatch.setattr(
        manager,
        "find_sumo_binary",
        lambda: "sumo.exe",
    )

    manager.start()

    manager.close()
    manager.close()

    assert not manager.connected
    assert fake_traci.closed
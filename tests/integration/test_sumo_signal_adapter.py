from src.signal_control.signal_executor import SignalState
from src.signal_control.sumo_signal_adapter import (
    SUMOSignalAdapter,
)


class FakeTrafficLight:

    def __init__(self):
        self.states = {
            "tls_alpha": "RRGG",
            "tls_beta": "GGRR",
        }

        self.calls = []

    def getIDList(self):
        return list(self.states.keys())

    def getRedYellowGreenState(self, tls_id):
        return self.states[tls_id]

    def setRedYellowGreenState(
        self,
        tls_id,
        state,
    ):
        self.states[tls_id] = state
        self.calls.append(
            (tls_id, state)
        )


class FakeTraCI:

    def __init__(self):
        self.trafficlight = FakeTrafficLight()


def test_discover_traffic_lights():

    traci = FakeTraCI()

    adapter = SUMOSignalAdapter(traci)

    lights = adapter.discover_traffic_lights()

    assert len(lights) == 2


def test_discover_signal_positions():

    traci = FakeTraCI()

    adapter = SUMOSignalAdapter(traci)

    lights = adapter.discover_traffic_lights()

    alpha = lights[0]

    assert alpha.tls_id == "tls_alpha"

    assert len(alpha.links) == 4


def test_build_all_red():

    traci = FakeTraCI()

    adapter = SUMOSignalAdapter(traci)

    adapter.discover_traffic_lights()

    state = adapter.build_all_red_state(
        "tls_alpha"
    )

    assert state == "RRRR"


def test_build_green():

    traci = FakeTraCI()

    adapter = SUMOSignalAdapter(traci)

    adapter.discover_traffic_lights()

    state = adapter.build_green_state(
        "tls_alpha",
        [2],
    )

    assert state == "RRG R".replace(" ", "")


def test_build_yellow():

    traci = FakeTraCI()

    adapter = SUMOSignalAdapter(traci)

    adapter.discover_traffic_lights()

    state = adapter.build_yellow_state(
        "tls_alpha",
        [2],
    )

    assert state == "RRYR"


def test_apply_all_red():

    traci = FakeTraCI()

    adapter = SUMOSignalAdapter(traci)

    adapter.discover_traffic_lights()

    state = adapter.apply_state(
        "tls_alpha",
        SignalState.ALL_RED,
        [1],
    )

    assert state == "RRRR"

    assert traci.trafficlight.calls[-1] == (
        "tls_alpha",
        "RRRR",
    )


def test_apply_green():

    traci = FakeTraCI()

    adapter = SUMOSignalAdapter(traci)

    adapter.discover_traffic_lights()

    state = adapter.apply_state(
        "tls_alpha",
        SignalState.GREEN,
        [1],
    )

    assert state == "RGRR"

    assert traci.trafficlight.calls[-1] == (
        "tls_alpha",
        "RGRR",
    )


def test_unknown_traffic_light():

    traci = FakeTraCI()

    adapter = SUMOSignalAdapter(traci)

    adapter.discover_traffic_lights()

    try:
        adapter.build_all_red_state(
            "unknown"
        )
        assert False
    except KeyError:
        assert True
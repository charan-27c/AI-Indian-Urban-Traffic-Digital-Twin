from pathlib import Path

from src.perception.junction.sumo_topology import (
    SumoTopology,
)

from src.traffic_state import (
    TrafficStateEngine,
    VehicleState,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

NETWORK_PATH = (
    PROJECT_ROOT
    / "simulation"
    / "networks"
    / "ai_traffic.net.xml"
)


def test_traffic_state_uses_dynamic_topology():

    loader = SumoTopology(
        str(NETWORK_PATH)
    )

    junctions = (
        loader.discover_all_junctions()
    )

    assert len(junctions) > 0

    engine = TrafficStateEngine()

    engine.set_topology(
        loader
    )

    # --------------------------------------------------------
    # Add a vehicle to a real discovered lane.
    # --------------------------------------------------------

    first_junction = next(
        iter(junctions.values())
    )

    first_approach = next(
        iter(
            first_junction.get_approaches()
        )
    )

    first_lane = (
        first_approach.lanes[0]
    )

    engine.update_vehicle(
        VehicleState(
            vehicle_id="test_vehicle_1",
            vehicle_class="car",
            lane_id=first_lane.lane_id,
            edge_id=first_lane.edge_id,
            speed=0.0,
            waiting_time=5.0,
            stopped=True,
        )
    )

    state = engine.build_state(
        timestamp=1.0
    )

    # --------------------------------------------------------
    # Vehicle must appear in network state.
    # --------------------------------------------------------

    assert (
        state.vehicle_count
        == 1
    )

    assert (
        state.queue_length
        == 1
    )

    # --------------------------------------------------------
    # Discovered junction must exist.
    # --------------------------------------------------------

    assert (
        first_junction.junction_id
        in state.junctions
    )

    junction_state = (
        state.junctions[
            first_junction.junction_id
        ]
    )

    assert (
        junction_state.vehicle_count
        == 1
    )

    assert (
        first_lane.lane_id
        in junction_state.lanes
    )
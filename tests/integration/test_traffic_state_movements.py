from pathlib import Path

from src.perception.junction.sumo_topology import (
    SumoTopology,
)

from src.traffic_state.engine import (
    TrafficStateEngine,
)

from src.traffic_state.models import (
    VehicleState,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

NETWORK_PATH = (
    PROJECT_ROOT
    / "simulation"
    / "networks"
    / "ai_traffic.net.xml"
)


def test_traffic_state_uses_dynamic_movements():

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

    first_junction = next(
        iter(junctions.values())
    )

    first_movement = next(
        iter(
            first_junction.get_movements()
        )
    )

    engine.update_vehicle(
        VehicleState(
            vehicle_id="movement_vehicle_1",
            vehicle_class="car",

            lane_id=(
                first_movement
                .from_lane_id
            ),

            edge_id=(
                first_movement
                .from_edge_id
            ),

            speed=0.0,
            waiting_time=5.0,
            stopped=True,
        )
    )

    state = engine.build_state(
        timestamp=1.0
    )

    assert state.vehicle_count == 1

    junction_state = (
        state.junctions[
            first_movement.junction_id
        ]
    )

    assert (
        first_movement.movement_id
        in junction_state.movements
    )

    movement_state = (
        junction_state.movements[
            first_movement.movement_id
        ]
    )

    assert movement_state.vehicle_count == 1

    assert movement_state.queue_length == 1

    assert (
        first_movement.from_lane_id
        in movement_state.lanes
    )
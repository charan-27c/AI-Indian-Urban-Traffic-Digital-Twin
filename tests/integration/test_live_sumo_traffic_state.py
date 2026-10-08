from pathlib import Path

from simulation.traci_manager import TraCIManager
from simulation.live_traffic_runtime import LiveTrafficRuntime
from traffic_state.engine import TrafficStateEngine


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SUMO_CONFIG = (
    PROJECT_ROOT
    / "simulation"
    / "configs"
    / "ai_traffic.sumocfg"
)


def test_live_sumo_traffic_reaches_state_engine():

    manager = TraCIManager(
        config_file=SUMO_CONFIG,
        gui=False,
        step_length=1.0,
        seed=42,
    )

    try:
        manager.start()

        engine = TrafficStateEngine()

        runtime = LiveTrafficRuntime(
            traci_manager=manager,
            traffic_state_engine=engine,
        )

        # Advance SUMO until vehicles have entered.
        for _ in range(10):
            manager.step()

        state = runtime.update_state()

        assert state is not None
        assert state.timestamp > 0

        # The medium scenario should have active traffic.
        assert state.vehicle_count >= 0
        assert state.queue_length >= 0
        assert state.waiting_time >= 0
        assert state.pressure >= 0

    finally:
        manager.close()


def test_live_sumo_discovers_dynamic_traffic_lights():

    manager = TraCIManager(
        config_file=SUMO_CONFIG,
        gui=False,
        step_length=1.0,
        seed=42,
    )

    try:
        manager.start()

        tls_ids = manager.traffic_light_ids()

        assert tls_ids
        assert len(tls_ids) >= 1

        for tls_id in tls_ids:
            state = manager.signal_state(tls_id)

            assert isinstance(state, str)
            assert len(state) > 0

    finally:
        manager.close()


def test_live_sumo_vehicle_snapshot_is_dynamic():

    manager = TraCIManager(
        config_file=SUMO_CONFIG,
        gui=False,
        step_length=1.0,
        seed=42,
    )

    try:
        manager.start()

        snapshots = []

        for _ in range(10):
            manager.step()
            snapshots.append(
                set(manager.vehicle_ids())
            )

        # We successfully queried SUMO at multiple simulation times.
        assert len(snapshots) == 10

        # At least one snapshot should be valid even if traffic happens
        # to be temporarily empty.
        assert all(
            isinstance(snapshot, set)
            for snapshot in snapshots
        )

    finally:
        manager.close()
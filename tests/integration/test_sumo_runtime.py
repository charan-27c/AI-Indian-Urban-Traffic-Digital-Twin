from pathlib import Path

from simulation.traci_manager import TraCIManager


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SUMO_CONFIG = (
    PROJECT_ROOT
    / "simulation"
    / "configs"
    / "ai_traffic.sumocfg"
)


def test_sumo_configuration_exists():
    assert SUMO_CONFIG.exists()
    assert SUMO_CONFIG.suffix == ".sumocfg"


def test_sumo_runtime_starts_and_discovers_network():
    manager = TraCIManager(
        config_file=SUMO_CONFIG,
        gui=False,
        step_length=1.0,
        seed=42,
    )

    try:
        manager.start()

        assert manager.connected

        tls_ids = manager.traffic_light_ids()

        assert tls_ids
        assert len(tls_ids) == 3

        assert set(tls_ids) == {
            "J1",
            "J2",
            "J3",
        }

    finally:
        manager.close()


def test_sumo_runtime_steps():
    manager = TraCIManager(
        config_file=SUMO_CONFIG,
        gui=False,
        step_length=1.0,
        seed=42,
    )

    try:
        manager.start()

        initial_time = manager.simulation_time()

        manager.step()

        assert manager.simulation_time() > initial_time

    finally:
        manager.close()
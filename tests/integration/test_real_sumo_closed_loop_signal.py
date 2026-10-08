from pathlib import Path

import pytest

from control.closed_loop_signal_runtime import ClosedLoopSignalRuntime
from signal_control.controller_pipeline import SignalControlPipeline
from signal_control.incident_aware_optimizer import IncidentAwareSignalOptimizer
from signal_control.incident_prediction_fusion import IncidentPredictionFusion
from signal_control.movement_candidate_builder import DynamicMovementCandidateBuilder
from signal_control.signal_executor import SafeSignalExecutor
from signal_control.sumo_movement_mapper import SUMOMovementMapper
from signal_control.sumo_signal_adapter import SUMOSignalAdapter
from simulation.live_topology_runtime import LiveSUMOTopologyRuntime
from simulation.live_traffic_runtime import LiveTrafficRuntime
from simulation.traci_manager import TraCIManager
from traffic_state.engine import TrafficStateEngine


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SUMO_CONFIG = (
    PROJECT_ROOT
    / "simulation"
    / "configs"
    / "ai_traffic.sumocfg"
)

NETWORK_FILE = (
    PROJECT_ROOT
    / "simulation"
    / "networks"
    / "ai_traffic.net.xml"
)


@pytest.mark.integration
def test_real_sumo_closed_loop_signal_runtime():
    """
    End-to-end integration test.

    Real SUMO
        ->
    TraCI
        ->
    live vehicle state
        ->
    traffic state engine
        ->
    dynamic movement discovery
        ->
    candidate generation
        ->
    incident/prediction fusion
        ->
    signal control pipeline
        ->
    real SUMO traffic lights.
    """

    assert SUMO_CONFIG.exists()
    assert NETWORK_FILE.exists()

    engine = TrafficStateEngine()

    traci_manager = TraCIManager(
        config_file=SUMO_CONFIG,
        step_length=1.0,
        seed=42,
    )

    topology_runtime = LiveSUMOTopologyRuntime(
        network_file=NETWORK_FILE,
        traffic_state_engine=engine,
    )

    live_traffic_runtime = LiveTrafficRuntime(
        traci_manager=traci_manager,
        traffic_state_engine=engine,
    )

    try:
        traci_manager.start()

        # ------------------------------------------------------
        # 1. Discover topology from the REAL SUMO network.
        # ------------------------------------------------------

        topology = topology_runtime.apply_to_engine()

        assert topology is not None
        assert len(topology) > 0

        import traci

        movement_mapper = SUMOMovementMapper(
            topology=topology,
        )
        movement_mapper.discover_controlled_links(
            traci
        )
        movement_mapper.build_mappings()

        signal_adapter = SUMOSignalAdapter(
            traci
        )
        signal_adapter.discover_traffic_lights()

        signal_pipeline = SignalControlPipeline(
            optimizer=IncidentAwareSignalOptimizer(),
            executor=SafeSignalExecutor(),
            movement_mapper=movement_mapper,
            sumo_adapter=signal_adapter,
        )

        runtime = ClosedLoopSignalRuntime(
            traci_manager=traci_manager,
            live_traffic_runtime=live_traffic_runtime,
            candidate_builder=DynamicMovementCandidateBuilder(),
            fusion=IncidentPredictionFusion(),
            signal_pipeline=signal_pipeline,
        )

        # ------------------------------------------------------
        # 2. Discover real SUMO traffic lights dynamically.
        # ------------------------------------------------------

        traffic_lights = traci_manager.traffic_light_ids()

        assert traffic_lights
        assert len(traffic_lights) >= 1

        # ------------------------------------------------------
        # 3. Run several real SUMO closed-loop cycles.
        # ------------------------------------------------------

        results = []

        for _ in range(15):
            result = runtime.control_cycle()

            assert result is not None

            results.append(result)

        # ------------------------------------------------------
        # 4. Verify traffic state was actually produced.
        # ------------------------------------------------------

        final_state = engine.get_state()

        assert final_state is not None
        assert final_state.timestamp >= 0.0

        # ------------------------------------------------------
        # 5. Verify dynamic junction discovery.
        # ------------------------------------------------------

        assert len(final_state.junctions) >= 1

        # ------------------------------------------------------
        # 6. Verify that SUMO still has valid signal states.
        # ------------------------------------------------------

        for tls_id in traffic_lights:
            state = traci_manager.signal_state(tls_id)

            assert isinstance(state, str)
            assert len(state) > 0

            # SUMO signal states must contain valid signal
            # characters only.
            assert set(state).issubset(
                set("rRgGyYoOu")
            )

        # ------------------------------------------------------
        # 7. Verify the runtime remained alive.
        # ------------------------------------------------------

        assert traci_manager.connected

    finally:
        traci_manager.close()
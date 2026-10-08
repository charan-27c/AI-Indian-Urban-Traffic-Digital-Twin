from __future__ import annotations

from typing import Any

from .live_state_bridge import (
    LiveSUMOVehicleStateAdapter,
)


class LiveTrafficRuntime:
    """
    Connects live TraCI vehicle telemetry to
    TrafficStateEngine.

    Pipeline:

        TraCI
          ↓
        vehicle snapshot
          ↓
        VehicleState
          ↓
        TrafficStateEngine
          ↓
        NetworkTrafficState
    """

    def __init__(
        self,
        traci_manager,
        traffic_state_engine,
        vehicle_adapter=None,
    ) -> None:

        self.traci_manager = (
            traci_manager
        )

        self.traffic_state_engine = (
            traffic_state_engine
        )

        self.vehicle_adapter = (
            vehicle_adapter
            or LiveSUMOVehicleStateAdapter()
        )

    # ========================================================
    # COLLECT LIVE VEHICLES
    # ========================================================

    def collect_vehicle_snapshot(
        self,
    ) -> dict[str, dict[str, Any]]:

        snapshot = {}

        for vehicle_id in (
            self.traci_manager.vehicle_ids()
        ):
            try:
                state = (
                    self.traci_manager
                    .vehicle_state(
                        vehicle_id
                    )
                )

                snapshot[
                    vehicle_id
                ] = state

            except Exception:
                # A vehicle may disappear between
                # getIDList() and get vehicle state.
                continue

        return snapshot

    # ========================================================
    # UPDATE ENGINE
    # ========================================================

    def update_state(self):

        snapshot = (
            self.collect_vehicle_snapshot()
        )

        vehicles = (
            self.vehicle_adapter
            .convert_snapshot(snapshot)
        )

        active_ids = {
            vehicle.vehicle_id
            for vehicle in vehicles
        }

        existing =dict (
            self.traffic_state_engine.vehicles
        )

        # Remove vehicles that have left SUMO.
        for vehicle_id in existing:

            if vehicle_id not in active_ids:

                self.traffic_state_engine.remove_vehicle(
                    vehicle_id
                )

        # Update current vehicles.
        for vehicle in vehicles:

            self.traffic_state_engine.update_vehicle(
                vehicle
            )

        # TraCIManager exposes simulation time.
        simulation_time = self.traci_manager.simulation_time

        if callable(simulation_time):
           simulation_time = simulation_time()

        timestamp = float(simulation_time)

        return (
            self.traffic_state_engine
            .build_state(timestamp)
        )
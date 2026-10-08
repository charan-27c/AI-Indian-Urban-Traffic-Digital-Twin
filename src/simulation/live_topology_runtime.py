from __future__ import annotations

from pathlib import Path


class RuntimeJunctionTopology(dict):
    """
    Adapter around dynamically discovered SUMO junctions.

    Keeps dictionary behavior so existing callers can use:

        topology["J1"]
        topology.items()
        len(topology)

    while also exposing the interface expected by
    TrafficStateEngine:

        topology.junctions
        topology.get_junctions()
    """

    @property
    def junctions(self):
        return self

    def get_junctions(self):
        return self


class LiveSUMOTopologyRuntime:
    """
    Runtime bridge between SUMO topology discovery and
    TrafficStateEngine.

    SUMO remains the authoritative topology source.

    No fixed:
        - junction IDs
        - approach IDs
        - lane IDs
        - movement IDs
    """

    def __init__(
        self,
        network_file,
        traffic_state_engine,
        topology_provider=None,
    ):
        self.network_file = (
            Path(network_file)
            .resolve()
        )

        if not self.network_file.exists():
            raise FileNotFoundError(
                f"SUMO network not found: "
                f"{self.network_file}"
            )

        self.traffic_state_engine = (
            traffic_state_engine
        )

        if topology_provider is None:
            from perception.junction.sumo_topology import (
                SumoTopology,
            )

            topology_provider = SumoTopology(
                str(self.network_file)
            )

        self.topology_provider = (
            topology_provider
        )

        self.snapshot = None

    # ========================================================
    # DISCOVER
    # ========================================================

    def discover(self):
        """
        Discover all junctions dynamically from SUMO.

        Returns a RuntimeJunctionTopology so it remains
        dictionary-compatible while also satisfying the
        TrafficStateEngine topology contract.
        """

        junctions = (
            self.topology_provider
            .discover_all_junctions()
        )

        if not junctions:
            raise RuntimeError(
                "No junctions were discovered "
                f"from SUMO network: {self.network_file}"
            )

        self.snapshot = RuntimeJunctionTopology(
            junctions
        )

        return self.snapshot

    # ========================================================
    # APPLY TO TRAFFIC STATE ENGINE
    # ========================================================

    def apply_to_engine(self):
        """
        Discover the current SUMO topology and register it
        with TrafficStateEngine.
        """

        topology = self.discover()

        self.traffic_state_engine.set_topology(
            topology
        )

        return topology

    # ========================================================
    # REFRESH
    # ========================================================

    def refresh(self):
        """
        Rediscover topology and replace the engine topology.

        This is intentionally dynamic and does not assume
        J1/J2/J3/J4.
        """

        return self.apply_to_engine()

    # ========================================================
    # CURRENT SNAPSHOT
    # ========================================================

    def current_snapshot(self):
        return self.snapshot
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Optional

from src.perception.junction.topology import JunctionTopology
from src.perception.junction.sumo_topology import SumoTopology


@dataclass(frozen=True)
class TopologySnapshot:
    """
    Immutable runtime snapshot of the discovered road topology.

    The snapshot represents one consistent topology state.

    Nothing here assumes:
        - fixed junction count
        - fixed approaches
        - fixed lanes
        - fixed movements
        - fixed directions
    """

    version: str

    created_at: str

    junctions: Dict[str, JunctionTopology]

    junction_count: int

    approach_count: int

    lane_count: int

    connection_count: int

    movement_count: int


class RuntimeTopologyManager:
    """
    Manages the current runtime topology.

    Responsibilities:

        1. Discover topology from the configured provider.
        2. Create a topology snapshot.
        3. Detect topology changes.
        4. Maintain the current topology version.
        5. Provide a stable topology to downstream systems.

    The manager does NOT hard-code junctions, approaches,
    lanes, movements, or directions.
    """

    def __init__(
        self,
        topology_provider: SumoTopology,
    ):
        self.topology_provider = topology_provider

        self._current_snapshot: Optional[
            TopologySnapshot
        ] = None

    # ========================================================
    # DISCOVERY
    # ========================================================

    def discover(self) -> TopologySnapshot:
        """
        Discover the complete runtime topology.

        If the topology has not changed, the existing version
        is preserved.

        If the topology has changed, a new version is created.
        """

        junctions = (
            self.topology_provider
            .discover_all_junctions()
        )

        topology_signature = (
            self._build_signature(junctions)
        )

        new_version = (
            self._generate_version(
                topology_signature
            )
        )

        if (
            self._current_snapshot is not None
            and
            self._current_snapshot.version
            == new_version
        ):
            return self._current_snapshot

        snapshot = self._create_snapshot(
            junctions,
            new_version,
        )

        self._current_snapshot = snapshot

        return snapshot

    # ========================================================
    # CURRENT SNAPSHOT
    # ========================================================

    def get_current_snapshot(
        self,
    ) -> TopologySnapshot:

        if self._current_snapshot is None:

            raise RuntimeError(
                "Runtime topology has not been "
                "discovered yet. Call discover() first."
            )

        return self._current_snapshot

    # ========================================================
    # CHANGE DETECTION
    # ========================================================

    def has_topology_changed(self) -> bool:
        """
        Re-discover topology and determine whether it changed.
        """

        previous = self._current_snapshot

        current = self.discover()

        if previous is None:
            return True

        return (
            previous.version
            != current.version
        )

    # ========================================================
    # SNAPSHOT CREATION
    # ========================================================

    def _create_snapshot(
        self,
        junctions: Dict[str, JunctionTopology],
        version: str,
    ) -> TopologySnapshot:

        approach_count = sum(
            topology.approach_count()
            for topology in junctions.values()
        )

        lane_count = sum(
            topology.lane_count()
            for topology in junctions.values()
        )

        connection_count = sum(
            topology.connection_count()
            for topology in junctions.values()
        )

        movement_count = sum(
            topology.movement_count()
            for topology in junctions.values()
        )

        return TopologySnapshot(
            version=version,

            created_at=(
                datetime.now(
                    timezone.utc
                ).isoformat()
            ),

            junctions=dict(junctions),

            junction_count=len(junctions),

            approach_count=approach_count,

            lane_count=lane_count,

            connection_count=connection_count,

            movement_count=movement_count,
        )

    # ========================================================
    # SIGNATURE
    # ========================================================

    def _build_signature(
        self,
        junctions: Dict[str, JunctionTopology],
    ) -> str:
        """
        Build a deterministic representation of topology.

        The signature changes when:

            - junction changes
            - approach changes
            - lane changes
            - connection changes
            - movement changes

        Ordering is deterministic so identical topology
        produces the same signature.
        """

        parts = []

        for junction_id in sorted(
            junctions.keys()
        ):

            topology = junctions[
                junction_id
            ]

            parts.append(
                f"JUNCTION:{junction_id}"
            )

            for approach in sorted(
                topology.get_approaches(),
                key=lambda item:
                item.approach_id,
            ):

                parts.append(
                    f"APPROACH:"
                    f"{approach.approach_id}"
                )

                for lane in sorted(
                    approach.get_lanes(),
                    key=lambda item:
                    item.lane_id,
                ):

                    parts.append(
                        f"LANE:"
                        f"{lane.lane_id}:"
                        f"{lane.edge_id}"
                    )

            for connection in sorted(
                topology.get_connections(),
                key=lambda item:
                item.connection_id,
            ):

                parts.append(
                    f"CONNECTION:"
                    f"{connection.connection_id}:"
                    f"{connection.from_lane_id}:"
                    f"{connection.to_lane_id}:"
                    f"{connection.from_edge_id}:"
                    f"{connection.to_edge_id}"
                )

            for movement in sorted(
                topology.get_movements(),
                key=lambda item:
                item.movement_id,
            ):

                parts.append(
                    f"MOVEMENT:"
                    f"{movement.movement_id}:"
                    f"{movement.connection_id}:"
                    f"{movement.from_lane_id}:"
                    f"{movement.to_lane_id}:"
                    f"{movement.approach_id}"
                )

        return "|".join(parts)

    # ========================================================
    # VERSION
    # ========================================================

    @staticmethod
    def _generate_version(
        signature: str,
    ) -> str:
        """
        Generate deterministic topology version.
        """

        import hashlib

        digest = hashlib.sha256(
            signature.encode("utf-8")
        ).hexdigest()

        return f"topology-{digest[:16]}"
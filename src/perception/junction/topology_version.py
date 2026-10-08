from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Dict

from src.perception.junction.topology import JunctionTopology


@dataclass(frozen=True)
class TopologyVersion:
    """
    Immutable identity of a discovered runtime topology.
    """

    fingerprint: str
    junction_count: int
    approach_count: int
    lane_count: int
    connection_count: int

    @property
    def short_id(self) -> str:
        """
        Short human-readable topology identifier.
        """
        return self.fingerprint[:12]

    def as_dict(self) -> dict:
        return {
            "fingerprint": self.fingerprint,
            "short_id": self.short_id,
            "junction_count": self.junction_count,
            "approach_count": self.approach_count,
            "lane_count": self.lane_count,
            "connection_count": self.connection_count,
        }


class TopologyVersionManager:
    """
    Creates deterministic versions of runtime topology.

    Important:

    This class does NOT assume:

        - fixed junction IDs
        - fixed approaches
        - fixed lanes
        - fixed movements
        - fixed directions

    The fingerprint is calculated from discovered topology.
    """

    def create_version(
        self,
        topologies: Dict[
            str,
            JunctionTopology,
        ],
    ) -> TopologyVersion:

        canonical = self._canonicalize(
            topologies
        )

        payload = json.dumps(
            canonical,
            sort_keys=True,
            separators=(",", ":"),
        )

        fingerprint = hashlib.sha256(
            payload.encode("utf-8")
        ).hexdigest()

        junction_count = len(topologies)

        approach_count = sum(
            topology.approach_count()
            for topology in topologies.values()
        )

        lane_count = sum(
            topology.lane_count()
            for topology in topologies.values()
        )

        connection_count = sum(
            topology.connection_count()
            for topology in topologies.values()
        )

        return TopologyVersion(
            fingerprint=fingerprint,
            junction_count=junction_count,
            approach_count=approach_count,
            lane_count=lane_count,
            connection_count=connection_count,
        )

    # ========================================================
    # TOPOLOGY CANONICALIZATION
    # ========================================================

    def _canonicalize(
        self,
        topologies: Dict[
            str,
            JunctionTopology,
        ],
    ) -> dict:
        """
        Convert topology into deterministic data.

        Sorting ensures that dictionary/list ordering does
        not accidentally produce a different version.
        """

        result = []

        for junction_id in sorted(topologies):

            topology = topologies[
                junction_id
            ]

            junction_data = {
                "junction_id": junction_id,
                "approaches": [],
                "connections": [],
            }

            # ------------------------------------------------
            # APPROACHES
            # ------------------------------------------------

            for approach in sorted(
                topology.get_approaches(),
                key=lambda item: item.approach_id,
            ):

                approach_data = {
                    "approach_id":
                        approach.approach_id,

                    "name":
                        approach.name,

                    "lanes": [],
                }

                for lane in sorted(
                    approach.get_lanes(),
                    key=lambda item: item.lane_id,
                ):

                    approach_data[
                        "lanes"
                    ].append(
                        {
                            "lane_id":
                                lane.lane_id,

                            "edge_id":
                                lane.edge_id,
                        }
                    )

                junction_data[
                    "approaches"
                ].append(
                    approach_data
                )

            # ------------------------------------------------
            # CONNECTIONS
            # ------------------------------------------------

            for connection in sorted(
                topology.get_connections(),
                key=lambda item: item.connection_id,
            ):

                junction_data[
                    "connections"
                ].append(
                    {
                        "connection_id":
                            connection.connection_id,

                        "from_lane_id":
                            connection.from_lane_id,

                        "to_lane_id":
                            connection.to_lane_id,

                        "from_edge_id":
                            connection.from_edge_id,

                        "to_edge_id":
                            connection.to_edge_id,
                    }
                )

            result.append(
                junction_data
            )

        return {
            "junctions": result
        }

    # ========================================================
    # COMPARISON
    # ========================================================

    def has_changed(
        self,
        previous: TopologyVersion,
        current: TopologyVersion,
    ) -> bool:
        """
        Return True if the topology version changed.
        """

        return (
            previous.fingerprint
            != current.fingerprint
        )

    def compare(
        self,
        previous: TopologyVersion,
        current: TopologyVersion,
    ) -> dict:
        """
        Return a compact topology change report.
        """

        changed = self.has_changed(
            previous,
            current,
        )

        return {
            "changed": changed,

            "previous_version":
                previous.short_id,

            "current_version":
                current.short_id,

            "previous_fingerprint":
                previous.fingerprint,

            "current_fingerprint":
                current.fingerprint,

            "junction_count_changed":
                (
                    previous.junction_count
                    != current.junction_count
                ),

            "approach_count_changed":
                (
                    previous.approach_count
                    != current.approach_count
                ),

            "lane_count_changed":
                (
                    previous.lane_count
                    != current.lane_count
                ),

            "connection_count_changed":
                (
                    previous.connection_count
                    != current.connection_count
                ),
        }
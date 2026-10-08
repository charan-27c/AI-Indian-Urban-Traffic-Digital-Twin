from dataclasses import dataclass
from typing import Any, Iterable, Optional


@dataclass(frozen=True)
class SUMOControlledLink:
    tls_id: str
    signal_index: int
    incoming_lane: Optional[str]
    outgoing_lane: Optional[str]


@dataclass(frozen=True)
class MovementSignalMapping:
    movement_id: str
    tls_id: str
    signal_indices: tuple[int, ...]
    incoming_lanes: tuple[str, ...]
    outgoing_lanes: tuple[str, ...]


class SUMOMovementMapper:
    """
    Dynamically maps SUMO controlled links to project movements.

    No J1/J2/J3/J4 assumptions are used.

    Matching is based on:
        SUMO incoming lane
            +
        SUMO outgoing lane
            ↓
        project movement source/destination lanes
    """

    def __init__(self, topology: Any = None):
        self.topology = topology

        self._controlled_links: dict[
            str,
            tuple[SUMOControlledLink, ...],
        ] = {}

        self._movement_mappings: dict[
            str,
            MovementSignalMapping,
        ] = {}

    # ================================================================
    # TOPOLOGY
    # ================================================================

    def set_topology(self, topology: Any) -> None:
        self.topology = topology
        self._movement_mappings.clear()

    # ================================================================
    # SUMO CONTROLLED LINKS
    # ================================================================

    def discover_controlled_links(
        self,
        traci: Any,
    ) -> dict[str, tuple[SUMOControlledLink, ...]]:
        """
        Discover all controlled links directly from SUMO.

        Expected TraCI API:

            traci.trafficlight.getIDList()

            traci.trafficlight.getControlledLinks(tls_id)

        SUMO returns controlled-link groups. Each group can contain
        one or more connection tuples.
        """

        discovered: dict[
            str,
            tuple[SUMOControlledLink, ...],
        ] = {}

        tls_ids = traci.trafficlight.getIDList()

        for tls_id in tls_ids:

            raw_links = (
                traci.trafficlight
                .getControlledLinks(tls_id)
            )

            links: list[SUMOControlledLink] = []

            for signal_index, connection_group in enumerate(
                raw_links
            ):

                if connection_group is None:
                    continue

                # SUMO normally provides a tuple/list of
                # controlled connection records.
                #
                # A signal position may control multiple
                # connections.

                if self._looks_like_single_connection(
                    connection_group
                ):
                    connection_group = [
                        connection_group
                    ]

                for connection in connection_group:

                    incoming_lane = None
                    outgoing_lane = None

                    if connection:
                        incoming_lane = (
                            connection[0]
                            if len(connection) > 0
                            else None
                        )

                        outgoing_lane = (
                            connection[1]
                            if len(connection) > 1
                            else None
                        )

                    links.append(
                        SUMOControlledLink(
                            tls_id=tls_id,
                            signal_index=signal_index,
                            incoming_lane=incoming_lane,
                            outgoing_lane=outgoing_lane,
                        )
                    )

            discovered[tls_id] = tuple(links)

        self._controlled_links = discovered

        return discovered

    @staticmethod
    def _looks_like_single_connection(
        value: Any,
    ) -> bool:

        if not isinstance(value, (list, tuple)):
            return False

        if len(value) < 2:
            return False

        first = value[0]

        return isinstance(first, str)

    # ================================================================
    # ACCESS
    # ================================================================

    def controlled_links(
        self,
        tls_id: str,
    ) -> tuple[SUMOControlledLink, ...]:

        return self._controlled_links.get(
            tls_id,
            (),
        )

    def all_controlled_links(
        self,
    ) -> tuple[SUMOControlledLink, ...]:

        result: list[SUMOControlledLink] = []

        for links in self._controlled_links.values():
            result.extend(links)

        return tuple(result)

    # ================================================================
    # MOVEMENT EXTRACTION
    # ================================================================

    def _get_junctions(self) -> Iterable[Any]:

        if self.topology is None:
            return ()

        getter = getattr(
            self.topology,
            "get_junctions",
            None,
        )

        if callable(getter):
            return getter()

        junctions = getattr(
            self.topology,
            "junctions",
            (),
        )

        if isinstance(junctions, dict):
            return junctions.values()

        return junctions

    def _get_movements(
        self,
        junction: Any,
    ) -> Iterable[Any]:

        getter = getattr(
            junction,
            "get_movements",
            None,
        )

        if callable(getter):
            return getter()

        movements = getattr(
            junction,
            "movements",
            (),
        )

        if isinstance(movements, dict):
            return movements.values()

        return movements

    @staticmethod
    def _movement_id(
        movement: Any,
    ) -> Optional[str]:

        value = getattr(
            movement,
            "movement_id",
            None,
        )

        if value is None:
            value = getattr(
                movement,
                "id",
                None,
            )

        return value

    @staticmethod
    def _lane_ids(
        movement: Any,
        attribute_names: tuple[str, ...],
    ) -> tuple[str, ...]:

        for attribute_name in attribute_names:

            value = getattr(
                movement,
                attribute_name,
                None,
            )

            if value is None:
                continue

            if isinstance(value, str):
                return (value,)

            try:
                return tuple(
                    str(item)
                    for item in value
                )
            except TypeError:
                continue

        return ()

    def _movement_sources(
        self,
        movement: Any,
    ) -> tuple[str, ...]:

        return self._lane_ids(
            movement,
            (
                "source_lanes",
                "incoming_lanes",
                "from_lanes",
                "source_lane_ids",
            ),
        )

    def _movement_destinations(
        self,
        movement: Any,
    ) -> tuple[str, ...]:

        return self._lane_ids(
            movement,
            (
                "destination_lanes",
                "outgoing_lanes",
                "to_lanes",
                "destination_lane_ids",
            ),
        )

    # ================================================================
    # MAPPING
    # ================================================================

    def build_mappings(
        self,
    ) -> tuple[MovementSignalMapping, ...]:

        mappings: list[MovementSignalMapping] = []

        for junction in self._get_junctions():

            for movement in self._get_movements(
                junction
            ):

                movement_id = self._movement_id(
                    movement
                )

                if not movement_id:
                    continue

                source_lanes = set(
                    self._movement_sources(
                        movement
                    )
                )

                destination_lanes = set(
                    self._movement_destinations(
                        movement
                    )
                )

                if not source_lanes:
                    continue

                matched: list[SUMOControlledLink] = []

                for link in self.all_controlled_links():

                    if link.incoming_lane not in source_lanes:
                        continue

                    # If destination lanes are available,
                    # require a destination match too.
                    if (
                        destination_lanes
                        and link.outgoing_lane
                        not in destination_lanes
                    ):
                        continue

                    matched.append(link)

                if not matched:
                    continue

                tls_ids = {
                    link.tls_id
                    for link in matched
                }

                # A movement should normally belong to one TLS.
                # If SUMO exposes multiple TLS IDs, preserve
                # the first deterministic mapping.
                tls_id = sorted(tls_ids)[0]

                signal_indices = tuple(
                    sorted(
                        {
                            link.signal_index
                            for link in matched
                        }
                    )
                )

                incoming = tuple(
                    sorted(
                        {
                            link.incoming_lane
                            for link in matched
                            if link.incoming_lane
                        }
                    )
                )

                outgoing = tuple(
                    sorted(
                        {
                            link.outgoing_lane
                            for link in matched
                            if link.outgoing_lane
                        }
                    )
                )

                mapping = MovementSignalMapping(
                    movement_id=movement_id,
                    tls_id=tls_id,
                    signal_indices=signal_indices,
                    incoming_lanes=incoming,
                    outgoing_lanes=outgoing,
                )

                self._movement_mappings[
                    movement_id
                ] = mapping

                mappings.append(mapping)

        return tuple(mappings)

    # ================================================================
    # LOOKUP
    # ================================================================

    def get_mapping(
        self,
        movement_id: str,
    ) -> Optional[MovementSignalMapping]:

        return self._movement_mappings.get(
            movement_id
        )

    def movement_ids(
        self,
    ) -> tuple[str, ...]:

        return tuple(
            self._movement_mappings.keys()
        )

    def signal_indices_for_movement(
        self,
        movement_id: str,
    ) -> tuple[int, ...]:

        mapping = self.get_mapping(
            movement_id
        )

        if mapping is None:
            return ()

        return mapping.signal_indices

    def tls_for_movement(
        self,
        movement_id: str,
    ) -> Optional[str]:

        mapping = self.get_mapping(
            movement_id
        )

        if mapping is None:
            return None

        return mapping.tls_id
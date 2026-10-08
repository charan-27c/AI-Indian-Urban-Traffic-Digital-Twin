from dataclasses import dataclass
from typing import Any, Optional, Protocol

from .signal_executor import SignalState


class TraCIInterface(Protocol):
    """
    Minimal TraCI interface required by this adapter.

    Keeping this as a protocol allows unit tests to use a fake
    TraCI implementation without starting SUMO.
    """

    class trafficlight:
        @staticmethod
        def getIDList() -> list[str]:
            ...

        @staticmethod
        def getRedYellowGreenState(tls_id: str) -> str:
            ...

        @staticmethod
        def setRedYellowGreenState(
            tls_id: str,
            state: str,
        ) -> None:
            ...


@dataclass(frozen=True)
class SUMOSignalLink:
    """
    One controlled connection/link belonging to a SUMO TLS.
    """

    tls_id: str
    link_index: int
    incoming_lane: Optional[str]
    outgoing_lane: Optional[str]
    movement_id: Optional[str]


@dataclass(frozen=True)
class SUMOTrafficLight:
    """
    Dynamically discovered SUMO traffic light.
    """

    tls_id: str
    signal_state: str
    links: tuple[SUMOSignalLink, ...]


class SUMOSignalAdapter:
    """
    Adapter between the project signal controller and SUMO TraCI.

    Important:
        This class does not know J1/J2/J3/J4.

    All TLS IDs and signal-state lengths are discovered from SUMO.
    """

    def __init__(self, traci: TraCIInterface):
        self.traci = traci

        self._traffic_lights: dict[str, SUMOTrafficLight] = {}

        self._movement_to_tls: dict[str, str] = {}
        self._movement_to_links: dict[
            str,
            tuple[SUMOSignalLink, ...],
        ] = {}

    # ================================================================
    # DISCOVERY
    # ================================================================

    def discover_traffic_lights(
        self,
    ) -> tuple[SUMOTrafficLight, ...]:

        tls_ids = tuple(
            self.traci.trafficlight.getIDList()
        )

        discovered: dict[str, SUMOTrafficLight] = {}

        for tls_id in tls_ids:

            state = (
                self.traci.trafficlight
                .getRedYellowGreenState(tls_id)
            )

            links = self._discover_links(
                tls_id=tls_id,
                state=state,
            )

            discovered[tls_id] = SUMOTrafficLight(
                tls_id=tls_id,
                signal_state=state,
                links=tuple(links),
            )

        self._traffic_lights = discovered

        self._rebuild_indexes()

        return tuple(
            discovered[tls_id]
            for tls_id in tls_ids
        )

    def _discover_links(
        self,
        tls_id: str,
        state: str,
    ) -> list[SUMOSignalLink]:

        links: list[SUMOSignalLink] = []

        for index, _signal in enumerate(state):

            links.append(
                SUMOSignalLink(
                    tls_id=tls_id,
                    link_index=index,
                    incoming_lane=None,
                    outgoing_lane=None,
                    movement_id=None,
                )
            )

        return links

    # ================================================================
    # INDEXES
    # ================================================================

    def _rebuild_indexes(self) -> None:

        self._movement_to_tls.clear()
        self._movement_to_links.clear()

        for tls in self._traffic_lights.values():

            for link in tls.links:

                if link.movement_id is None:
                    continue

                self._movement_to_tls[
                    link.movement_id
                ] = tls.tls_id

                existing = list(
                    self._movement_to_links.get(
                        link.movement_id,
                        (),
                    )
                )

                existing.append(link)

                self._movement_to_links[
                    link.movement_id
                ] = tuple(existing)

    # ================================================================
    # DISCOVERY ACCESS
    # ================================================================

    def traffic_lights(
        self,
    ) -> tuple[SUMOTrafficLight, ...]:

        return tuple(
            self._traffic_lights.values()
        )

    def traffic_light_ids(
        self,
    ) -> tuple[str, ...]:

        return tuple(
            self._traffic_lights.keys()
        )

    def get_traffic_light(
        self,
        tls_id: str,
    ) -> Optional[SUMOTrafficLight]:

        return self._traffic_lights.get(tls_id)

    # ================================================================
    # SIGNAL STATE
    # ================================================================

    def get_signal_state(
        self,
        tls_id: str,
    ) -> str:

        if tls_id not in self._traffic_lights:
            raise KeyError(
                f"Unknown SUMO traffic light: {tls_id}"
            )

        return (
            self.traci.trafficlight
            .getRedYellowGreenState(tls_id)
        )

    def set_signal_state(
        self,
        tls_id: str,
        state: str,
    ) -> None:

        tls = self._traffic_lights.get(tls_id)

        if tls is None:
            raise KeyError(
                f"Unknown SUMO traffic light: {tls_id}"
            )

        if len(state) != len(tls.signal_state):
            raise ValueError(
                "SUMO signal state length mismatch: "
                f"expected {len(tls.signal_state)}, "
                f"got {len(state)}"
            )

        normalized = state.upper()

        allowed = {
            "R",
            "Y",
            "G",
        }

        if any(
            character not in allowed
            for character in normalized
        ):
            raise ValueError(
                "SUMO signal state may contain only "
                "R, Y and G"
            )

        self.traci.trafficlight.setRedYellowGreenState(
            tls_id,
            normalized,
        )

    # ================================================================
    # SAFE SIGNAL STATE BUILDING
    # ================================================================

    def build_all_red_state(
        self,
        tls_id: str,
    ) -> str:

        tls = self._traffic_lights.get(tls_id)

        if tls is None:
            raise KeyError(
                f"Unknown SUMO traffic light: {tls_id}"
            )

        return "R" * len(tls.signal_state)

    def build_green_state(
        self,
        tls_id: str,
        link_indices: list[int] | tuple[int, ...],
    ) -> str:

        tls = self._traffic_lights.get(tls_id)

        if tls is None:
            raise KeyError(
                f"Unknown SUMO traffic light: {tls_id}"
            )

        state = ["R"] * len(tls.signal_state)

        for index in link_indices:

            if index < 0 or index >= len(state):
                raise IndexError(
                    f"Invalid SUMO signal index: {index}"
                )

            state[index] = "G"

        return "".join(state)

    def build_yellow_state(
        self,
        tls_id: str,
        link_indices: list[int] | tuple[int, ...],
    ) -> str:

        tls = self._traffic_lights.get(tls_id)

        if tls is None:
            raise KeyError(
                f"Unknown SUMO traffic light: {tls_id}"
            )

        state = ["R"] * len(tls.signal_state)

        for index in link_indices:

            if index < 0 or index >= len(state):
                raise IndexError(
                    f"Invalid SUMO signal index: {index}"
                )

            state[index] = "Y"

        return "".join(state)

    # ================================================================
    # APPLY SAFE STATE
    # ================================================================

    def apply_state(
        self,
        tls_id: str,
        signal_state: SignalState,
        link_indices: list[int] | tuple[int, ...] = (),
    ) -> str:

        if signal_state == SignalState.ALL_RED:
            state = self.build_all_red_state(
                tls_id
            )

        elif signal_state == SignalState.GREEN:
            state = self.build_green_state(
                tls_id,
                link_indices,
            )

        elif signal_state == SignalState.YELLOW:
            state = self.build_yellow_state(
                tls_id,
                link_indices,
            )

        elif signal_state == SignalState.RED:
            state = self.build_all_red_state(
                tls_id
            )

        else:
            raise ValueError(
                f"Unsupported signal state: "
                f"{signal_state}"
            )

        self.set_signal_state(
            tls_id,
            state,
        )

        return state
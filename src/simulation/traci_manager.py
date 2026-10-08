from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Optional


class TraCIConnectionError(RuntimeError):
    """Raised when SUMO/TraCI cannot be started or connected."""


class TraCIManager:
    """
    Runtime bridge between the AI traffic system and SUMO.

    Responsibilities:
        - validate SUMO configuration
        - locate SUMO / SUMO-GUI
        - start SUMO
        - connect through TraCI
        - discover traffic lights dynamically
        - inspect controlled links
        - advance simulation
        - close SUMO safely

    No junction IDs are hard-coded.
    """

    def __init__(
        self,
        config_file: str | Path,
        *,
        gui: bool = False,
        step_length: float = 1.0,
        seed: Optional[int] = None,
        traci_module: Any = None,
    ) -> None:

        self.config_file = Path(config_file).resolve()
        self.gui = gui
        self.step_length = float(step_length)
        self.seed = seed

        self._traci = traci_module
        self._process: Optional[subprocess.Popen] = None
        self._connected = False
        self._started_by_manager = False

    # ============================================================
    # VALIDATION
    # ============================================================

    def validate_config(self) -> Path:
        """Validate the SUMO configuration file."""

        if not self.config_file.exists():
            raise FileNotFoundError(
                f"SUMO configuration not found: "
                f"{self.config_file}"
            )

        if self.config_file.suffix != ".sumocfg":
            raise ValueError(
                "Expected a SUMO .sumocfg file: "
                f"{self.config_file}"
            )

        return self.config_file

    # ============================================================
    # SUMO EXECUTABLE
    # ============================================================

    def find_sumo_binary(self) -> str:
        """
        Locate SUMO executable.

        Priority:
            SUMO_BINARY environment variable
            SUMO_HOME/bin/sumo
            PATH
        """

        env_binary = os.environ.get("SUMO_BINARY")

        if env_binary:
            path = Path(env_binary)

            if path.exists():
                return str(path)

        sumo_home = os.environ.get("SUMO_HOME")

        if sumo_home:
            candidate = (
                Path(sumo_home)
                / "bin"
                / (
                    "sumo-gui.exe"
                    if self.gui
                    else "sumo.exe"
                )
            )

            if candidate.exists():
                return str(candidate)

        executable = (
            "sumo-gui.exe"
            if self.gui
            else "sumo.exe"
        )

        located = shutil.which(executable)

        if located:
            return located

        raise TraCIConnectionError(
            "SUMO executable could not be found. "
            "Set SUMO_HOME or SUMO_BINARY."
        )

    # ============================================================
    # TRACI IMPORT
    # ============================================================

    def _load_traci(self) -> Any:
        """Load the TraCI Python module lazily."""

        if self._traci is not None:
            return self._traci

        try:
            import traci
        except ImportError as exc:
            raise TraCIConnectionError(
                "Python package 'traci' is not available. "
                "Install/use the TraCI package from SUMO."
            ) from exc

        self._traci = traci

        return traci

    # ============================================================
    # START SUMO
    # ============================================================

    def start(
        self,
        *,
        label: str = "ai_traffic_digital_twin",
    ) -> None:
        """
        Start SUMO and establish TraCI connection.
        """

        if self._connected:
            return

        self.validate_config()

        traci = self._load_traci()

        sumo_binary = self.find_sumo_binary()

        command = [
            sumo_binary,
            "-c",
            str(self.config_file),
            "--step-length",
            str(self.step_length),
        ]

        if self.seed is not None:
            command.extend(
                [
                    "--seed",
                    str(self.seed),
                ]
            )

        try:
            traci.start(
                command,
                label=label,
            )
        except Exception as exc:
            raise TraCIConnectionError(
                "Failed to start SUMO through TraCI."
            ) from exc

        self._connected = True
        self._started_by_manager = True

    # ============================================================
    # EXTERNAL CONNECTION
    # ============================================================

    def attach(
        self,
        *,
        label: str = "ai_traffic_digital_twin",
    ) -> None:
        """
        Attach to an already running TraCI connection.
        """

        traci = self._load_traci()

        try:
            traci.switch(label)
        except Exception as exc:
            raise TraCIConnectionError(
                f"Unable to attach to TraCI connection: {label}"
            ) from exc

        self._connected = True
        self._started_by_manager = False

    # ============================================================
    # STATUS
    # ============================================================

    @property
    def connected(self) -> bool:
        return self._connected

    # ============================================================
    # SIMULATION TIME
    # ============================================================

    def simulation_time(self) -> float:
        self._require_connection()

        traci = self._load_traci()

        return float(
            traci.simulation.getTime()
        )

    # ============================================================
    # STEP
    # ============================================================

    def step(
        self,
        step: Optional[float] = None,
    ) -> float:
        """
        Advance SUMO by one simulation step.
        """

        self._require_connection()

        traci = self._load_traci()

        if step is None:
            step = self.step_length

        target_time = (
            self.simulation_time()
            + float(step)
        )

        traci.simulationStep(
            target_time
        )

        return self.simulation_time()

    # ============================================================
    # TRAFFIC LIGHT DISCOVERY
    # ============================================================

    def traffic_light_ids(self) -> tuple[str, ...]:
        """
        Discover all SUMO traffic lights dynamically.
        """

        self._require_connection()

        traci = self._load_traci()

        ids = traci.trafficlight.getIDList()

        return tuple(ids)

    # ============================================================
    # CONTROLLED LINKS
    # ============================================================

    def controlled_links(
        self,
        traffic_light_id: str,
    ) -> tuple[Any, ...]:
        """
        Return controlled links for one traffic light.
        """

        self._require_connection()

        traci = self._load_traci()

        links = traci.trafficlight.getControlledLinks(
            traffic_light_id
        )

        return tuple(links)

    # ============================================================
    # SIGNAL STATE
    # ============================================================

    def signal_state(
        self,
        traffic_light_id: str,
    ) -> str:
        """Return the current SUMO signal state."""

        self._require_connection()

        traci = self._load_traci()

        return str(
            traci.trafficlight.getRedYellowGreenState(
                traffic_light_id
            )
        )

    # ============================================================
    # SET SIGNAL STATE
    # ============================================================

    def set_signal_state(
        self,
        traffic_light_id: str,
        state: str,
    ) -> None:
        """
        Apply a complete SUMO RYG state.

        Safety decisions remain owned by the
        SafeSignalExecutor / controller pipeline.
        """

        self._require_connection()

        traci = self._load_traci()

        traci.trafficlight.setRedYellowGreenState(
            traffic_light_id,
            state,
        )

    # ============================================================
    # VEHICLE DISCOVERY
    # ============================================================

    def vehicle_ids(self) -> tuple[str, ...]:
        """Return all vehicles currently inside SUMO."""

        self._require_connection()

        traci = self._load_traci()

        return tuple(
            traci.vehicle.getIDList()
        )

    # ============================================================
    # VEHICLE STATE
    # ============================================================

    def vehicle_state(
        self,
        vehicle_id: str,
    ) -> dict[str, Any]:
        """
        Read basic runtime vehicle state.

        This is intentionally generic so the perception/state
        engine can later consume the data.
        """

        self._require_connection()

        traci = self._load_traci()

        return {
            "vehicle_id": vehicle_id,
            "speed": float(
                traci.vehicle.getSpeed(
                    vehicle_id
                )
            ),
            "lane_id": str(
                traci.vehicle.getLaneID(
                    vehicle_id
                )
            ),
            "edge_id": str(
                traci.vehicle.getRoadID(
                    vehicle_id
                )
            ),
            "position": tuple(
                traci.vehicle.getPosition(
                    vehicle_id
                )
            ),
        }

    # ============================================================
    # CLOSE
    # ============================================================

    def close(self) -> None:
        """
        Close TraCI safely.

        Safe to call repeatedly.
        """

        if not self._connected:
            return

        try:
            traci = self._load_traci()
            traci.close()
        finally:
            self._connected = False
            self._started_by_manager = False
            self._process = None

    # ============================================================
    # CONTEXT MANAGER
    # ============================================================

    def __enter__(self) -> "TraCIManager":
        self.start()
        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ) -> None:
        self.close()

    # ============================================================
    # INTERNAL SAFETY
    # ============================================================

    def _require_connection(self) -> None:
        if not self._connected:
            raise TraCIConnectionError(
                "TraCI is not connected. "
                "Call start() or attach() first."
            )
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class TrafficFeatureConfig:
    """
    Configuration for traffic feature derivation.

    Units must be confirmed against the input dataset before
    using these values with real PEMS-BAY data.
    """

    free_flow_speed: float = 65.0

    vehicle_length: float = 5.0

    interval_seconds: float = 5.0


class TrafficFeatureEngineer:
    """
    Converts raw traffic measurements into the feature space
    consumed by the prediction model.

    No junction, lane, approach, or movement is hard-coded.
    """

    def __init__(
        self,
        config: Optional[TrafficFeatureConfig] = None,
    ) -> None:

        self.config = (
            config
            if config is not None
            else TrafficFeatureConfig()
        )

    # ========================================================
    # FLOW → VEHICLE COUNT
    # ========================================================

    def vehicle_count_from_flow(
        self,
        flow_rate: float,
    ) -> float:

        if flow_rate < 0:
            raise ValueError(
                "flow_rate cannot be negative."
            )

        return (
            flow_rate
            * self.config.interval_seconds
            / 3600.0
        )

    # ========================================================
    # FLOW + SPEED → DENSITY
    # ========================================================

    def density_from_flow_speed(
        self,
        flow_rate: float,
        speed: float,
    ) -> float:

        if flow_rate < 0:
            raise ValueError(
                "flow_rate cannot be negative."
            )

        if speed < 0:
            raise ValueError(
                "speed cannot be negative."
            )

        if speed == 0:
            return 0.0

        return flow_rate / speed

    # ========================================================
    # SPEED → DELAY
    # ========================================================

    def delay_from_speed(
        self,
        speed: float,
    ) -> float:

        if speed < 0:
            raise ValueError(
                "speed cannot be negative."
            )

        free_flow = (
            self.config.free_flow_speed
        )

        if speed >= free_flow:
            return 0.0

        if speed == 0:
            return free_flow

        return (
            free_flow - speed
        ) / free_flow

    # ========================================================
    # PRESSURE
    # ========================================================

    def pressure(
        self,
        queue_length: float,
        waiting_time: float,
        vehicle_count: float,
    ) -> float:

        if queue_length < 0:
            raise ValueError(
                "queue_length cannot be negative."
            )

        if waiting_time < 0:
            raise ValueError(
                "waiting_time cannot be negative."
            )

        if vehicle_count < 0:
            raise ValueError(
                "vehicle_count cannot be negative."
            )

        return (
            queue_length
            + waiting_time
            + vehicle_count
        )
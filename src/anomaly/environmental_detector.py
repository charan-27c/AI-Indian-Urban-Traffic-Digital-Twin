from typing import List, Optional

from .environmental_anomaly import (
    EnvironmentalAnomaly,
    EnvironmentalAnomalyType,
    EnvironmentalObservation,
)


class EnvironmentalAnomalyDetector:
    """
    Detects persistent environmental and road-condition anomalies.

    No fixed junction, lane, approach, or movement is assumed.
    """

    def __init__(
        self,
        min_confidence: float = 0.50,
        persistence_time: float = 2.0,
    ) -> None:

        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError(
                "min_confidence must be between 0 and 1."
            )

        if persistence_time < 0:
            raise ValueError(
                "persistence_time must be non-negative."
            )

        self.min_confidence = min_confidence
        self.persistence_time = persistence_time

        self._observations = {}

    # ========================================================
    # OBSERVATION
    # ========================================================

    def observe(
        self,
        observation: EnvironmentalObservation,
    ) -> Optional[EnvironmentalAnomaly]:

        if observation.confidence < self.min_confidence:
            return None

        key = (
            observation.camera_id,
            observation.anomaly_type,
        )

        previous = self._observations.get(key)

        if previous is None:

            self._observations[key] = observation

            if self.persistence_time == 0:

                return self._create_anomaly(
                    observation,
                    observation,
                )

            return None

        self._observations[key] = observation

        elapsed = (
            observation.timestamp
            - previous.timestamp
        )

        if elapsed < self.persistence_time:
            return None

        return self._create_anomaly(
            previous,
            observation,
        )

    # ========================================================
    # CREATE ANOMALY
    # ========================================================

    def _create_anomaly(
        self,
        first: EnvironmentalObservation,
        latest: EnvironmentalObservation,
    ) -> EnvironmentalAnomaly:

        return EnvironmentalAnomaly(
            anomaly_type=latest.anomaly_type,
            camera_id=latest.camera_id,

            first_seen=first.timestamp,
            last_seen=latest.timestamp,

            confidence=max(
                first.confidence,
                latest.confidence,
            ),

            severity=max(
                first.severity,
                latest.severity,
            ),

            affected_area=max(
                first.affected_area,
                latest.affected_area,
            ),

            active=True,
        )

    # ========================================================
    # BATCH
    # ========================================================

    def process(
        self,
        observations: List[EnvironmentalObservation],
    ) -> List[EnvironmentalAnomaly]:

        anomalies = []

        for observation in observations:

            anomaly = self.observe(
                observation
            )

            if anomaly is not None:
                anomalies.append(anomaly)

        return anomalies

    # ========================================================
    # RESET
    # ========================================================

    def reset(self) -> None:

        self._observations.clear()
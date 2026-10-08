from src.anomaly.environmental_anomaly import (
    EnvironmentalAnomalyType,
    EnvironmentalObservation,
)

from src.anomaly.environmental_detector import (
    EnvironmentalAnomalyDetector,
)


def make_observation(
    timestamp: float,
    anomaly_type: EnvironmentalAnomalyType,
    confidence: float = 0.90,
    severity: float = 0.5,
    affected_area: float = 10.0,
):

    return EnvironmentalObservation(
        timestamp=timestamp,
        camera_id="CAM_TEST",
        anomaly_type=anomaly_type,
        confidence=confidence,
        severity=severity,
        affected_area=affected_area,
    )


# ============================================================
# FLOODING
# ============================================================

def test_flooding_detection():

    detector = EnvironmentalAnomalyDetector(
        persistence_time=2.0
    )

    assert detector.observe(
        make_observation(
            0.0,
            EnvironmentalAnomalyType.FLOODING,
        )
    ) is None

    result = detector.observe(
        make_observation(
            2.0,
            EnvironmentalAnomalyType.FLOODING,
        )
    )

    assert result is not None
    assert result.anomaly_type == (
        EnvironmentalAnomalyType.FLOODING
    )


# ============================================================
# ROAD BLOCKAGE
# ============================================================

def test_road_blockage_detection():

    detector = EnvironmentalAnomalyDetector(
        persistence_time=2.0
    )

    detector.observe(
        make_observation(
            10.0,
            EnvironmentalAnomalyType.ROAD_BLOCKAGE,
        )
    )

    result = detector.observe(
        make_observation(
            12.0,
            EnvironmentalAnomalyType.ROAD_BLOCKAGE,
        )
    )

    assert result is not None
    assert result.anomaly_type == (
        EnvironmentalAnomalyType.ROAD_BLOCKAGE
    )


# ============================================================
# ACCIDENT
# ============================================================

def test_accident_detection():

    detector = EnvironmentalAnomalyDetector(
        persistence_time=1.0
    )

    detector.observe(
        make_observation(
            20.0,
            EnvironmentalAnomalyType.ACCIDENT,
            severity=0.9,
        )
    )

    result = detector.observe(
        make_observation(
            21.0,
            EnvironmentalAnomalyType.ACCIDENT,
            severity=0.9,
        )
    )

    assert result is not None
    assert result.anomaly_type == (
        EnvironmentalAnomalyType.ACCIDENT
    )

    assert result.severity == 0.9


# ============================================================
# LOW CONFIDENCE
# ============================================================

def test_low_confidence_is_ignored():

    detector = EnvironmentalAnomalyDetector(
        min_confidence=0.70
    )

    result = detector.observe(
        make_observation(
            0.0,
            EnvironmentalAnomalyType.FLOODING,
            confidence=0.40,
        )
    )

    assert result is None


# ============================================================
# DIFFERENT ANOMALIES
# ============================================================

def test_different_anomalies_are_independent():

    detector = EnvironmentalAnomalyDetector(
        persistence_time=1.0
    )

    detector.observe(
        make_observation(
            0.0,
            EnvironmentalAnomalyType.FLOODING,
        )
    )

    detector.observe(
        make_observation(
            0.0,
            EnvironmentalAnomalyType.ROAD_BLOCKAGE,
        )
    )

    flooding = detector.observe(
        make_observation(
            1.0,
            EnvironmentalAnomalyType.FLOODING,
        )
    )

    blockage = detector.observe(
        make_observation(
            1.0,
            EnvironmentalAnomalyType.ROAD_BLOCKAGE,
        )
    )

    assert flooding is not None
    assert blockage is not None


# ============================================================
# RESET
# ============================================================

def test_environmental_detector_reset():

    detector = EnvironmentalAnomalyDetector(
        persistence_time=1.0
    )

    detector.observe(
        make_observation(
            0.0,
            EnvironmentalAnomalyType.FLOODING,
        )
    )

    detector.reset()

    result = detector.observe(
        make_observation(
            2.0,
            EnvironmentalAnomalyType.FLOODING,
        )
    )

    assert result is None
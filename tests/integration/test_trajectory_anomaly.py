from src.anomaly.trajectory_features import (
    VehicleObservation,
    TrajectoryFeatureExtractor,
)

from src.anomaly.trajectory_detector import (
    TrajectoryAnomalyDetector,
)

from src.anomaly.models import (
    AnomalyType,
)


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def test_trajectory_feature_extraction():

    previous = VehicleObservation(
        vehicle_id="V1",
        timestamp=0.0,
        x=0.0,
        y=0.0,
        speed=10.0,
        lane_id="L1",
    )

    current = VehicleObservation(
        vehicle_id="V1",
        timestamp=1.0,
        x=10.0,
        y=0.0,
        speed=12.0,
        lane_id="L1",
    )

    extractor = TrajectoryFeatureExtractor()

    features = extractor.extract(
        previous,
        current,
    )

    assert features.vehicle_id == "V1"

    assert features.displacement == 10.0

    assert features.acceleration == 2.0

    assert features.lane_change is False


# ============================================================
# VEHICLE ID VALIDATION
# ============================================================

def test_trajectory_requires_same_vehicle():

    previous = VehicleObservation(
        vehicle_id="V1",
        timestamp=0.0,
        x=0.0,
        y=0.0,
    )

    current = VehicleObservation(
        vehicle_id="V2",
        timestamp=1.0,
        x=1.0,
        y=1.0,
    )

    extractor = TrajectoryFeatureExtractor()

    try:
        extractor.extract(
            previous,
            current,
        )

        assert False

    except ValueError:
        assert True


# ============================================================
# TIMESTAMP VALIDATION
# ============================================================

def test_trajectory_requires_increasing_timestamp():

    previous = VehicleObservation(
        vehicle_id="V1",
        timestamp=10.0,
        x=0.0,
        y=0.0,
    )

    current = VehicleObservation(
        vehicle_id="V1",
        timestamp=10.0,
        x=1.0,
        y=1.0,
    )

    extractor = TrajectoryFeatureExtractor()

    try:
        extractor.extract(
            previous,
            current,
        )

        assert False

    except ValueError:
        assert True


# ============================================================
# LANE CHANGE
# ============================================================

def test_lane_change_is_detected():

    previous = VehicleObservation(
        vehicle_id="V1",
        timestamp=0.0,
        x=0.0,
        y=0.0,
        speed=10.0,
        lane_id="L1",
    )

    current = VehicleObservation(
        vehicle_id="V1",
        timestamp=1.0,
        x=10.0,
        y=1.0,
        speed=10.0,
        lane_id="L2",
    )

    extractor = TrajectoryFeatureExtractor()

    features = extractor.extract(
        previous,
        current,
    )

    assert features.lane_change is True


# ============================================================
# HEADING CHANGE
# ============================================================

def test_heading_change_is_calculated():

    previous = VehicleObservation(
        vehicle_id="V1",
        timestamp=0.0,
        x=0.0,
        y=0.0,
        speed=10.0,
        heading=0.0,
    )

    current = VehicleObservation(
        vehicle_id="V1",
        timestamp=1.0,
        x=5.0,
        y=5.0,
        speed=10.0,
        heading=90.0,
    )

    extractor = TrajectoryFeatureExtractor()

    features = extractor.extract(
        previous,
        current,
    )

    assert features.heading_change == 90.0


# ============================================================
# STOPPED VEHICLE
# ============================================================

def test_stopped_vehicle_detection():

    detector = TrajectoryAnomalyDetector()

    previous = VehicleObservation(
        vehicle_id="V1",
        timestamp=0.0,
        x=0.0,
        y=0.0,
        speed=0.5,
    )

    current = VehicleObservation(
        vehicle_id="V1",
        timestamp=31.0,
        x=0.0,
        y=0.0,
        speed=0.5,
    )

    extractor = TrajectoryFeatureExtractor()

    features = extractor.extract(
        previous,
        current,
    )

    events = detector.detect(
        features,
        timestamp=31.0,
        junction_id="J1",
        lane_id="L1",
    )

    assert any(
        event.anomaly_type
        == AnomalyType.STOPPED_VEHICLE
        for event in events
    )


# ============================================================
# SUDDEN BRAKING
# ============================================================

def test_sudden_braking_detection():

    detector = TrajectoryAnomalyDetector()

    previous = VehicleObservation(
        vehicle_id="V1",
        timestamp=0.0,
        x=0.0,
        y=0.0,
        speed=20.0,
    )

    current = VehicleObservation(
        vehicle_id="V1",
        timestamp=1.0,
        x=10.0,
        y=0.0,
        speed=10.0,
    )

    extractor = TrajectoryFeatureExtractor()

    features = extractor.extract(
        previous,
        current,
    )

    events = detector.detect(
        features,
        timestamp=1.0,
        junction_id="J1",
    )

    assert any(
        event.anomaly_type
        == AnomalyType.SUDDEN_CONGESTION
        for event in events
    )


# ============================================================
# WRONG WAY
# ============================================================

def test_wrong_way_detection():

    detector = TrajectoryAnomalyDetector()

    features = (
        __import__(
            "src.anomaly.trajectory_features",
            fromlist=["TrajectoryFeatures"],
        ).TrajectoryFeatures(
            vehicle_id="V1",
            speed=10.0,
            acceleration=0.0,
            displacement=10.0,
            stationary_duration=0.0,
            heading_change=0.0,
            wrong_way_score=0.95,
            sudden_braking_score=0.0,
            lane_change=False,
        )
    )

    events = detector.detect(
        features,
        timestamp=10.0,
        junction_id="J1",
        movement_id="M1",
        lane_id="L1",
    )

    assert any(
        event.anomaly_type
        == AnomalyType.WRONG_WAY
        for event in events
    )


# ============================================================
# NORMAL VEHICLE
# ============================================================

def test_normal_vehicle_has_no_anomaly():

    detector = TrajectoryAnomalyDetector()

    previous = VehicleObservation(
        vehicle_id="V1",
        timestamp=0.0,
        x=0.0,
        y=0.0,
        speed=10.0,
    )

    current = VehicleObservation(
        vehicle_id="V1",
        timestamp=1.0,
        x=10.0,
        y=0.0,
        speed=11.0,
    )

    extractor = TrajectoryFeatureExtractor()

    features = extractor.extract(
        previous,
        current,
    )

    events = detector.detect(
        features,
        timestamp=1.0,
    )

    assert events == []
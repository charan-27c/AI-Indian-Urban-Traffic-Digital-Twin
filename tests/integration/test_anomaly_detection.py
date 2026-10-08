
import pytest

from src.anomaly.models import (
    AnomalyEvent,
    AnomalyType,
)

from src.anomaly.feature_engine import (
    AnomalyFeatures,
)

from src.anomaly.detector import (
    TrafficAnomalyDetector,
)

from src.anomaly.incident_manager import (
    IncidentManager,
)


# ============================================================
# HELPERS
# ============================================================

def normal_features():

    return AnomalyFeatures(
        vehicle_count=10,
        stopped_vehicle_count=0,
        average_speed=30,
        queue_length=3,
        density=0.2,
        occupancy=0.3,
        speed_drop=0,
        queue_growth=0,
        occupancy_growth=0,
        stationary_duration=0,
    )


# ============================================================
# NORMAL TRAFFIC
# ============================================================

def test_normal_traffic_produces_no_anomaly():

    detector = TrafficAnomalyDetector()

    features = normal_features()

    events = detector.detect(
        features=features,
        timestamp=10.0,
        junction_id="J_TEST",
        movement_id="M_TEST",
        lane_id="L_TEST",
    )

    assert events == []


# ============================================================
# SUDDEN CONGESTION
# ============================================================

def test_sudden_congestion_is_detected():

    detector = TrafficAnomalyDetector()

    features = AnomalyFeatures(
        vehicle_count=40,
        average_speed=5,
        queue_length=20,
        density=0.8,
        occupancy=0.85,
        speed_drop=10,
        queue_growth=8,
    )

    events = detector.detect(
        features=features,
        timestamp=20.0,
        junction_id="J1",
        movement_id="M1",
        lane_id="L1",
    )

    assert len(events) >= 1

    assert any(
        event.anomaly_type
        == AnomalyType.SUDDEN_CONGESTION
        for event in events
    )


# ============================================================
# BLOCKAGE
# ============================================================

def test_blockage_is_detected():

    detector = TrafficAnomalyDetector()

    features = AnomalyFeatures(
        vehicle_count=50,
        average_speed=1,
        queue_length=20,
        density=0.9,
        occupancy=0.9,
    )

    events = detector.detect(
        features=features,
        timestamp=30.0,
        junction_id="J2",
        movement_id="M2",
        lane_id="L2",
    )

    assert any(
        event.anomaly_type
        == AnomalyType.BLOCKAGE
        for event in events
    )


# ============================================================
# STOPPED VEHICLE
# ============================================================

def test_stopped_vehicle_is_detected():

    detector = TrafficAnomalyDetector()

    features = AnomalyFeatures(
        vehicle_count=15,
        stopped_vehicle_count=1,
        average_speed=0,
        queue_length=5,
        density=0.4,
        occupancy=0.5,
        stationary_duration=40,
    )

    events = detector.detect(
        features=features,
        timestamp=40.0,
        junction_id="J3",
        movement_id="M3",
        lane_id="L3",
    )

    assert any(
        event.anomaly_type
        == AnomalyType.STOPPED_VEHICLE
        for event in events
    )


# ============================================================
# DYNAMIC TOPOLOGY REFERENCES
# ============================================================

def test_anomaly_preserves_dynamic_topology():

    detector = TrafficAnomalyDetector()

    features = AnomalyFeatures(
        vehicle_count=20,
        average_speed=1,
        queue_length=20,
        density=0.8,
        occupancy=0.9,
    )

    events = detector.detect(
        features=features,
        timestamp=50.0,
        junction_id="J_DYNAMIC",
        movement_id="M_DYNAMIC",
        lane_id="LANE_DYNAMIC",
    )

    assert len(events) > 0

    for event in events:

        assert event.junction_id == "J_DYNAMIC"
        assert event.movement_id == "M_DYNAMIC"
        assert event.lane_id == "LANE_DYNAMIC"


# ============================================================
# CONFIDENCE
# ============================================================

def test_anomaly_confidence_is_valid():

    detector = TrafficAnomalyDetector()

    features = AnomalyFeatures(
        vehicle_count=50,
        average_speed=1,
        queue_length=20,
        density=0.9,
        occupancy=0.9,
    )

    events = detector.detect(
        features=features,
        timestamp=60.0,
    )

    assert len(events) > 0

    for event in events:

        assert 0.0 <= event.confidence <= 1.0


# ============================================================
# SEVERITY
# ============================================================

def test_anomaly_severity_is_valid():

    detector = TrafficAnomalyDetector()

    features = AnomalyFeatures(
        vehicle_count=50,
        average_speed=1,
        queue_length=20,
        density=0.9,
        occupancy=0.9,
    )

    events = detector.detect(
        features=features,
        timestamp=70.0,
    )

    assert len(events) > 0

    for event in events:

        assert 0.0 <= event.severity <= 1.0


# ============================================================
# MULTIPLE ANOMALIES
# ============================================================

def test_multiple_anomalies_can_be_detected():

    detector = TrafficAnomalyDetector()

    features = AnomalyFeatures(
        vehicle_count=50,
        stopped_vehicle_count=2,
        average_speed=1,
        queue_length=20,
        density=0.9,
        occupancy=0.9,
        speed_drop=10,
        queue_growth=8,
        stationary_duration=40,
    )

    events = detector.detect(
        features=features,
        timestamp=80.0,
        junction_id="J4",
    )

    types = {
        event.anomaly_type
        for event in events
    }

    assert AnomalyType.SUDDEN_CONGESTION in types
    assert AnomalyType.BLOCKAGE in types
    assert AnomalyType.STOPPED_VEHICLE in types


# ============================================================
# INCIDENT MANAGER
# ============================================================

def test_incident_manager_registers_event():

    manager = IncidentManager()

    event = AnomalyEvent(
        anomaly_type=AnomalyType.BLOCKAGE,
        confidence=0.9,
        severity=0.8,
        timestamp=100.0,
        junction_id="J1",
        movement_id="M1",
        lane_id="L1",
    )

    incidents = manager.update(
        [event]
    )

    assert len(incidents) == 1

    assert incidents[0] == event


# ============================================================
# INCIDENT DEDUPLICATION
# ============================================================

def test_incident_manager_deduplicates_same_incident():

    manager = IncidentManager()

    event1 = AnomalyEvent(
        anomaly_type=AnomalyType.BLOCKAGE,
        confidence=0.8,
        severity=0.7,
        timestamp=100.0,
        junction_id="J1",
        movement_id="M1",
        lane_id="L1",
    )

    event2 = AnomalyEvent(
        anomaly_type=AnomalyType.BLOCKAGE,
        confidence=0.9,
        severity=0.8,
        timestamp=105.0,
        junction_id="J1",
        movement_id="M1",
        lane_id="L1",
    )

    manager.update([event1])
    incidents = manager.update([event2])

    assert len(incidents) == 1

    assert incidents[0].timestamp == 105.0


# ============================================================
# DIFFERENT INCIDENTS
# ============================================================

def test_incident_manager_keeps_different_incidents():

    manager = IncidentManager()

    blockage = AnomalyEvent(
        anomaly_type=AnomalyType.BLOCKAGE,
        confidence=0.9,
        severity=0.8,
        timestamp=100.0,
        junction_id="J1",
        movement_id="M1",
        lane_id="L1",
    )

    stopped_vehicle = AnomalyEvent(
        anomaly_type=AnomalyType.STOPPED_VEHICLE,
        confidence=0.9,
        severity=0.6,
        timestamp=101.0,
        junction_id="J1",
        movement_id="M2",
        lane_id="L2",
    )

    incidents = manager.update(
        [
            blockage,
            stopped_vehicle,
        ]
    )

    assert len(incidents) == 2


# ============================================================
# RESET
# ============================================================

def test_incident_manager_reset():

    manager = IncidentManager()

    event = AnomalyEvent(
        anomaly_type=AnomalyType.BLOCKAGE,
        confidence=0.9,
        severity=0.8,
        timestamp=100.0,
        junction_id="J1",
    )

    manager.update([event])

    assert len(manager.active_incidents) == 1

    manager.reset()

    assert len(manager.active_incidents) == 0

from src.incidents.incident import (
    IncidentStatus,
    IncidentType,
)

from src.incidents.incident_manager import (
    IncidentManager,
)


def test_register_accident():

    manager = IncidentManager()

    incident = manager.register(
        incident_type=IncidentType.ACCIDENT,
        timestamp=10.0,
        confidence=0.95,
        severity=0.9,
        camera_id="CAM_J1_NORTH",
        junction_id="J1",
    )

    assert incident is not None
    assert incident.incident_type == (
        IncidentType.ACCIDENT
    )

    assert incident.status == (
        IncidentStatus.ACTIVE
    )


def test_register_flooding():

    manager = IncidentManager()

    incident = manager.register(
        incident_type=IncidentType.FLOODING,
        timestamp=20.0,
        confidence=0.92,
        severity=0.8,
        camera_id="CAM_J2_NORTH",
        junction_id="J2",
        affected_area=100.0,
    )

    assert incident.incident_type == (
        IncidentType.FLOODING
    )

    assert incident.affected_area == 100.0


def test_register_road_blockage():

    manager = IncidentManager()

    incident = manager.register(
        incident_type=IncidentType.ROAD_BLOCKAGE,
        timestamp=30.0,
        confidence=0.88,
        severity=0.7,
        camera_id="CAM_J3",
        junction_id="J3",
    )

    assert incident.incident_type == (
        IncidentType.ROAD_BLOCKAGE
    )


def test_different_incidents_are_preserved():

    manager = IncidentManager()

    accident = manager.register(
        incident_type=IncidentType.ACCIDENT,
        timestamp=10.0,
        confidence=0.9,
        severity=0.9,
        camera_id="CAM1",
        junction_id="J1",
    )

    flooding = manager.register(
        incident_type=IncidentType.FLOODING,
        timestamp=11.0,
        confidence=0.9,
        severity=0.8,
        camera_id="CAM2",
        junction_id="J2",
    )

    active = manager.active_incidents()

    assert len(active) == 2

    assert accident.incident_id != (
        flooding.incident_id
    )


def test_repeated_detection_is_deduplicated():

    manager = IncidentManager()

    first = manager.register(
        incident_type=IncidentType.ROAD_BLOCKAGE,
        timestamp=10.0,
        confidence=0.80,
        severity=0.5,
        camera_id="CAM1",
        junction_id="J1",
    )

    second = manager.register(
        incident_type=IncidentType.ROAD_BLOCKAGE,
        timestamp=12.0,
        confidence=0.95,
        severity=0.8,
        camera_id="CAM1",
        junction_id="J1",
    )

    assert first.incident_id == (
        second.incident_id
    )

    assert second.confidence == 0.95
    assert second.severity == 0.8

    assert len(
        manager.active_incidents()
    ) == 1


def test_resolve_incident():

    manager = IncidentManager()

    incident = manager.register(
        incident_type=IncidentType.ACCIDENT,
        timestamp=10.0,
        confidence=0.95,
        severity=1.0,
        camera_id="CAM1",
        junction_id="J1",
    )

    resolved = manager.resolve(
        incident.incident_id,
        timestamp=50.0,
    )

    assert resolved is not None

    assert resolved.status == (
        IncidentStatus.RESOLVED
    )

    assert len(
        manager.active_incidents()
    ) == 0


def test_reset():

    manager = IncidentManager()

    manager.register(
        incident_type=IncidentType.FLOODING,
        timestamp=1.0,
        confidence=0.9,
        severity=0.8,
    )

    manager.reset()

    assert len(
        manager.all_incidents()
    ) == 0
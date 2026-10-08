from src.incidents.incident import (
    IncidentType,
)

from src.incidents.incident_manager import (
    IncidentManager,
)

from src.incidents.impact_analyzer import (
    IncidentImpactAnalyzer,
)


def create_incident(
    manager,
    incident_type,
    severity=0.8,
):

    return manager.register(
        incident_type=incident_type,
        timestamp=10.0,
        confidence=0.95,
        severity=severity,
        camera_id="CAM_TEST",
        junction_id="J_DYNAMIC",
    )


def test_accident_creates_network_impact():

    manager = IncidentManager()

    incident = create_incident(
        manager,
        IncidentType.ACCIDENT,
    )

    analyzer = IncidentImpactAnalyzer()

    impact = analyzer.analyze(
        incident,
        affected_edges=["edge_a"],
        affected_lanes=["lane_a"],
        affected_movements=["movement_a"],
    )

    assert impact.blocked is True

    assert impact.affected_edges == (
        "edge_a",
    )

    assert impact.affected_lanes == (
        "lane_a",
    )

    assert impact.affected_movements == (
        "movement_a",
    )

    assert impact.capacity_reduction > 0.0


def test_road_blockage_has_high_spillback_risk():

    manager = IncidentManager()

    incident = create_incident(
        manager,
        IncidentType.ROAD_BLOCKAGE,
        severity=0.9,
    )

    analyzer = IncidentImpactAnalyzer()

    impact = analyzer.analyze(
        incident
    )

    assert impact.blocked is True

    assert impact.spillback_risk >= 0.70


def test_flooding_reduces_capacity():

    manager = IncidentManager()

    incident = create_incident(
        manager,
        IncidentType.FLOODING,
        severity=0.8,
    )

    analyzer = IncidentImpactAnalyzer()

    impact = analyzer.analyze(
        incident
    )

    assert impact.blocked is True

    assert (
        impact.capacity_reduction
        >= 0.50
    )


def test_stopped_vehicle_is_not_full_blockage():

    manager = IncidentManager()

    incident = create_incident(
        manager,
        IncidentType.STOPPED_VEHICLE,
        severity=0.5,
    )

    analyzer = IncidentImpactAnalyzer()

    impact = analyzer.analyze(
        incident
    )

    assert impact.blocked is False

    assert (
        impact.capacity_reduction > 0.0
    )


def test_significant_incident():

    manager = IncidentManager()

    incident = create_incident(
        manager,
        IncidentType.ACCIDENT,
        severity=0.9,
    )

    analyzer = IncidentImpactAnalyzer()

    impact = analyzer.analyze(
        incident
    )

    assert impact.is_significant() is True


def test_dynamic_topology_is_not_hardcoded():

    manager = IncidentManager()

    incident = manager.register(
        incident_type=IncidentType.ROAD_BLOCKAGE,
        timestamp=20.0,
        confidence=0.9,
        severity=0.8,
        camera_id="CAM_X",
        junction_id="J_CUSTOM_17",
    )

    analyzer = IncidentImpactAnalyzer()

    impact = analyzer.analyze(
        incident,
        affected_edges=[
            "custom_edge_17"
        ],
        affected_lanes=[
            "custom_lane_4"
        ],
        affected_movements=[
            "custom_movement_9"
        ],
    )

    assert impact.junction_id == (
        "J_CUSTOM_17"
    )

    assert impact.affected_edges == (
        "custom_edge_17",
    )

    assert impact.affected_movements == (
        "custom_movement_9",
    )
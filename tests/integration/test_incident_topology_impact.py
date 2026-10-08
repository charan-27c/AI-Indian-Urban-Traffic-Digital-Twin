from src.incidents.topology_impact import (
    IncidentTopologyImpactManager,
)


def test_lane_can_be_blocked():

    manager = (
        IncidentTopologyImpactManager()
    )

    impact = manager.apply(
        incident_id="INC-000001",
        blocked_lanes=["lane_A"],
        junction_id="J_DYNAMIC",
    )

    assert impact.incident_id == (
        "INC-000001"
    )

    assert manager.is_lane_blocked(
        "lane_A"
    )

    assert not manager.is_lane_available(
        "lane_A"
    )


def test_unaffected_lane_remains_available():

    manager = (
        IncidentTopologyImpactManager()
    )

    manager.apply(
        incident_id="INC-000001",
        blocked_lanes=["lane_A"],
    )

    assert manager.is_lane_available(
        "lane_B"
    )


def test_movement_can_be_blocked():

    manager = (
        IncidentTopologyImpactManager()
    )

    manager.apply(
        incident_id="INC-000001",
        blocked_movements=[
            "movement_A"
        ],
    )

    assert manager.is_movement_blocked(
        "movement_A"
    )

    assert not manager.is_movement_available(
        "movement_A"
    )


def test_remove_restores_lane():

    manager = (
        IncidentTopologyImpactManager()
    )

    manager.apply(
        incident_id="INC-000001",
        blocked_lanes=["lane_A"],
    )

    assert manager.is_lane_blocked(
        "lane_A"
    )

    manager.remove(
        "INC-000001"
    )

    assert manager.is_lane_available(
        "lane_A"
    )


def test_remove_restores_movement():

    manager = (
        IncidentTopologyImpactManager()
    )

    manager.apply(
        incident_id="INC-000001",
        blocked_movements=[
            "movement_A"
        ],
    )

    manager.remove(
        "INC-000001"
    )

    assert manager.is_movement_available(
        "movement_A"
    )


def test_two_incidents_can_share_lane():

    manager = (
        IncidentTopologyImpactManager()
    )

    manager.apply(
        incident_id="INC-000001",
        blocked_lanes=["lane_A"],
    )

    manager.apply(
        incident_id="INC-000002",
        blocked_lanes=["lane_A"],
    )

    manager.remove(
        "INC-000001"
    )

    # Still blocked because INC-000002
    # affects the same lane.
    assert manager.is_lane_blocked(
        "lane_A"
    )

    manager.remove(
        "INC-000002"
    )

    assert manager.is_lane_available(
        "lane_A"
    )


def test_dynamic_ids_are_supported():

    manager = (
        IncidentTopologyImpactManager()
    )

    manager.apply(
        incident_id="INC-009999",
        blocked_lanes=[
            "lane_custom_17"
        ],
        blocked_movements=[
            "movement_custom_42"
        ],
        junction_id="J_CUSTOM_17",
    )

    assert manager.is_lane_blocked(
        "lane_custom_17"
    )

    assert manager.is_movement_blocked(
        "movement_custom_42"
    )

    assert (
        "lane_custom_17"
        in manager.blocked_lanes()
    )

    assert (
        "movement_custom_42"
        in manager.blocked_movements()
    )


def test_reset_clears_all_impacts():

    manager = (
        IncidentTopologyImpactManager()
    )

    manager.apply(
        incident_id="INC-1",
        blocked_lanes=["lane_A"],
        blocked_movements=["movement_A"],
    )

    manager.apply(
        incident_id="INC-2",
        blocked_lanes=["lane_B"],
        blocked_movements=["movement_B"],
    )

    manager.reset()

    assert (
        manager.blocked_lanes()
        == set()
    )

    assert (
        manager.blocked_movements()
        == set()
    )

    assert (
        manager.active_impacts()
        == []
    )
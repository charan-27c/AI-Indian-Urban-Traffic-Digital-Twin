from src.incidents.availability import (
    RuntimeAvailabilityFilter,
)
from src.incidents.topology_impact import (
    IncidentTopologyImpactManager,
)


def test_only_blocked_lane_is_filtered():

    manager = (
        IncidentTopologyImpactManager()
    )

    manager.apply(
        incident_id="INC-001",
        blocked_lanes=["lane_2"],
    )

    availability = RuntimeAvailabilityFilter(
        manager
    )

    result = availability.available_lanes(
        [
            "lane_1",
            "lane_2",
            "lane_3",
        ]
    )

    assert result == [
        "lane_1",
        "lane_3",
    ]


def test_only_blocked_movement_is_filtered():

    manager = (
        IncidentTopologyImpactManager()
    )

    manager.apply(
        incident_id="INC-001",
        blocked_movements=[
            "movement_2"
        ],
    )

    availability = RuntimeAvailabilityFilter(
        manager
    )

    result = availability.available_movements(
        [
            "movement_1",
            "movement_2",
            "movement_3",
        ]
    )

    assert result == [
        "movement_1",
        "movement_3",
    ]


def test_multiple_incidents_are_combined():

    manager = (
        IncidentTopologyImpactManager()
    )

    manager.apply(
        incident_id="INC-001",
        blocked_lanes=["lane_1"],
        blocked_movements=["movement_1"],
    )

    manager.apply(
        incident_id="INC-002",
        blocked_lanes=["lane_3"],
        blocked_movements=["movement_3"],
    )

    availability = RuntimeAvailabilityFilter(
        manager
    )

    assert availability.available_lanes(
        [
            "lane_1",
            "lane_2",
            "lane_3",
        ]
    ) == ["lane_2"]

    assert availability.available_movements(
        [
            "movement_1",
            "movement_2",
            "movement_3",
        ]
    ) == ["movement_2"]


def test_resolved_incident_restores_resource():

    manager = (
        IncidentTopologyImpactManager()
    )

    manager.apply(
        incident_id="INC-001",
        blocked_movements=[
            "movement_1"
        ],
    )

    availability = RuntimeAvailabilityFilter(
        manager
    )

    assert availability.available_movements(
        ["movement_1"]
    ) == []

    manager.remove(
        "INC-001"
    )

    assert availability.available_movements(
        ["movement_1"]
    ) == ["movement_1"]


def test_unaffected_resources_remain_available():

    manager = (
        IncidentTopologyImpactManager()
    )

    manager.apply(
        incident_id="INC-001",
        blocked_lanes=["lane_A"],
        blocked_movements=["movement_A"],
    )

    availability = RuntimeAvailabilityFilter(
        manager
    )

    assert availability.available_lanes(
        ["lane_B"]
    ) == ["lane_B"]

    assert availability.available_movements(
        ["movement_B"]
    ) == ["movement_B"]
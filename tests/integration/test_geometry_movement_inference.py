from pathlib import Path

from src.perception.junction.sumo_topology import (
    SumoTopology,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

NETWORK_PATH = (
    PROJECT_ROOT
    / "simulation"
    / "networks"
    / "ai_traffic.net.xml"
)


def test_sumo_lane_geometry_is_discovered():

    loader = SumoTopology(
        str(NETWORK_PATH)
    )

    registry = (
        loader.geometry_registry
    )

    assert registry.count() > 0

    for lane_id, geometry in (
        registry.get_all().items()
    ):

        assert lane_id

        assert geometry.edge_id

        assert isinstance(
            geometry.shape,
            list,
        )
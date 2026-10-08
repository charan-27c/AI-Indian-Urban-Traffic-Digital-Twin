from pathlib import Path
import sys

import pytest


# =========================================================
# PROJECT ROOT
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.perception.junction.sumo_topology import (
    SumoTopology,
)


# =========================================================
# NETWORK DISCOVERY
# =========================================================

def find_sumo_network() -> Path:
    """
    Find a SUMO .net.xml file inside the project.

    No fixed network directory is assumed.
    """

    candidates = list(
        PROJECT_ROOT.rglob("*.net.xml")
    )

    # Ignore generated/cache locations if any.
    candidates = [
        path
        for path in candidates
        if ".git" not in path.parts
        and "__pycache__" not in path.parts
    ]

    if not candidates:
        pytest.skip(
            "No SUMO .net.xml network found "
            "inside the project."
        )

    # Prefer the existing AI traffic network if present.
    preferred = [
        path
        for path in candidates
        if path.name == "ai_traffic.net.xml"
    ]

    if preferred:
        return preferred[0]

    return candidates[0]


# =========================================================
# TEST 1
# =========================================================

def test_sumo_network_can_be_loaded():

    network_path = find_sumo_network()

    topology_loader = SumoTopology(
        str(network_path)
    )

    assert topology_loader.net is not None


# =========================================================
# TEST 2
# =========================================================

def test_sumo_discovers_junctions_dynamically():

    network_path = find_sumo_network()

    topology_loader = SumoTopology(
        str(network_path)
    )

    junctions = (
        topology_loader
        .discover_all_junctions()
    )

    assert isinstance(
        junctions,
        dict,
    )

    assert len(junctions) > 0


# =========================================================
# TEST 3
# =========================================================

def test_junction_topology_is_dynamic():

    network_path = find_sumo_network()

    topology_loader = SumoTopology(
        str(network_path)
    )

    junctions = (
        topology_loader
        .discover_all_junctions()
    )

    for junction_id, topology in (
        junctions.items()
    ):

        # Junction ID comes from SUMO.
        assert (
            topology.junction_id
            == junction_id
        )

        # Counts come from discovered topology.
        assert (
            topology.approach_count()
            >= 0
        )

        assert (
            topology.lane_count()
            >= 0
        )

        assert (
            topology.connection_count()
            >= 0
        )


# =========================================================
# TEST 4
# =========================================================

def test_discovered_lanes_belong_to_valid_approaches():

    network_path = find_sumo_network()

    topology_loader = SumoTopology(
        str(network_path)
    )

    junctions = (
        topology_loader
        .discover_all_junctions()
    )

    for topology in junctions.values():

        approach_ids = set(
            topology.approaches.keys()
        )

        for approach in (
            topology.get_approaches()
        ):

            assert (
                approach.approach_id
                in approach_ids
            )

            for lane in (
                approach.get_lanes()
            ):

                assert (
                    lane.approach_id
                    == approach.approach_id
                )

                assert lane.edge_id != ""
                assert lane.lane_id != ""


# =========================================================
# TEST 5
# =========================================================

def test_connections_reference_real_lanes():

    network_path = find_sumo_network()

    topology_loader = SumoTopology(
        str(network_path)
    )

    junctions = (
        topology_loader
        .discover_all_junctions()
    )

    for topology in junctions.values():

        lane_ids = set()

        for approach in (
            topology.get_approaches()
        ):

            for lane in (
                approach.get_lanes()
            ):

                lane_ids.add(
                    lane.lane_id
                )

        for connection in (
            topology.get_connections()
        ):

            assert (
                connection.from_lane_id
                in lane_ids
            )

            assert (
                connection.to_lane_id
                not in {"", None}
            )

            assert (
                connection.from_edge_id
                != ""
            )

            assert (
                connection.to_edge_id
                != ""
            )
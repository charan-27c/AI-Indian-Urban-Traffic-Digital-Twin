from dataclasses import dataclass

from src.signal_control.sumo_movement_mapper import (
    SUMOMovementMapper,
)


@dataclass
class FakeMovement:
    movement_id: str
    source_lanes: tuple[str, ...]
    destination_lanes: tuple[str, ...]


@dataclass
class FakeJunction:
    junction_id: str
    movements: tuple[FakeMovement, ...]


@dataclass
class FakeTopology:
    junctions: tuple[FakeJunction, ...]


class FakeTrafficLight:

    def __init__(self):
        self.controlled_links = {
            "tls_alpha": [
                [
                    ("lane_in_1", "lane_out_1"),
                ],
                [
                    ("lane_in_2", "lane_out_2"),
                ],
                [
                    ("lane_in_3", "lane_out_3"),
                ],
                [
                    ("lane_in_4", "lane_out_4"),
                ],
            ],
        }

    def getIDList(self):
        return list(
            self.controlled_links.keys()
        )

    def getControlledLinks(self, tls_id):
        return self.controlled_links[tls_id]


class FakeTraCI:

    def __init__(self):
        self.trafficlight = (
            FakeTrafficLight()
        )


def build_topology():

    movement_1 = FakeMovement(
        movement_id="movement_1",
        source_lanes=("lane_in_1",),
        destination_lanes=("lane_out_1",),
    )

    movement_2 = FakeMovement(
        movement_id="movement_2",
        source_lanes=("lane_in_2",),
        destination_lanes=("lane_out_2",),
    )

    junction = FakeJunction(
        junction_id="dynamic_junction_a",
        movements=(
            movement_1,
            movement_2,
        ),
    )

    return FakeTopology(
        junctions=(junction,)
    )


def test_discovers_controlled_links():

    mapper = SUMOMovementMapper()

    links = mapper.discover_controlled_links(
        FakeTraCI()
    )

    assert "tls_alpha" in links

    assert len(
        links["tls_alpha"]
    ) == 4


def test_controlled_link_signal_indices():

    mapper = SUMOMovementMapper()

    mapper.discover_controlled_links(
        FakeTraCI()
    )

    links = mapper.controlled_links(
        "tls_alpha"
    )

    assert [
        link.signal_index
        for link in links
    ] == [0, 1, 2, 3]


def test_maps_movement_to_signal_index():

    mapper = SUMOMovementMapper(
        topology=build_topology()
    )

    mapper.discover_controlled_links(
        FakeTraCI()
    )

    mappings = mapper.build_mappings()

    assert len(mappings) == 2

    movement_1 = mapper.get_mapping(
        "movement_1"
    )

    assert movement_1 is not None

    assert movement_1.tls_id == "tls_alpha"

    assert movement_1.signal_indices == (0,)


def test_maps_second_movement():

    mapper = SUMOMovementMapper(
        topology=build_topology()
    )

    mapper.discover_controlled_links(
        FakeTraCI()
    )

    mapper.build_mappings()

    movement_2 = mapper.get_mapping(
        "movement_2"
    )

    assert movement_2 is not None

    assert movement_2.signal_indices == (1,)


def test_tls_lookup():

    mapper = SUMOMovementMapper(
        topology=build_topology()
    )

    mapper.discover_controlled_links(
        FakeTraCI()
    )

    mapper.build_mappings()

    assert (
        mapper.tls_for_movement(
            "movement_1"
        )
        == "tls_alpha"
    )


def test_signal_index_lookup():

    mapper = SUMOMovementMapper(
        topology=build_topology()
    )

    mapper.discover_controlled_links(
        FakeTraCI()
    )

    mapper.build_mappings()

    assert (
        mapper.signal_indices_for_movement(
            "movement_2"
        )
        == (1,)
    )


def test_unknown_movement_returns_empty():

    mapper = SUMOMovementMapper(
        topology=build_topology()
    )

    mapper.discover_controlled_links(
        FakeTraCI()
    )

    mapper.build_mappings()

    assert (
        mapper.signal_indices_for_movement(
            "unknown"
        )
        == ()
    )

    assert (
        mapper.tls_for_movement(
            "unknown"
        )
        is None
    )


def test_unmatched_movement_is_not_mapped():

    topology = FakeTopology(
        junctions=(
            FakeJunction(
                junction_id="another_junction",
                movements=(
                    FakeMovement(
                        movement_id="no_match",
                        source_lanes=(
                            "nonexistent_lane",
                        ),
                        destination_lanes=(
                            "nonexistent_out",
                        ),
                    ),
                ),
            ),
        )
    )

    mapper = SUMOMovementMapper(
        topology=topology
    )

    mapper.discover_controlled_links(
        FakeTraCI()
    )

    mappings = mapper.build_mappings()

    assert mappings == ()

    assert (
        mapper.get_mapping("no_match")
        is None
    )
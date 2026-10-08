from pathlib import Path

from src.digital_twin.network.network_builder import (
    NetworkBuilder,
)


PROJECT_ROOT = Path(
    __file__
).resolve().parents[2]


NETWORK_CONFIG = (
    PROJECT_ROOT
    / "config"
    / "network.yaml"
)


def test_network_definition_loads():

    builder = NetworkBuilder(
        str(NETWORK_CONFIG)
    )

    assert builder.validate() is True


def test_network_discovers_dynamic_junctions():

    builder = NetworkBuilder(
        str(NETWORK_CONFIG)
    )

    junctions = builder.get_junctions()

    assert len(junctions) == 3


def test_junctions_have_different_approach_counts():

    builder = NetworkBuilder(
        str(NETWORK_CONFIG)
    )

    assert (
        builder.get_approach_count("J1")
        == 4
    )

    assert (
        builder.get_approach_count("J2")
        == 3
    )

    assert (
        builder.get_approach_count("J3")
        == 2
    )


def test_network_does_not_require_four_approaches():

    builder = NetworkBuilder(
        str(NETWORK_CONFIG)
    )

    assert (
        builder.get_approach_count("J2")
        != 4
    )

    assert (
        builder.get_approach_count("J3")
        != 4
    )
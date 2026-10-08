from src.perception.junction.geometry_movement_inference import (
    GeometryMovementInference,
    LEFT,
    RIGHT,
    STRAIGHT,
    U_TURN,
)

from src.perception.junction.lane_geometry import (
    LaneGeometry,
)


def make_lane(
    lane_id,
    shape,
):
    return LaneGeometry(
        lane_id=lane_id,
        edge_id=f"{lane_id}_EDGE",
        shape=shape,
    )


def test_straight_movement():

    analyzer = GeometryMovementInference()

    incoming = make_lane(
        "IN",
        [
            (0.0, 0.0),
            (10.0, 0.0),
        ],
    )

    outgoing = make_lane(
        "OUT",
        [
            (10.0, 0.0),
            (20.0, 0.0),
        ],
    )

    result = analyzer.infer(
        incoming,
        outgoing,
    )

    assert result == STRAIGHT


def test_left_movement():

    analyzer = GeometryMovementInference()

    incoming = make_lane(
        "IN",
        [
            (0.0, 0.0),
            (10.0, 0.0),
        ],
    )

    outgoing = make_lane(
        "OUT",
        [
            (10.0, 0.0),
            (10.0, 10.0),
        ],
    )

    result = analyzer.infer(
        incoming,
        outgoing,
    )

    assert result == LEFT


def test_right_movement():

    analyzer = GeometryMovementInference()

    incoming = make_lane(
        "IN",
        [
            (0.0, 0.0),
            (10.0, 0.0),
        ],
    )

    outgoing = make_lane(
        "OUT",
        [
            (10.0, 0.0),
            (10.0, -10.0),
        ],
    )

    result = analyzer.infer(
        incoming,
        outgoing,
    )

    assert result == RIGHT


def test_u_turn():

    analyzer = GeometryMovementInference()

    incoming = make_lane(
        "IN",
        [
            (0.0, 0.0),
            (10.0, 0.0),
        ],
    )

    outgoing = make_lane(
        "OUT",
        [
            (10.0, 0.0),
            (0.0, 0.0),
        ],
    )

    result = analyzer.infer(
        incoming,
        outgoing,
    )

    assert result == U_TURN
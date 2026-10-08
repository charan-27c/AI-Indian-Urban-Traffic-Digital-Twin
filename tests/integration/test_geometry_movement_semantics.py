from src.perception.junction.movement_geometry import (
    MovementGeometryInferencer,
)


def test_straight_movement():

    inferencer = MovementGeometryInferencer()

    result = inferencer.infer(
        movement_id="M1",
        from_lane_id="A_IN",
        to_lane_id="B_OUT",
        incoming_shape=[
            (0.0, 0.0),
            (10.0, 0.0),
        ],
        outgoing_shape=[
            (10.0, 0.0),
            (20.0, 0.0),
        ],
    )

    assert result.movement_type == "straight"
    assert result.confidence > 0


def test_left_movement():

    inferencer = MovementGeometryInferencer()

    result = inferencer.infer(
        movement_id="M2",
        from_lane_id="A_IN",
        to_lane_id="B_OUT",
        incoming_shape=[
            (0.0, 0.0),
            (10.0, 0.0),
        ],
        outgoing_shape=[
            (10.0, 0.0),
            (10.0, 10.0),
        ],
    )

    assert result.movement_type == "left"


def test_right_movement():

    inferencer = MovementGeometryInferencer()

    result = inferencer.infer(
        movement_id="M3",
        from_lane_id="A_IN",
        to_lane_id="B_OUT",
        incoming_shape=[
            (0.0, 0.0),
            (10.0, 0.0),
        ],
        outgoing_shape=[
            (10.0, 0.0),
            (10.0, -10.0),
        ],
    )

    assert result.movement_type == "right"


def test_uturn_movement():

    inferencer = MovementGeometryInferencer()

    result = inferencer.infer(
        movement_id="M4",
        from_lane_id="A_IN",
        to_lane_id="A_OUT",
        incoming_shape=[
            (0.0, 0.0),
            (10.0, 0.0),
        ],
        outgoing_shape=[
            (10.0, 0.0),
            (0.0, 0.0),
        ],
    )

    assert result.movement_type == "u_turn"
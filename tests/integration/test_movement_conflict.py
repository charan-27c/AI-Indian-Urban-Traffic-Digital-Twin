from src.perception.junction.movement_conflict import (
    MovementConflict,
    MovementConflictMatrix,
    MovementConflictAnalyzer,
)


class DummyMovement:

    def __init__(
        self,
        movement_id,
        from_lane_id,
        to_lane_id,
        from_approach_id,
    ):

        self.movement_id = movement_id
        self.from_lane_id = from_lane_id
        self.to_lane_id = to_lane_id
        self.from_approach_id = (
            from_approach_id
        )


def test_conflict_matrix_registration():

    matrix = MovementConflictMatrix()

    matrix.register(
        MovementConflict(
            movement_a="M1",
            movement_b="M2",
            relationship="conflict",
            confidence=1.0,
        )
    )

    assert matrix.count() == 1

    assert (
        matrix.get(
            "M1",
            "M2",
        ).relationship
        == "conflict"
    )

    # Pair ordering must not matter.

    assert (
        matrix.get(
            "M2",
            "M1",
        ).relationship
        == "conflict"
    )


def test_same_lane_movements_conflict():

    analyzer = MovementConflictAnalyzer()

    m1 = DummyMovement(
        movement_id="M1",
        from_lane_id="LANE_A",
        to_lane_id="LANE_X",
        from_approach_id="APP_A",
    )

    m2 = DummyMovement(
        movement_id="M2",
        from_lane_id="LANE_A",
        to_lane_id="LANE_Y",
        from_approach_id="APP_A",
    )

    matrix = analyzer.analyze(
        [m1, m2]
    )

    assert (
        matrix.get(
            "M1",
            "M2",
        ).relationship
        == "conflict"
    )


def test_same_approach_movements_conflict():

    analyzer = MovementConflictAnalyzer()

    m1 = DummyMovement(
        movement_id="M1",
        from_lane_id="LANE_A",
        to_lane_id="LANE_X",
        from_approach_id="APP_A",
    )

    m2 = DummyMovement(
        movement_id="M2",
        from_lane_id="LANE_B",
        to_lane_id="LANE_Y",
        from_approach_id="APP_A",
    )

    matrix = analyzer.analyze(
        [m1, m2]
    )

    assert (
        matrix.get(
            "M1",
            "M2",
        ).relationship
        == "conflict"
    )


def test_different_movements_can_be_unknown():

    analyzer = MovementConflictAnalyzer()

    m1 = DummyMovement(
        movement_id="M1",
        from_lane_id="LANE_A",
        to_lane_id="LANE_X",
        from_approach_id="APP_A",
    )

    m2 = DummyMovement(
        movement_id="M2",
        from_lane_id="LANE_B",
        to_lane_id="LANE_Y",
        from_approach_id="APP_B",
    )

    matrix = analyzer.analyze(
        [m1, m2]
    )

    assert (
        matrix.get(
            "M1",
            "M2",
        ).relationship
        == "unknown"
    )


def test_conflict_lookup():

    matrix = MovementConflictMatrix()

    matrix.register(
        MovementConflict(
            "M1",
            "M2",
            "conflict",
            1.0,
        )
    )

    matrix.register(
        MovementConflict(
            "M1",
            "M3",
            "compatible",
            1.0,
        )
    )

    assert matrix.conflicts_with(
        "M1"
    ) == ["M2"]

    assert matrix.compatible_with(
        "M1"
    ) == ["M3"]
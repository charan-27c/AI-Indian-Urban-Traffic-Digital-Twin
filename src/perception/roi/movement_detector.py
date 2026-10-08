from collections import defaultdict


class MovementDetector:

    def __init__(self):

        self.vehicle_entries = {}

        self.vehicle_exits = {}

        self.vehicle_movements = {}

    def update(
        self,
        vehicle_id,
        center,
        roi_manager
    ):

        entry = self._detect_entry(
            center,
            roi_manager
        )

        exit_zone = self._detect_exit(
            center,
            roi_manager
        )

        if (
            entry is not None
            and vehicle_id not in self.vehicle_entries
        ):

            self.vehicle_entries[
                vehicle_id
            ] = entry

        if (
            exit_zone is not None
            and vehicle_id in self.vehicle_entries
        ):

            entry = self.vehicle_entries[
                vehicle_id
            ]

            movement = self._create_movement(
                entry,
                exit_zone
            )

            self.vehicle_exits[
                vehicle_id
            ] = exit_zone

            self.vehicle_movements[
                vehicle_id
            ] = movement

            return movement

        return None

    def _detect_entry(
        self,
        center,
        roi_manager
    ):

        for direction in [
            "north",
            "south",
            "east",
            "west"
        ]:

            zone = f"entry_{direction}"

            if roi_manager.point_inside(
                center,
                zone
            ):
                return direction

        return None

    def _detect_exit(
        self,
        center,
        roi_manager
    ):

        for direction in [
            "north",
            "south",
            "east",
            "west"
        ]:

            zone = f"entry_{direction}"

            if roi_manager.point_inside(
                center,
                zone
            ):
                return direction

        return None

    def _create_movement(
        self,
        entry,
        exit_zone
    ):

        if entry == exit_zone:

            return None

        return (
            f"{entry.upper()}_TO_"
            f"{exit_zone.upper()}"
        )
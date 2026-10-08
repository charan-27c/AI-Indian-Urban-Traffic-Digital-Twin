from dataclasses import dataclass


@dataclass(frozen=True)
class AnomalyFeatures:

    vehicle_count: float = 0.0
    stopped_vehicle_count: float = 0.0

    average_speed: float = 0.0
    queue_length: float = 0.0

    density: float = 0.0
    occupancy: float = 0.0

    speed_drop: float = 0.0
    queue_growth: float = 0.0
    occupancy_growth: float = 0.0

    stationary_duration: float = 0.0

    flood_probability: float = 0.0
    wrong_way_count: float = 0.0
    pedestrian_intrusion_count: float = 0.0

class AnomalyFeatureEngine:

    def build(
        self,
        current_state,
        previous_state=None,
    ) -> AnomalyFeatures:

        if previous_state is None:

            return AnomalyFeatures(
                vehicle_count=current_state.vehicle_count,
                queue_length=current_state.queue_length,
                average_speed=current_state.average_speed,
                density=current_state.density,
                occupancy=current_state.occupancy,
            )

        speed_drop = max(
            0.0,
            previous_state.average_speed
            - current_state.average_speed,
        )

        queue_growth = max(
            0.0,
            current_state.queue_length
            - previous_state.queue_length,
        )

        occupancy_growth = max(
            0.0,
            current_state.occupancy
            - previous_state.occupancy,
        )

        return AnomalyFeatures(
            vehicle_count=current_state.vehicle_count,
            queue_length=current_state.queue_length,
            average_speed=current_state.average_speed,
            density=current_state.density,
            occupancy=current_state.occupancy,
            speed_drop=speed_drop,
            queue_growth=queue_growth,
            occupancy_growth=occupancy_growth,
        )
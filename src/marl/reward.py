from __future__ import annotations

from .models import JunctionObservation, MARLReward


class MARLRewardCalculator:
    """
    Multi-objective reward for traffic signal agents.

    Positive:
        throughput
        emergency service
        coordination

    Negative:
        queue
        waiting
        unsafe actions
    """

    def __init__(
        self,
        throughput_weight: float = 1.0,
        queue_weight: float = 1.0,
        waiting_weight: float = 0.5,
        emergency_weight: float = 5.0,
        coordination_weight: float = 1.0,
        safety_weight: float = 10.0,
    ) -> None:
        self.throughput_weight = throughput_weight
        self.queue_weight = queue_weight
        self.waiting_weight = waiting_weight
        self.emergency_weight = emergency_weight
        self.coordination_weight = coordination_weight
        self.safety_weight = safety_weight

    def calculate(
        self,
        before: JunctionObservation,
        after: JunctionObservation,
        *,
        emergency_served: bool = False,
        coordinated: bool = False,
        unsafe_action: bool = False,
    ) -> MARLReward:

        throughput = (
            before.vehicle_count
            - after.vehicle_count
        )

        queue_change = (
            before.queue_length
            - after.queue_length
        )

        waiting_change = (
            before.waiting_time
            - after.waiting_time
        )

        throughput_reward = (
            self.throughput_weight
            * throughput
        )

        queue_penalty = (
            self.queue_weight
            * queue_change
        )

        waiting_penalty = (
            self.waiting_weight
            * waiting_change
        )

        emergency_reward = (
            self.emergency_weight
            if emergency_served
            else 0.0
        )

        coordination_reward = (
            self.coordination_weight
            if coordinated
            else 0.0
        )

        safety_penalty = (
            -self.safety_weight
            if unsafe_action
            else 0.0
        )

        return MARLReward(
            junction_id=after.junction_id,
            throughput_reward=throughput_reward,
            queue_penalty=queue_penalty,
            waiting_penalty=waiting_penalty,
            emergency_reward=emergency_reward,
            coordination_reward=coordination_reward,
            safety_penalty=safety_penalty,
        )
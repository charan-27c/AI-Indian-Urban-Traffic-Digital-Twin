from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional, Sequence

from .models import JunctionAction, JunctionObservation
from .observation import MARLObservationBuilder
from .reward import MARLRewardCalculator


@dataclass(frozen=True)
class MARLStepResult:
    observations: Dict[str, JunctionObservation]
    rewards: Dict[str, float]
    actions: Dict[str, JunctionAction]
    terminated: bool
    truncated: bool
    timestamp: float


class MARLEnvironment:
    """
    Dynamic multi-agent traffic-control environment.

    One agent represents one discovered junction.

    Junction IDs, movement IDs and agent count are completely dynamic.
    No J1/J2/J3/J4 assumptions are made.

    The environment does NOT directly change signal states.
    Actions are passed through an optional action executor/safety layer.
    """

    def __init__(
        self,
        *,
        observation_builder: Optional[
            MARLObservationBuilder
        ] = None,
        reward_calculator: Optional[
            MARLRewardCalculator
        ] = None,
        action_executor=None,
        max_episode_steps: int = 3600,
    ) -> None:

        if max_episode_steps <= 0:
            raise ValueError(
                "max_episode_steps must be greater than zero."
            )

        self.observation_builder = (
            observation_builder
            or MARLObservationBuilder()
        )

        self.reward_calculator = (
            reward_calculator
            or MARLRewardCalculator()
        )

        self.action_executor = action_executor

        self.max_episode_steps = int(
            max_episode_steps
        )

        self.agents: list[str] = []
        self._junctions: Dict[str, object] = {}

        self._previous_observations: Dict[
            str,
            JunctionObservation,
        ] = {}

        self._last_actions: Dict[
            str,
            JunctionAction,
        ] = {}

        self._step_count = 0
        self._timestamp = 0.0

    # =========================================================
    # RESET
    # =========================================================

    def reset(
        self,
        junctions: Iterable,
        *,
        timestamp: float = 0.0,
    ) -> Dict[str, JunctionObservation]:

        self._step_count = 0
        self._timestamp = float(timestamp)

        self._junctions.clear()
        self._previous_observations.clear()
        self._last_actions.clear()

        observations = self._build_observations(
            junctions
        )

        self._previous_observations = dict(
            observations
        )

        return observations

    # =========================================================
    # DYNAMIC AGENT DISCOVERY
    # =========================================================

    def discover_agents(
        self,
        junctions: Iterable,
    ) -> list[str]:

        agents = []

        for junction in junctions:

            junction_id = str(
                getattr(
                    junction,
                    "junction_id",
                    "",
                )
            )

            if not junction_id:
                continue

            agents.append(junction_id)

        self.agents = list(
            dict.fromkeys(agents)
        )

        return list(self.agents)

    # =========================================================
    # OBSERVATION
    # =========================================================

    def _build_observations(
        self,
        junctions: Iterable,
    ) -> Dict[str, JunctionObservation]:

        junction_list = list(junctions)

        self.discover_agents(
            junction_list
        )

        self._junctions = {
            str(
                getattr(
                    junction,
                    "junction_id",
                )
            ): junction
            for junction in junction_list
            if getattr(
                junction,
                "junction_id",
                None,
            )
        }

        observations = {}

        total_pressure = sum(
            max(
                0.0,
                float(
                    getattr(
                        junction,
                        "pressure",
                        0.0,
                    )
                ),
            )
            for junction in junction_list
        )

        for junction in junction_list:

            junction_id = str(
                getattr(
                    junction,
                    "junction_id",
                    "",
                )
            )

            if not junction_id:
                continue

            local_pressure = max(
                0.0,
                float(
                    getattr(
                        junction,
                        "pressure",
                        0.0,
                    )
                ),
            )

            neighbor_pressure = max(
                0.0,
                total_pressure - local_pressure,
            )

            observation = (
                self.observation_builder.build(
                    junction,
                    neighbor_pressure=(
                        neighbor_pressure
                    ),
                )
            )

            observations[junction_id] = (
                observation
            )

        return observations

    # =========================================================
    # ACTION MASK
    # =========================================================

    @staticmethod
    def _movement_id(movement) -> str:
        return str(
            getattr(
                movement,
                "movement_id",
                "",
            )
        )

    def get_action_mask(
        self,
        junction,
        *,
        blocked_movements: Optional[
            Mapping[str, bool]
        ] = None,
        available_movements: Optional[
            Mapping[str, bool]
        ] = None,
    ) -> Dict[str, bool]:

        blocked_movements = (
            blocked_movements or {}
        )

        available_movements = (
            available_movements or {}
        )

        movements = getattr(
            junction,
            "movements",
            (),
        )

        mask = {}

        for movement in movements:

            movement_id = (
                self._movement_id(
                    movement
                )
            )

            if not movement_id:
                continue

            blocked = bool(
                blocked_movements.get(
                    movement_id,
                    False,
                )
            )

            available = bool(
                available_movements.get(
                    movement_id,
                    True,
                )
            )

            mask[movement_id] = (
                available and not blocked
            )

        return mask

    # =========================================================
    # ACTION VALIDATION
    # =========================================================

    def validate_action(
        self,
        action: JunctionAction,
        *,
        action_mask: Mapping[str, bool],
    ) -> JunctionAction:

        movement_id = action.movement_id

        if movement_id is None:
            return action

        if not action_mask.get(
            movement_id,
            False,
        ):
            # Unsafe/invalid learned action.
            # Do NOT allow it through.
            return JunctionAction(
                junction_id=action.junction_id,
                movement_id=None,
                green_extension=0.0,
                emergency_priority=False,
                request_all_red=True,
            )

        return action

    # =========================================================
    # STEP
    # =========================================================

    def step(
        self,
        actions: Mapping[
            str,
            JunctionAction,
        ],
        junctions: Optional[
            Iterable
        ] = None,
        *,
        timestamp: Optional[float] = None,
        blocked_movements: Optional[
            Mapping[str, bool]
        ] = None,
        available_movements: Optional[
            Mapping[str, bool]
        ] = None,
    ) -> MARLStepResult:

        if junctions is None:
            junction_list = list(
                self._junctions.values()
            )
        else:
            junction_list = list(junctions)

        if not junction_list:
            raise ValueError(
                "No junctions are available."
            )

        if timestamp is None:
            self._timestamp += 1.0
        else:
            self._timestamp = float(
                timestamp
            )

        validated_actions = {}

        for junction in junction_list:

            junction_id = str(
                getattr(
                    junction,
                    "junction_id",
                    "",
                )
            )

            if not junction_id:
                continue

            action = actions.get(
                junction_id,
                JunctionAction(
                    junction_id=junction_id
                ),
            )

            mask = self.get_action_mask(
                junction,
                blocked_movements=(
                    blocked_movements
                ),
                available_movements=(
                    available_movements
                ),
            )

            validated = self.validate_action(
                action,
                action_mask=mask,
            )

            validated_actions[
                junction_id
            ] = validated

        # -----------------------------------------------------
        # Safety-controlled execution
        # -----------------------------------------------------

        if self.action_executor is not None:

            for junction_id, action in (
                validated_actions.items()
            ):
                self.action_executor(
                    junction_id,
                    action,
                )

        new_observations = (
            self._build_observations(
                junction_list
            )
        )

        rewards = {}

        for junction_id, observation in (
            new_observations.items()
        ):

            before = (
                self._previous_observations.get(
                    junction_id,
                    observation,
                )
            )

            action = validated_actions.get(
                junction_id,
                JunctionAction(
                    junction_id=junction_id
                ),
            )

            unsafe = (
                action.movement_id is None
                and action.request_all_red is False
            )

            reward = (
                self.reward_calculator.calculate(
                    before,
                    observation,
                    emergency_served=(
                        action.emergency_priority
                    ),
                    coordinated=False,
                    unsafe_action=unsafe,
                )
            )

            rewards[junction_id] = (
                reward.total
            )

        self._previous_observations = dict(
            new_observations
        )

        self._last_actions = dict(
            validated_actions
        )

        self._step_count += 1

        truncated = (
            self._step_count
            >= self.max_episode_steps
        )

        terminated = False

        return MARLStepResult(
            observations=new_observations,
            rewards=rewards,
            actions=validated_actions,
            terminated=terminated,
            truncated=truncated,
            timestamp=self._timestamp,
        )

    # =========================================================
    # STATE
    # =========================================================

    @property
    def step_count(self) -> int:
        return self._step_count

    @property
    def timestamp(self) -> float:
        return self._timestamp

    @property
    def agent_ids(self) -> tuple[str, ...]:
        return tuple(self.agents)

    def reset_state(self) -> None:
        self.agents.clear()
        self._junctions.clear()
        self._previous_observations.clear()
        self._last_actions.clear()
        self._step_count = 0
        self._timestamp = 0.0
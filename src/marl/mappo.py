from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from torch.distributions import Categorical


@dataclass(frozen=True)
class MAPPOActionOutput:
    movement_index: torch.Tensor
    green_extension: torch.Tensor
    log_probability: torch.Tensor
    entropy: torch.Tensor
    value: torch.Tensor


def clipped_policy_loss(
    new_log_probability: torch.Tensor,
    old_log_probability: torch.Tensor,
    advantages: torch.Tensor,
    clip_epsilon: float = 0.2,
) -> torch.Tensor:
    if clip_epsilon <= 0:
        raise ValueError(
            "clip_epsilon must be positive."
        )

    ratio = torch.exp(
        new_log_probability
        - old_log_probability
    )
    unclipped = ratio * advantages
    clipped = torch.clamp(
        ratio,
        1.0 - clip_epsilon,
        1.0 + clip_epsilon,
    ) * advantages

    return -torch.min(
        unclipped,
        clipped,
    ).mean()


def value_loss(
    predicted_values: torch.Tensor,
    returns: torch.Tensor,
) -> torch.Tensor:
    return (
        predicted_values - returns
    ).pow(2).mean()


def entropy_bonus(
    entropy: torch.Tensor,
) -> torch.Tensor:
    return entropy.mean()


def generalized_advantage_estimation(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> tuple[torch.Tensor, torch.Tensor]:
    advantages = torch.zeros_like(rewards)
    gae = torch.zeros(
        (),
        dtype=rewards.dtype,
        device=rewards.device,
    )

    for index in reversed(range(len(rewards))):
        if index == len(rewards) - 1:
            next_value = values[index]
        else:
            next_value = values[index + 1]

        non_terminal = 1.0 - dones[index]
        delta = (
            rewards[index]
            + gamma * next_value * non_terminal
            - values[index]
        )
        gae = (
            delta
            + gamma
            * gae_lambda
            * non_terminal
            * gae
        )
        advantages[index] = gae

    returns = advantages + values
    return advantages, returns


class SharedActor(nn.Module):
    """
    Shared actor used by every dynamically discovered junction.

    The actor does not know junction names.
    """

    def __init__(
        self,
        observation_dim: int = 15,
        action_dim: int = 16,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()

        if observation_dim <= 0:
            raise ValueError(
                "observation_dim must be positive."
            )

        if action_dim <= 0:
            raise ValueError(
                "action_dim must be positive."
            )

        self.observation_dim = observation_dim
        self.action_dim = action_dim

        self.network = nn.Sequential(
            nn.Linear(
                observation_dim,
                hidden_dim,
            ),
            nn.Tanh(),
            nn.Linear(
                hidden_dim,
                hidden_dim,
            ),
            nn.Tanh(),
        )

        self.movement_head = nn.Linear(
            hidden_dim,
            action_dim,
        )

        self.green_head = nn.Linear(
            hidden_dim,
            1,
        )

    def forward(
        self,
        observations: torch.Tensor,
        action_mask: Optional[
            torch.Tensor
        ] = None,
    ):
        features = self.network(
            observations
        )

        logits = self.movement_head(
            features
        )

        if action_mask is not None:

            if action_mask.shape != logits.shape:
                raise ValueError(
                    "action_mask shape must match "
                    "movement logits shape."
                )

            logits = logits.masked_fill(
                ~action_mask.bool(),
                -1e9,
            )

        green_extension = (
            torch.sigmoid(
                self.green_head(features)
            ).squeeze(-1)
        )

        return logits, green_extension

    def distribution(
        self,
        observations: torch.Tensor,
        action_mask: Optional[
            torch.Tensor
        ] = None,
    ) -> Categorical:

        logits, _ = self.forward(
            observations,
            action_mask,
        )

        return Categorical(
            logits=logits
        )

    def sample(
        self,
        observations: torch.Tensor,
        action_mask: Optional[
            torch.Tensor
        ] = None,
    ) -> MAPPOActionOutput:

        logits, green_extension = (
            self.forward(
                observations,
                action_mask,
            )
        )

        distribution = Categorical(
            logits=logits
        )

        movement_index = (
            distribution.sample()
        )

        log_probability = (
            distribution.log_prob(
                movement_index
            )
        )

        entropy = (
            distribution.entropy()
        )

        return MAPPOActionOutput(
            movement_index=movement_index,
            green_extension=green_extension,
            log_probability=log_probability,
            entropy=entropy,
            value=torch.zeros_like(
                log_probability
            ),
        )


class CentralizedCritic(nn.Module):
    """
    Centralized critic.

    During training it receives the global state composed
    of all junction observations.
    """

    def __init__(
        self,
        observation_dim: int = 15,
        max_agents: int = 16,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()

        if observation_dim <= 0:
            raise ValueError(
                "observation_dim must be positive."
            )

        if max_agents <= 0:
            raise ValueError(
                "max_agents must be positive."
            )

        self.observation_dim = observation_dim
        self.max_agents = max_agents

        input_dim = (
            observation_dim * max_agents
        )

        self.network = nn.Sequential(
            nn.Linear(
                input_dim,
                hidden_dim,
            ),
            nn.Tanh(),
            nn.Linear(
                hidden_dim,
                hidden_dim,
            ),
            nn.Tanh(),
            nn.Linear(
                hidden_dim,
                1,
            ),
        )

    def forward(
        self,
        observations: torch.Tensor,
    ) -> torch.Tensor:

        if observations.ndim == 2:

            observations = (
                observations.unsqueeze(0)
            )

        if observations.ndim != 3:
            raise ValueError(
                "observations must have shape "
                "(batch, agents, features)."
            )

        batch_size, agent_count, feature_count = (
            observations.shape
        )

        if feature_count != self.observation_dim:
            raise ValueError(
                "Unexpected observation dimension."
            )

        if agent_count > self.max_agents:
            raise ValueError(
                "Agent count exceeds max_agents."
            )

        if agent_count < self.max_agents:

            padding = torch.zeros(
                (
                    batch_size,
                    self.max_agents
                    - agent_count,
                    feature_count,
                ),
                dtype=observations.dtype,
                device=observations.device,
            )

            observations = torch.cat(
                [
                    observations,
                    padding,
                ],
                dim=1,
            )

        flattened = observations.reshape(
            batch_size,
            -1,
        )

        return self.network(
            flattened
        ).squeeze(-1)


class MAPPOPolicy(nn.Module):
    """
    MAPPO policy containing:

    - shared decentralized actor
    - centralized critic

    The actor can be used independently during execution.
    """

    def __init__(
        self,
        observation_dim: int = 15,
        action_dim: int = 16,
        max_agents: int = 16,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()

        self.actor = SharedActor(
            observation_dim=observation_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
        )

        self.critic = CentralizedCritic(
            observation_dim=observation_dim,
            max_agents=max_agents,
            hidden_dim=hidden_dim,
        )

    @torch.no_grad()
    def act(
        self,
        observation: torch.Tensor,
        action_mask: Optional[
            torch.Tensor
        ] = None,
        deterministic: bool = False,
    ) -> MAPPOActionOutput:

        if observation.ndim == 1:
            observation = observation.unsqueeze(0)

        logits, green_extension = (
            self.actor(
                observation,
                action_mask,
            )
        )

        distribution = Categorical(
            logits=logits
        )

        if deterministic:
            movement_index = (
                torch.argmax(
                    logits,
                    dim=-1,
                )
            )
        else:
            movement_index = (
                distribution.sample()
            )

        log_probability = (
            distribution.log_prob(
                movement_index
            )
        )

        entropy = (
            distribution.entropy()
        )

        return MAPPOActionOutput(
            movement_index=movement_index,
            green_extension=green_extension,
            log_probability=log_probability,
            entropy=entropy,
            value=torch.zeros_like(
                log_probability
            ),
        )

    def value(
        self,
        global_observations: torch.Tensor,
    ) -> torch.Tensor:

        return self.critic(
            global_observations
        )
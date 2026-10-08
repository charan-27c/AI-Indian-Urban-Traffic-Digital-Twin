from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import torch
from torch import nn
from torch.optim import Adam

from .mappo import (
    MAPPOPolicy,
    clipped_policy_loss,
    entropy_bonus,
    generalized_advantage_estimation,
    value_loss,
)


@dataclass
class RolloutStep:
    """
    One multi-agent transition.

    observations:
        Local observation for every currently known junction agent.
        Shape: [num_agents, observation_dim]

    global_observations:
        Same agent observations supplied to the centralized critic.
        Shape: [num_agents, observation_dim]

    actions:
        Movement action selected by each agent.
        Shape: [num_agents]

    action_masks:
        Valid movement mask for every agent.
        Shape: [num_agents, action_dim]

    log_probabilities:
        Old policy log probability for every selected action.
        Shape: [num_agents]

    values:
        Centralized critic value for the state.
        Shape: [num_agents] or [1]

    rewards:
        Reward received by every agent.
        Shape: [num_agents]

    dones:
        Episode termination flag.
        Shape: [num_agents]
    """

    observations: torch.Tensor
    global_observations: torch.Tensor
    actions: torch.Tensor
    action_masks: torch.Tensor
    log_probabilities: torch.Tensor
    values: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor

    agent_ids: List[str] = field(default_factory=list)


@dataclass
class PPOTrainingConfig:
    learning_rate: float = 3e-4
    critic_learning_rate: float = 3e-4

    gamma: float = 0.99
    gae_lambda: float = 0.95

    clip_epsilon: float = 0.2

    entropy_coefficient: float = 0.01
    value_loss_coefficient: float = 0.5

    ppo_epochs: int = 4
    mini_batch_size: int = 64

    max_grad_norm: float = 0.5

    device: str = "cpu"

    checkpoint_dir: str = "checkpoints/mappo"


@dataclass
class TrainingMetrics:
    policy_loss: float = 0.0
    value_loss: float = 0.0
    entropy: float = 0.0
    total_loss: float = 0.0

    mean_reward: float = 0.0
    episode_reward: float = 0.0

    mean_advantage: float = 0.0
    mean_return: float = 0.0

    actor_gradient_norm: float = 0.0
    critic_gradient_norm: float = 0.0

    update_count: int = 0


class RolloutBuffer:
    """
    Stores multi-agent trajectories for MAPPO.

    The buffer deliberately stores dynamic agent IDs instead of assuming
    J1/J2/J3/J4.
    """

    def __init__(self) -> None:
        self.steps: List[RolloutStep] = []

    def clear(self) -> None:
        self.steps.clear()

    def __len__(self) -> int:
        return len(self.steps)

    def add(
        self,
        observations: torch.Tensor,
        global_observations: torch.Tensor,
        actions: torch.Tensor,
        action_masks: torch.Tensor,
        log_probabilities: torch.Tensor,
        values: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        agent_ids: Optional[Sequence[str]] = None,
    ) -> None:
        self.steps.append(
            RolloutStep(
                observations=observations.detach().cpu(),
                global_observations=global_observations.detach().cpu(),
                actions=actions.detach().cpu(),
                action_masks=action_masks.detach().cpu(),
                log_probabilities=log_probabilities.detach().cpu(),
                values=values.detach().cpu(),
                rewards=rewards.detach().cpu(),
                dones=dones.detach().cpu(),
                agent_ids=list(agent_ids or []),
            )
        )

    def rewards_tensor(self) -> torch.Tensor:
        if not self.steps:
            return torch.empty(0)

        return torch.stack(
            [step.rewards.reshape(-1) for step in self.steps],
            dim=0,
        )

    def values_tensor(self) -> torch.Tensor:
        if not self.steps:
            return torch.empty(0)

        return torch.stack(
            [step.values.reshape(-1) for step in self.steps],
            dim=0,
        )

    def dones_tensor(self) -> torch.Tensor:
        if not self.steps:
            return torch.empty(0)

        return torch.stack(
            [step.dones.reshape(-1) for step in self.steps],
            dim=0,
        )

    def compute_advantages(
        self,
        last_values: torch.Tensor,
        last_dones: torch.Tensor,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute GAE independently for each dynamically discovered agent.

        Returns:
            advantages: [time, agents]
            returns:    [time, agents]
        """
        if not self.steps:
            raise ValueError("Cannot compute advantages from an empty rollout.")

        rewards = self.rewards_tensor()
        values = self.values_tensor()
        dones = self.dones_tensor()

        last_values = last_values.reshape(-1).cpu()
        last_dones = last_dones.reshape(-1).cpu()

        advantages = torch.zeros_like(rewards)

        gae = torch.zeros_like(last_values)

        for t in reversed(range(len(self.steps))):
            if t == len(self.steps) - 1:
                next_values = last_values
                next_non_terminal = 1.0 - last_dones
            else:
                next_values = values[t + 1]
                next_non_terminal = 1.0 - dones[t]

            delta = (
                rewards[t]
                + gamma * next_values * next_non_terminal
                - values[t]
            )

            gae = (
                delta
                + gamma
                * gae_lambda
                * next_non_terminal
                * gae
            )

            advantages[t] = gae

        returns = advantages + values

        return advantages, returns

    def flatten(
        self,
        advantages: torch.Tensor,
        returns: torch.Tensor,
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        """
        Flatten time and agent dimensions into PPO samples.
        """

        observations = torch.cat(
            [step.observations for step in self.steps],
            dim=0,
        )

        global_observations = torch.cat(
            [step.global_observations for step in self.steps],
            dim=0,
        )

        actions = torch.cat(
            [step.actions.reshape(-1) for step in self.steps],
            dim=0,
        )

        action_masks = torch.cat(
            [step.action_masks for step in self.steps],
            dim=0,
        )

        old_log_probabilities = torch.cat(
            [
                step.log_probabilities.reshape(-1)
                for step in self.steps
            ],
            dim=0,
        )

        old_values = torch.cat(
            [step.values.reshape(-1) for step in self.steps],
            dim=0,
        )

        return {
            "observations": observations.to(device),
            "global_observations": global_observations.to(device),
            "actions": actions.to(device).long(),
            "action_masks": action_masks.to(device).bool(),
            "old_log_probabilities": old_log_probabilities.to(device),
            "old_values": old_values.to(device),
            "advantages": advantages.reshape(-1).to(device),
            "returns": returns.reshape(-1).to(device),
        }


class MAPPOTrainer:
    """
    PPO training engine for the shared MAPPO actor and centralized critic.
    """

    def __init__(
        self,
        policy: MAPPOPolicy,
        config: Optional[PPOTrainingConfig] = None,
    ) -> None:
        self.policy = policy
        self.config = config or PPOTrainingConfig()

        self.device = torch.device(self.config.device)

        self.policy.to(self.device)

        self.actor_optimizer = Adam(
            self.policy.actor.parameters(),
            lr=self.config.learning_rate,
        )

        self.critic_optimizer = Adam(
            self.policy.critic.parameters(),
            lr=self.config.critic_learning_rate,
        )

        self.update_count = 0

    def _actor_log_prob_entropy(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        action_masks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate log probability and entropy for selected actions.

        This mirrors the action-mask behavior used by SharedActor.
        """

        distribution = self.policy.actor.distribution(
            observations,
            action_mask=action_masks,
        )


        log_probabilities = distribution.log_prob(actions)

        entropy = distribution.entropy()

        return log_probabilities, entropy

    def _critic_values(
        self,
        global_observations: torch.Tensor,
        sample_count: int,
    ) -> torch.Tensor:
        """
        Evaluate centralized critic.

        The current centralized critic accepts:
            [num_agents, observation_dim]

        and returns a scalar state value.

        During flattened PPO training, the same global state value is
        expanded across the agent samples belonging to that state.
        """

        # The trainer can also work with a critic that returns one value
        # per input state/agent.
        values = self.policy.critic(global_observations)

        values = values.reshape(-1)

        if values.numel() == 1:
            values = values.expand(sample_count)

        return values

    def _gradient_norm(
        self,
        module: nn.Module,
    ) -> float:
        total = 0.0

        for parameter in module.parameters():
            if parameter.grad is None:
                continue

            norm = parameter.grad.detach().norm(2)
            total += float(norm.item() ** 2)

        return total ** 0.5

    def update(
        self,
        buffer: RolloutBuffer,
        last_values: torch.Tensor,
        last_dones: torch.Tensor,
    ) -> TrainingMetrics:
        if len(buffer) == 0:
            raise ValueError("Cannot update MAPPO with an empty rollout.")

        advantages, returns = buffer.compute_advantages(
            last_values=last_values,
            last_dones=last_dones,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
        )

        # Normalize advantages globally.
        advantages = (
            advantages - advantages.mean()
        ) / (advantages.std(unbiased=False) + 1e-8)

        batch = buffer.flatten(
            advantages=advantages,
            returns=returns,
            device=self.device,
        )

        sample_count = batch["actions"].shape[0]

        if sample_count == 0:
            raise ValueError("Rollout contains zero training samples.")

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_loss = 0.0

        total_actor_norm = 0.0
        total_critic_norm = 0.0

        update_iterations = 0

        for _ in range(self.config.ppo_epochs):
            permutation = torch.randperm(
                sample_count,
                device=self.device,
            )

            for start in range(
                0,
                sample_count,
                self.config.mini_batch_size,
            ):
                indices = permutation[
                    start:start + self.config.mini_batch_size
                ]

                observations = batch["observations"][indices]
                actions = batch["actions"][indices]
                action_masks = batch["action_masks"][indices]

                old_log_probs = batch[
                    "old_log_probabilities"
                ][indices]

                minibatch_advantages = batch[
                    "advantages"
                ][indices]

                minibatch_returns = batch[
                    "returns"
                ][indices]

                global_observations = batch[
                    "global_observations"
                ][indices]

                new_log_probs, entropy = self._actor_log_prob_entropy(
                    observations,
                    actions,
                    action_masks,
                )

                policy_loss = clipped_policy_loss(
                    new_log_probability=new_log_probs,
                    old_log_probability=old_log_probs,
                    advantages=minibatch_advantages,
                    clip_epsilon=self.config.clip_epsilon,
                )

                entropy_value = entropy_bonus(entropy)

                actor_loss = (
                    policy_loss
                    - self.config.entropy_coefficient
                    * entropy_value
                )

                self.actor_optimizer.zero_grad(
                    set_to_none=True
                )

                actor_loss.backward()

                actor_norm = torch.nn.utils.clip_grad_norm_(
                    self.policy.actor.parameters(),
                    self.config.max_grad_norm,
                )

                self.actor_optimizer.step()

                values = self._critic_values(
                    global_observations,
                    sample_count=len(indices),
                )

                critic_loss = value_loss(
                    predicted_values=values,
                    returns=minibatch_returns,
                )

                critic_total_loss = (
                    self.config.value_loss_coefficient
                    * critic_loss
                )

                self.critic_optimizer.zero_grad(
                    set_to_none=True
                )

                critic_total_loss.backward()

                critic_norm = torch.nn.utils.clip_grad_norm_(
                    self.policy.critic.parameters(),
                    self.config.max_grad_norm,
                )

                self.critic_optimizer.step()

                total_policy_loss += float(
                    policy_loss.detach().item()
                )

                total_value_loss += float(
                    critic_loss.detach().item()
                )

                total_entropy += float(
                    entropy_value.detach().item()
                )

                total_loss += float(
                    (
                        actor_loss.detach()
                        + critic_total_loss.detach()
                    ).item()
                )

                total_actor_norm += float(
                    actor_norm.detach().item()
                    if torch.is_tensor(actor_norm)
                    else actor_norm
                )

                total_critic_norm += float(
                    critic_norm.detach().item()
                    if torch.is_tensor(critic_norm)
                    else critic_norm
                )

                update_iterations += 1

        self.update_count += 1

        divisor = max(update_iterations, 1)

        metrics = TrainingMetrics(
            policy_loss=total_policy_loss / divisor,
            value_loss=total_value_loss / divisor,
            entropy=total_entropy / divisor,
            total_loss=total_loss / divisor,
            mean_reward=float(
                buffer.rewards_tensor().mean().item()
            ),
            episode_reward=float(
                buffer.rewards_tensor().sum().item()
            ),
            mean_advantage=float(
                advantages.mean().item()
            ),
            mean_return=float(
                returns.mean().item()
            ),
            actor_gradient_norm=total_actor_norm / divisor,
            critic_gradient_norm=total_critic_norm / divisor,
            update_count=self.update_count,
        )

        buffer.clear()

        return metrics

    def save_checkpoint(
        self,
        path: Optional[str] = None,
        metrics: Optional[TrainingMetrics] = None,
    ) -> Path:
        if path is None:
            checkpoint_dir = Path(
                self.config.checkpoint_dir
            )

            checkpoint_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            path = str(
                checkpoint_dir
                / f"mappo_update_{self.update_count:06d}.pt"
            )

        checkpoint_path = Path(path)

        checkpoint_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        torch.save(
            {
                "policy_state_dict": self.policy.state_dict(),
                "actor_optimizer_state_dict": (
                    self.actor_optimizer.state_dict()
                ),
                "critic_optimizer_state_dict": (
                    self.critic_optimizer.state_dict()
                ),
                "update_count": self.update_count,
                "config": self.config.__dict__,
                "metrics": (
                    metrics.__dict__
                    if metrics is not None
                    else None
                ),
            },
            checkpoint_path,
        )

        return checkpoint_path

    def load_checkpoint(
        self,
        path: str,
        strict: bool = True,
    ) -> TrainingMetrics | None:
        checkpoint = torch.load(
            path,
            map_location=self.device,
        )

        self.policy.load_state_dict(
            checkpoint["policy_state_dict"],
            strict=strict,
        )

        self.actor_optimizer.load_state_dict(
            checkpoint["actor_optimizer_state_dict"]
        )

        self.critic_optimizer.load_state_dict(
            checkpoint["critic_optimizer_state_dict"]
        )

        self.update_count = int(
            checkpoint.get("update_count", 0)
        )

        metrics_data = checkpoint.get("metrics")

        if metrics_data is None:
            return None

        return TrainingMetrics(**metrics_data)
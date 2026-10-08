import pytest
import torch

from marl.mappo import (
    CentralizedCritic,
    MAPPOPolicy,
    SharedActor,
    clipped_policy_loss,
    entropy_bonus,
    generalized_advantage_estimation,
    value_loss,
)


def test_shared_actor_output_shape():

    actor = SharedActor(
        observation_dim=15,
        action_dim=8,
    )

    observations = torch.randn(
        4,
        15,
    )

    logits, green = actor(
        observations
    )

    assert logits.shape == (4, 8)
    assert green.shape == (4,)


def test_actor_green_extension_is_bounded():

    actor = SharedActor(
        observation_dim=15,
        action_dim=8,
    )

    observations = torch.randn(
        10,
        15,
    )

    _, green = actor(
        observations
    )

    assert torch.all(
        green >= 0.0
    )

    assert torch.all(
        green <= 1.0
    )


def test_action_mask_blocks_actions():

    actor = SharedActor(
        observation_dim=15,
        action_dim=4,
    )

    observations = torch.randn(
        1,
        15,
    )

    mask = torch.tensor(
        [[True, False, True, False]]
    )

    logits, _ = actor(
        observations,
        mask,
    )

    assert logits[0, 1] < -1e8
    assert logits[0, 3] < -1e8


def test_policy_sampling_respects_mask():

    policy = MAPPOPolicy(
        observation_dim=15,
        action_dim=4,
    )

    observation = torch.randn(
        15
    )

    mask = torch.tensor(
        [[False, True, False, False]]
    )

    output = policy.act(
        observation,
        mask,
    )

    assert (
        output.movement_index.item()
        == 1
    )


def test_deterministic_action_is_repeatable():

    policy = MAPPOPolicy(
        observation_dim=15,
        action_dim=4,
    )

    observation = torch.randn(
        15
    )

    mask = torch.ones(
        1,
        4,
        dtype=torch.bool,
    )

    first = policy.act(
        observation,
        mask,
        deterministic=True,
    )

    second = policy.act(
        observation,
        mask,
        deterministic=True,
    )

    assert torch.equal(
        first.movement_index,
        second.movement_index,
    )


def test_centralized_critic_accepts_dynamic_agent_count():

    critic = CentralizedCritic(
        observation_dim=15,
        max_agents=8,
    )

    observations = torch.randn(
        3,
        5,
        15,
    )

    values = critic(
        observations
    )

    assert values.shape == (3,)


def test_centralized_critic_accepts_single_state():

    critic = CentralizedCritic(
        observation_dim=15,
        max_agents=8,
    )

    observations = torch.randn(
        5,
        15,
    )

    values = critic(
        observations
    )

    assert values.shape == (1,)


def test_critic_rejects_too_many_agents():

    critic = CentralizedCritic(
        observation_dim=15,
        max_agents=3,
    )

    observations = torch.randn(
        2,
        4,
        15,
    )

    with pytest.raises(
        ValueError
    ):
        critic(observations)


def test_policy_value_output():

    policy = MAPPOPolicy(
        observation_dim=15,
        action_dim=8,
        max_agents=4,
    )

    observations = torch.randn(
        2,
        3,
        15,
    )

    values = policy.value(
        observations
    )

    assert values.shape == (2,)


def test_ppo_loss_is_finite():

    new_log_probs = torch.tensor(
        [-0.2, -0.3, -0.4]
    )

    old_log_probs = torch.tensor(
        [-0.25, -0.35, -0.45]
    )

    advantages = torch.tensor(
        [1.0, 0.5, -0.5]
    )

    loss = clipped_policy_loss(
        new_log_probs,
        old_log_probs,
        advantages,
    )

    assert torch.isfinite(loss)


def test_value_loss_is_finite():

    predicted = torch.tensor(
        [1.0, 2.0, 3.0]
    )

    returns = torch.tensor(
        [1.5, 2.0, 2.5]
    )

    loss = value_loss(
        predicted,
        returns,
    )

    assert torch.isfinite(loss)
    assert loss.item() >= 0.0


def test_entropy_bonus_is_positive():

    entropy = torch.tensor(
        [0.5, 0.8, 1.0]
    )

    result = entropy_bonus(
        entropy
    )

    assert result.item() > 0.0


def test_gae_returns_expected_shapes():

    rewards = torch.tensor(
        [1.0, 2.0, 3.0, 1.0]
    )

    values = torch.tensor(
        [0.5, 0.7, 1.0, 0.8]
    )

    dones = torch.tensor(
        [0.0, 0.0, 0.0, 1.0]
    )

    advantages, returns = (
        generalized_advantage_estimation(
            rewards,
            values,
            dones,
        )
    )

    assert advantages.shape == rewards.shape
    assert returns.shape == rewards.shape

    assert torch.all(
        torch.isfinite(advantages)
    )

    assert torch.all(
        torch.isfinite(returns)
    )


def test_mappo_has_shared_actor_and_centralized_critic():

    policy = MAPPOPolicy(
        observation_dim=15,
        action_dim=8,
        max_agents=6,
    )

    assert isinstance(
        policy.actor,
        SharedActor,
    )

    assert isinstance(
        policy.critic,
        CentralizedCritic,
    )


def test_no_hardcoded_junction_ids():

    policy = MAPPOPolicy(
        observation_dim=15,
        action_dim=8,
        max_agents=10,
    )

    observations = torch.randn(
        5,
        15,
    )

    mask = torch.ones(
        5,
        8,
        dtype=torch.bool,
    )

    logits, _ = policy.actor(
        observations,
        mask,
    )

    assert logits.shape == (
        5,
        8,
    )
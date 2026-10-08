import torch

from src.marl.mappo import MAPPOPolicy
from src.marl.training import (
    MAPPOTrainer,
    PPOTrainingConfig,
    RolloutBuffer,
)


OBSERVATION_DIM = 15
ACTION_DIM = 16
AGENT_COUNT = 3


def make_policy():
    return MAPPOPolicy(
        observation_dim=OBSERVATION_DIM,
        action_dim=ACTION_DIM,
        hidden_dim=32,
        max_agents=8,
    )


def make_rollout(
    steps=5,
    agents=AGENT_COUNT,
):
    buffer = RolloutBuffer()

    for _ in range(steps):
        observations = torch.randn(
            agents,
            OBSERVATION_DIM,
        )

        global_observations = observations.clone()

        actions = torch.randint(
            0,
            ACTION_DIM,
            (agents,),
        )

        action_masks = torch.ones(
            agents,
            ACTION_DIM,
            dtype=torch.bool,
        )

        log_probabilities = torch.randn(
            agents,
        )

        values = torch.randn(
            agents,
        )

        rewards = torch.randn(
            agents,
        )

        dones = torch.zeros(
            agents,
        )

        buffer.add(
            observations=observations,
            global_observations=global_observations,
            actions=actions,
            action_masks=action_masks,
            log_probabilities=log_probabilities,
            values=values,
            rewards=rewards,
            dones=dones,
            agent_ids=[
                f"junction_{i}"
                for i in range(agents)
            ],
        )

    return buffer


def test_rollout_buffer_stores_steps():
    buffer = make_rollout()

    assert len(buffer) == 5


def test_rollout_buffer_rewards_shape():
    buffer = make_rollout(
        steps=4,
        agents=3,
    )

    rewards = buffer.rewards_tensor()

    assert rewards.shape == (4, 3)


def test_rollout_buffer_values_shape():
    buffer = make_rollout(
        steps=4,
        agents=3,
    )

    values = buffer.values_tensor()

    assert values.shape == (4, 3)


def test_rollout_buffer_computes_gae():
    buffer = make_rollout(
        steps=5,
        agents=3,
    )

    advantages, returns = buffer.compute_advantages(
        last_values=torch.zeros(3),
        last_dones=torch.zeros(3),
    )

    assert advantages.shape == (5, 3)
    assert returns.shape == (5, 3)


def test_rollout_flatten():
    buffer = make_rollout(
        steps=5,
        agents=3,
    )

    advantages, returns = buffer.compute_advantages(
        last_values=torch.zeros(3),
        last_dones=torch.zeros(3),
    )

    batch = buffer.flatten(
        advantages=advantages,
        returns=returns,
        device=torch.device("cpu"),
    )

    assert batch["observations"].shape == (
        15,
        OBSERVATION_DIM,
    )

    assert batch["actions"].shape == (15,)

    assert batch["action_masks"].shape == (
        15,
        ACTION_DIM,
    )


def test_mappo_trainer_updates_policy():
    policy = make_policy()

    trainer = MAPPOTrainer(
        policy,
        PPOTrainingConfig(
            ppo_epochs=2,
            mini_batch_size=4,
        ),
    )

    buffer = make_rollout(
        steps=5,
        agents=3,
    )

    metrics = trainer.update(
        buffer,
        last_values=torch.zeros(3),
        last_dones=torch.zeros(3),
    )

    assert metrics.update_count == 1
    assert torch.isfinite(
        torch.tensor(metrics.policy_loss)
    )
    assert torch.isfinite(
        torch.tensor(metrics.value_loss)
    )
    assert torch.isfinite(
        torch.tensor(metrics.entropy)
    )


def test_buffer_is_cleared_after_update():
    policy = make_policy()

    trainer = MAPPOTrainer(
        policy,
        PPOTrainingConfig(
            ppo_epochs=1,
            mini_batch_size=8,
        ),
    )

    buffer = make_rollout()

    assert len(buffer) > 0

    trainer.update(
        buffer,
        last_values=torch.zeros(AGENT_COUNT),
        last_dones=torch.zeros(AGENT_COUNT),
    )

    assert len(buffer) == 0


def test_training_multiple_updates():
    policy = make_policy()

    trainer = MAPPOTrainer(
        policy,
        PPOTrainingConfig(
            ppo_epochs=1,
            mini_batch_size=8,
        ),
    )

    for _ in range(3):
        buffer = make_rollout()

        metrics = trainer.update(
            buffer,
            last_values=torch.zeros(AGENT_COUNT),
            last_dones=torch.zeros(AGENT_COUNT),
        )

        assert metrics.update_count > 0

    assert trainer.update_count == 3


def test_checkpoint_save_and_load(tmp_path):
    policy = make_policy()

    trainer = MAPPOTrainer(
        policy,
        PPOTrainingConfig(
            ppo_epochs=1,
            mini_batch_size=8,
        ),
    )

    buffer = make_rollout()

    metrics = trainer.update(
        buffer,
        last_values=torch.zeros(AGENT_COUNT),
        last_dones=torch.zeros(AGENT_COUNT),
    )

    checkpoint = tmp_path / "mappo.pt"

    saved_path = trainer.save_checkpoint(
        str(checkpoint),
        metrics,
    )

    assert saved_path.exists()

    new_policy = make_policy()

    new_trainer = MAPPOTrainer(
        new_policy,
        PPOTrainingConfig(
            ppo_epochs=1,
            mini_batch_size=8,
        ),
    )

    loaded_metrics = new_trainer.load_checkpoint(
        str(checkpoint)
    )

    assert new_trainer.update_count == 1
    assert loaded_metrics is not None


def test_checkpoint_restores_parameters(tmp_path):
    policy = make_policy()

    trainer = MAPPOTrainer(
        policy,
        PPOTrainingConfig(
            ppo_epochs=1,
            mini_batch_size=8,
        ),
    )

    checkpoint = tmp_path / "mappo.pt"

    trainer.save_checkpoint(
        str(checkpoint)
    )

    new_policy = make_policy()

    new_trainer = MAPPOTrainer(
        new_policy,
        PPOTrainingConfig(
            ppo_epochs=1,
            mini_batch_size=8,
        ),
    )

    new_trainer.load_checkpoint(
        str(checkpoint)
    )

    for p1, p2 in zip(
        policy.parameters(),
        new_policy.parameters(),
    ):
        assert torch.allclose(
            p1,
            p2,
        )


def test_dynamic_agent_count():
    policy = make_policy()

    trainer = MAPPOTrainer(
        policy,
        PPOTrainingConfig(
            ppo_epochs=1,
            mini_batch_size=4,
        ),
    )

    for agent_count in [1, 2, 4, 6]:
        buffer = make_rollout(
            steps=3,
            agents=agent_count,
        )

        metrics = trainer.update(
            buffer,
            last_values=torch.zeros(agent_count),
            last_dones=torch.zeros(agent_count),
        )

        assert torch.isfinite(
            torch.tensor(metrics.total_loss)
        )


def test_training_config():
    config = PPOTrainingConfig()

    assert config.gamma == 0.99
    assert config.gae_lambda == 0.95
    assert config.clip_epsilon == 0.2
    assert config.ppo_epochs == 4
    assert config.entropy_coefficient > 0
    assert config.value_loss_coefficient > 0
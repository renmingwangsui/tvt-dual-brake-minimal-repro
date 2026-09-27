"""Checkpoint equivalence, exact RNG restoration and frozen deterministic evaluation."""
from __future__ import annotations

from pathlib import Path
import random
import sys
import tempfile
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from envs.heavy_platoon_env import HeavyPlatoonEnv  # noqa: E402
from envs.scenario import build_synthetic_debug_scenario  # noqa: E402
from rl.mappo.actor import SharedActor, local_observation_vector  # noqa: E402
from rl.mappo.checkpoint import initialize_reproducibility, load_checkpoint, save_checkpoint  # noqa: E402
from rl.mappo.critic import CentralizedCritic, centralized_state_vector  # noqa: E402
from rl.mappo.evaluation import evaluate_deterministic  # noqa: E402
from rl.mappo.reward import DebugReward, DebugRewardWeights  # noqa: E402
from rl.mappo.trainer import MAPPOTrainer, PPOConfig  # noqa: E402
from rl.mappo.buffer import RolloutBatch  # noqa: E402
from rl.mappo.torch_backend import (  # noqa: E402
    TorchCentralizedCritic,
    TorchMAPPOTrainer,
    TorchSharedActor,
    evaluate_torch_deterministic,
    load_torch_checkpoint,
    save_torch_checkpoint,
)

rng, reproducibility = initialize_reproducibility(1, 2, 3, 4, 5)
env = HeavyPlatoonEnv(build_synthetic_debug_scenario(3))
obs_dim = local_observation_vector(env.current_local_observations()[0]).size
state_dim = centralized_state_vector(env.centralized_training_state()).size
actor = SharedActor(obs_dim, 8, np.zeros(2), np.asarray([120_000.0, 80_000.0]), rng)
critic = CentralizedCritic(state_dim, 8, 3, rng)
actor.normalizer.update(np.ones((3, obs_dim)))
config = PPOConfig(epochs=1, minibatch_size=3)
trainer = MAPPOTrainer(actor, critic, config)
configuration = {"config": config, "N": 3}
saved_actor_checksum = actor.checksum()
saved_critic_checksum = critic.checksum()

with tempfile.TemporaryDirectory() as directory:
    path = Path(directory) / "checkpoint.pkl"
    payload = save_checkpoint(
        path, actor, critic, trainer.actor_optimizer, trainer.critic_optimizer,
        7, 11, rng, configuration, reproducibility,
    )
    expected_generator = rng.random(4)
    expected_python = random.random()
    expected_numpy = np.random.random(4)
    actor.parameters["bmu"] += 1.0
    critic.parameters["bout"] += 1.0
    restored = load_checkpoint(
        path, actor, critic, trainer.actor_optimizer, trainer.critic_optimizer,
        rng, configuration,
    )
    assert restored["training_step"] == 7 and restored["environment_step"] == 11
    assert np.array_equal(rng.random(4), expected_generator)
    assert random.random() == expected_python
    assert np.array_equal(np.random.random(4), expected_numpy)
    assert actor.checksum() == saved_actor_checksum and critic.checksum() == saved_critic_checksum
    assert payload["paper_eligible"] is False and payload["diffqp_backward_active"] is False

reward = DebugReward(DebugRewardWeights(0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01), 18.0, 65.0)
factory = lambda: HeavyPlatoonEnv(build_synthetic_debug_scenario(3))
normalizer_before = actor.normalizer.state_dict()
result_a = evaluate_deterministic(actor, factory, reward, 2)
result_b = evaluate_deterministic(actor, factory, reward, 2)
assert result_a == result_b
assert actor.normalizer.count == normalizer_before["count"]
assert np.array_equal(actor.normalizer.mean, normalizer_before["mean"])
assert np.array_equal(actor.normalizer.m2, normalizer_before["m2"])

torch_actor = TorchSharedActor.from_numpy(actor)
torch_critic = TorchCentralizedCritic.from_numpy(critic)
torch_trainer = TorchMAPPOTrainer(torch_actor, torch_critic, config)
observations = np.stack([local_observation_vector(item) for item in env.current_local_observations()])
sample = torch_actor.sample_numpy(observations)
central = centralized_state_vector(env.centralized_training_state())
torch_batch = RolloutBatch(
    observations=observations,
    centralized_states=np.repeat(central[None, :], 3, axis=0),
    agent_indices=np.arange(3),
    nominal_actions=sample.action,
    provisional_actions=sample.action * 0.9,
    executed_actions=sample.action * 0.8,
    old_log_probs=sample.log_prob,
    advantages=np.asarray([1.0, -0.5, 0.25]),
    returns=np.asarray([0.2, -0.1, 0.3]),
    supervisor_modes=np.asarray(["normal_actor"] * 3),
    intervention_norms=np.ones(3),
    rho=np.ones(3),
    rho_H_cert=np.ones(3),
)
torch_trainer.update_minibatch(torch_batch)
assert torch_trainer.actor_optimizer.state and torch_trainer.critic_optimizer.state
torch_actor_checksum = torch_actor.checksum()
torch_critic_checksum = torch_critic.checksum()

with tempfile.TemporaryDirectory() as directory:
    path = Path(directory) / "torch_checkpoint.pt"
    torch.manual_seed(12345)
    save_torch_checkpoint(
        path, torch_actor, torch_critic, torch_trainer, 17, rng,
        configuration, reproducibility,
    )
    expected_torch = torch.rand(4)
    expected_generator = rng.random(4)
    expected_python = random.random()
    expected_numpy = np.random.random(4)
    with torch.no_grad():
        torch_actor.bmu.add_(1.0)
        torch_critic.bout.add_(1.0)
    restored = load_torch_checkpoint(
        path, torch_actor, torch_critic, torch_trainer, rng, configuration
    )
    assert restored["backend"] == "pytorch_autograd"
    assert restored["torch_cpu_rng_state"] is not None
    assert restored["torch_cuda_rng_state_all"] is None
    assert torch.equal(torch.rand(4), expected_torch)
    assert np.array_equal(rng.random(4), expected_generator)
    assert random.random() == expected_python
    assert np.array_equal(np.random.random(4), expected_numpy)
    assert torch_actor.checksum() == torch_actor_checksum
    assert torch_critic.checksum() == torch_critic_checksum
    assert torch_trainer.actor_optimizer.state and torch_trainer.critic_optimizer.state

    # Deterministic continuation includes optimizer moments and counters, not
    # merely identical inference after loading.
    torch_trainer.update_minibatch(torch_batch)
    continued_actor_checksum = torch_actor.checksum()
    continued_critic_checksum = torch_critic.checksum()
    continuation_step = torch_trainer.training_step

    clone_rng = np.random.default_rng(999)
    clone_actor = TorchSharedActor(
        obs_dim, 8, np.zeros(2), np.asarray([120_000.0, 80_000.0]), clone_rng
    )
    clone_critic = TorchCentralizedCritic(state_dim, 8, 3, clone_rng)
    clone_trainer = TorchMAPPOTrainer(clone_actor, clone_critic, config)
    load_torch_checkpoint(path, clone_actor, clone_critic, clone_trainer, clone_rng, configuration)
    clone_trainer.update_minibatch(torch_batch)
    assert clone_actor.checksum() == continued_actor_checksum
    assert clone_critic.checksum() == continued_critic_checksum
    assert clone_trainer.training_step == continuation_step

    # Restore the saved state for the deterministic closed-loop evaluation.
    load_torch_checkpoint(path, torch_actor, torch_critic, torch_trainer, rng, configuration)

torch_evaluation_a = evaluate_torch_deterministic(torch_actor, factory, reward, 2)
torch_evaluation_b = evaluate_torch_deterministic(torch_actor, factory, reward, 2)
assert torch_evaluation_a == torch_evaluation_b
evaluation_discrepancy = max(
    float(np.max(np.abs(np.asarray(left) - np.asarray(right))))
    for left, right in (
        (torch_evaluation_a.nominal_actions, torch_evaluation_b.nominal_actions),
        (torch_evaluation_a.provisional_actions, torch_evaluation_b.provisional_actions),
        (torch_evaluation_a.executed_actions, torch_evaluation_b.executed_actions),
        (torch_evaluation_a.state_trajectory, torch_evaluation_b.state_trajectory),
    )
)
assert evaluation_discrepancy == 0.0

print(
    "PASS: NumPy and PyTorch checkpoint/optimizer/RNG restoration plus deterministic "
    f"u_RL/u_star0/u_final/state evaluation max_discrepancy={evaluation_discrepancy:.3e}"
)

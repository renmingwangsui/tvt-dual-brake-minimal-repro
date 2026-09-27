"""PPO ratio/clipping, finite gradients, optimizer sharing and parameter updates."""
from __future__ import annotations

from pathlib import Path
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rl.mappo.actor import SharedActor  # noqa: E402
from rl.mappo.buffer import RolloutBuffer  # noqa: E402
from rl.mappo.critic import CentralizedCritic  # noqa: E402
from rl.mappo.trainer import MAPPOTrainer, PPOConfig, clipped_surrogate, ppo_probability_ratio  # noqa: E402
from rl.mappo.torch_backend import (  # noqa: E402
    TorchCentralizedCritic,
    TorchMAPPOTrainer,
    TorchSharedActor,
    torch_ppo_losses,
)
from envs.scenario import build_synthetic_debug_scenario  # noqa: E402

ratio = ppo_probability_ratio(np.log(np.asarray([1.25, 0.75, 1.10])), np.zeros(3))
assert np.allclose(ratio, [1.25, 0.75, 1.10])
surrogate, clipped = clipped_surrogate(ratio, np.asarray([1.0, -1.0, 2.0]), 0.2)
assert np.allclose(surrogate, [1.2, -0.8, 2.2]) and np.array_equal(clipped, [True, True, False])

rng = np.random.default_rng(42)
actor = SharedActor(3, 7, np.zeros(2), np.ones(2), rng)
critic = CentralizedCritic(5, 6, 2, rng)
observations = np.asarray([[0.2, -0.1, 0.3], [-0.4, 0.5, 0.1]])
actor.normalizer.update(observations)
sample = actor.sample(observations, rng)

# A direct gradient-descent probe on -log pi(a|o) must increase that sampled
# nominal action's likelihood; this is a software-level direction sanity check.
probe = SharedActor(3, 7, np.zeros(2), np.ones(2), np.random.default_rng(9))
probe.normalizer.update(observations[:1])
probe_sample = probe.sample(observations[:1], np.random.default_rng(10))
probe_before = float(probe.log_prob(observations[:1], probe_sample.action)[0])
probe_gradients = probe.log_prob_gradients(observations[:1], probe_sample.action, np.asarray([-1.0]))
for name in probe.parameters:
    probe.parameters[name] -= 1e-4 * probe_gradients[name]
probe_after = float(probe.log_prob(observations[:1], probe_sample.action)[0])
assert probe_after > probe_before

# Both controlled trucks contribute non-zero terms to the same shared actor gradient.
coefficient_a = np.asarray([-0.5, 0.0])
coefficient_b = np.asarray([0.0, 0.7])
gradient_a = actor.log_prob_gradients(observations, sample.action, coefficient_a)
gradient_b = actor.log_prob_gradients(observations, sample.action, coefficient_b)
gradient_both = actor.log_prob_gradients(observations, sample.action, coefficient_a + coefficient_b)
assert any(np.linalg.norm(gradient_a[name]) > 0.0 for name in gradient_a)
assert any(np.linalg.norm(gradient_b[name]) > 0.0 for name in gradient_b)
assert all(np.allclose(gradient_both[name], gradient_a[name] + gradient_b[name]) for name in gradient_both)

buffer = RolloutBuffer(2)
for time in range(3):
    step_obs = observations + 0.02 * time
    step_sample = actor.sample(step_obs, rng)
    buffer.add(
        observations=step_obs, centralized_state=np.asarray([time, 1.0, 2.0, 3.0, 4.0]),
        nominal_actions=step_sample.action, old_log_probs=step_sample.log_prob,
        provisional_actions=step_sample.action * 0.8, executed_actions=step_sample.action * 0.6,
        rewards=np.asarray([1.0 + time, 0.5 + time]), terminals=np.zeros(2, dtype=bool),
        truncations=np.asarray([time == 2] * 2), values=np.zeros(2), next_values=np.zeros(2),
        supervisor_modes=np.asarray(["normal_actor", "normal_actor"]),
        intervention_norms=np.ones(2), rho=np.ones(2), rho_H_cert=np.ones(2),
        critical_vehicle_ids=np.asarray([1, 2]), critical_components=np.asarray(["x", "y"]),
        message_validity=np.asarray([True, True]),
    )
buffer.compute_gae(0.9, 0.95)
config = PPOConfig(epochs=2, minibatch_size=3)

# Frozen-minibatch numerical equivalence before either optimizer changes the
# mapped weights. PPO likelihood is evaluated on nominal_actions (u_RL), not
# provisional_actions (u_star0) or executed_actions (u_final).
frozen_batch = buffer.flattened()
torch_actor = TorchSharedActor.from_numpy(actor)
torch_critic = TorchCentralizedCritic.from_numpy(critic)
torch_losses = torch_ppo_losses(torch_actor, torch_critic, frozen_batch, config)
numpy_new_log_prob = actor.log_prob(frozen_batch.observations, frozen_batch.nominal_actions)
numpy_ratio = ppo_probability_ratio(numpy_new_log_prob, frozen_batch.old_log_probs)
numpy_surrogate, _ = clipped_surrogate(numpy_ratio, frozen_batch.advantages, config.clip_epsilon)
numpy_policy_loss = -float(np.mean(numpy_surrogate))
numpy_entropy = -float(np.mean(numpy_new_log_prob))
numpy_actor_loss = numpy_policy_loss - config.entropy_coefficient * numpy_entropy
numpy_value_loss, _, _ = critic.loss_and_gradients(
    frozen_batch.centralized_states, frozen_batch.agent_indices, frozen_batch.returns
)
numpy_critic_loss = config.value_coefficient * numpy_value_loss
ratio_error = np.abs(torch_losses.ratio.detach().numpy() - numpy_ratio)
loss_errors = np.abs(np.asarray([
    torch_losses.policy_loss.detach().item() - numpy_policy_loss,
    torch_losses.entropy.detach().item() - numpy_entropy,
    torch_losses.actor_loss.detach().item() - numpy_actor_loss,
    torch_losses.value_loss.detach().item() - numpy_value_loss,
    torch_losses.critic_loss.detach().item() - numpy_critic_loss,
]))
assert np.max(ratio_error) < 1e-12 and np.max(loss_errors) < 1e-12

torch_trainer = TorchMAPPOTrainer(torch_actor, torch_critic, config)
torch_actor_before = torch_actor.checksum()
torch_critic_before = torch_critic.checksum()
torch_actor_norm_before = float(torch.sqrt(sum(parameter.detach().square().sum() for parameter in torch_actor.parameters())))
torch_critic_norm_before = float(torch.sqrt(sum(parameter.detach().square().sum() for parameter in torch_critic.parameters())))
safety_parameters_before = repr(build_synthetic_debug_scenario(3).parameters)
torch_trainer.update_minibatch(frozen_batch)
assert torch_actor.checksum() != torch_actor_before
assert torch_critic.checksum() != torch_critic_before
torch_actor_norm_after = float(torch.sqrt(sum(parameter.detach().square().sum() for parameter in torch_actor.parameters())))
torch_critic_norm_after = float(torch.sqrt(sum(parameter.detach().square().sum() for parameter in torch_critic.parameters())))
assert repr(build_synthetic_debug_scenario(3).parameters) == safety_parameters_before
assert torch_trainer.last_actor_gradient_norm > 0.0
assert torch_trainer.last_critic_gradient_norm > 0.0
for module in (torch_actor, torch_critic):
    assert all(parameter.requires_grad for parameter in module.parameters())
    assert all(parameter.grad is not None for parameter in module.parameters())
    assert all(torch.all(torch.isfinite(parameter.grad)) for parameter in module.parameters())

trainer = MAPPOTrainer(actor, critic, config)
actor_before, critic_before = actor.checksum(), critic.checksum()
metrics = trainer.update(buffer, rng)
assert actor.checksum() != actor_before and critic.checksum() != critic_before
assert trainer.actor_optimizer.step_count == metrics.minibatch_updates
assert np.all(np.isfinite([
    metrics.policy_loss, metrics.value_loss, metrics.entropy, metrics.approximate_kl,
    metrics.clip_fraction, metrics.actor_gradient_norm, metrics.critic_gradient_norm,
]))
assert metrics.actor_gradient_norm > 0.0 and metrics.critic_gradient_norm > 0.0

print(
    "PASS: PPO nominal-action likelihood, NumPy/PyTorch loss equivalence, and genuine autograd update "
    f"ratio_max={np.max(ratio_error):.3e} loss_max={np.max(loss_errors):.3e} "
    f"actor_grad_norm={torch_trainer.last_actor_gradient_norm:.6e} "
    f"critic_grad_norm={torch_trainer.last_critic_gradient_norm:.6e} "
    f"actor_checksum={torch_actor_before[:12]}->{torch_actor.checksum()[:12]} "
    f"critic_checksum={torch_critic_before[:12]}->{torch_critic.checksum()[:12]} "
    f"actor_param_norm={torch_actor_norm_before:.6e}->{torch_actor_norm_after:.6e} "
    f"critic_param_norm={torch_critic_norm_before:.6e}->{torch_critic_norm_after:.6e}"
)

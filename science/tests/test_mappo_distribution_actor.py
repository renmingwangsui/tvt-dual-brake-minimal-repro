"""Actor bounds, information separation and exact squashed-density tests."""
from __future__ import annotations

from dataclasses import fields
from pathlib import Path
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from envs.heavy_platoon_env import HeavyPlatoonEnv  # noqa: E402
from envs.observations import LocalActorObservation  # noqa: E402
from envs.scenario import build_synthetic_debug_scenario  # noqa: E402
from rl.mappo.actor import SharedActor, local_observation_vector  # noqa: E402
from rl.mappo.critic import CentralizedCritic, centralized_state_vector  # noqa: E402
from rl.mappo.distributions import SquashedGaussian  # noqa: E402
from rl.mappo.torch_backend import (  # noqa: E402
    TorchCentralizedCritic,
    TorchSharedActor,
)
from rl.mappo import (  # noqa: E402
    CentralizedCritic as DefaultCentralizedCritic,
    MAPPOTrainer as DefaultMAPPOTrainer,
    SharedActor as DefaultSharedActor,
)
from rl.mappo.torch_backend import TorchMAPPOTrainer  # noqa: E402

assert DefaultSharedActor is TorchSharedActor
assert DefaultCentralizedCritic is TorchCentralizedCritic
assert DefaultMAPPOTrainer is TorchMAPPOTrainer

rng = np.random.default_rng(12)
lower = np.asarray([0.0, 10.0])
upper = np.asarray([120_000.0, 80_000.0])
distribution = SquashedGaussian(lower, upper)
mean = np.asarray([[0.2, -0.3], [0.8, 0.1]])
log_std = np.asarray([[-0.7, -0.2], [-0.4, -1.0]])
raw = np.asarray([[0.4, -0.5], [1.1, 0.25]])
action = distribution.transform(raw)
implemented = distribution.log_prob(mean, log_std, action)
normal = -0.5 * (((raw - mean) / np.exp(log_std)) ** 2 + 2.0 * log_std + np.log(2.0 * np.pi))
jacobian = np.log((upper - lower) / 2.0) + np.log(1.0 - np.tanh(raw) ** 2)
reference = np.sum(normal - jacobian, axis=1)
assert np.allclose(implemented, reference, rtol=1e-12, atol=1e-12)
assert np.allclose(distribution.inverse(action)[0], raw)
assert np.allclose(distribution.transform(np.zeros((1, 2))), ((lower + upper) / 2.0)[None, :])

env = HeavyPlatoonEnv(build_synthetic_debug_scenario(3))
observations = np.stack([local_observation_vector(item) for item in env.current_local_observations()])
actor = SharedActor(observations.shape[1], 12, lower, upper, rng)
sample = actor.sample(observations, rng)
assert sample.action.shape == (3, 2) and sample.log_prob.shape == (3,)
assert np.all(sample.action >= lower) and np.all(sample.action <= upper)
_, actor_log_std = actor.distribution_parameters(observations)
assert np.all(np.isfinite(actor_log_std)) and np.all((-5.0 <= actor_log_std) & (actor_log_std <= 2.0))

local_fields = {field.name for field in fields(LocalActorObservation)}
privileged = {
    "global_fleet_state", "centralized_fleet_minimum", "future_predecessor_state",
    "true_disturbance_realization", "other_vehicle_states", "critic_state",
}
assert local_fields.isdisjoint(privileged)
assert observations.shape[1] == len(local_fields) == 11

central = centralized_state_vector(env.centralized_training_state())
critic = CentralizedCritic(central.size, 9, 3, rng)
assert critic.forward(central).shape == (1, 3)
try:
    critic.forward(np.append(central, 0.0))
except ValueError:
    pass
else:
    raise AssertionError("fixed-N critic must reject silent padding")

torch_actor = TorchSharedActor.from_numpy(actor)
numpy_mean, numpy_log_std = actor.distribution_parameters(observations)
torch_mean, torch_log_std = torch_actor.distribution_parameters(observations)
actor_error = np.abs(numpy_mean - torch_mean)
log_std_error = np.abs(numpy_log_std - torch_log_std)
assert np.max(actor_error) < 1e-12
assert np.max(log_std_error) < 1e-12

# Controlled raw samples isolate the distribution transformation and density
# from RNG implementation details while retaining the reparameterized graph.
mean_tensor, log_std_tensor = torch_actor(observations)
raw_tensor = torch.as_tensor(raw[:1].repeat(3, axis=0), dtype=torch.float64)
noise_tensor = (raw_tensor - mean_tensor) / torch.exp(log_std_tensor)
torch_sample = torch_actor.distribution.rsample(mean_tensor, log_std_tensor, noise=noise_tensor)
numpy_action = distribution.transform(raw_tensor.detach().numpy())
numpy_log_prob = distribution.log_prob_from_raw(
    mean_tensor.detach().numpy(), log_std_tensor.detach().numpy(), raw_tensor.detach().numpy()
)
log_prob_error = np.abs(torch_sample.log_prob.detach().numpy() - numpy_log_prob)
assert np.allclose(
    torch_sample.action.detach().numpy(), numpy_action, rtol=1e-14, atol=1e-10
)
assert np.max(log_prob_error) < 1e-12
(-torch_sample.log_prob.mean()).backward()
assert all(parameter.grad is not None for parameter in torch_actor.parameters())
assert all(torch.all(torch.isfinite(parameter.grad)) for parameter in torch_actor.parameters())

torch_critic = TorchCentralizedCritic.from_numpy(critic)
numpy_values = critic.forward(central)
torch_values = torch_critic.values_numpy(central)
critic_error = np.abs(numpy_values - torch_values)
assert np.max(critic_error) < 1e-12

print(
    "PASS: MAPPO NumPy reference plus PyTorch actor/critic/squashed-Gaussian equivalence "
    f"actor_max={np.max(actor_error):.3e} actor_median={np.median(actor_error):.3e} "
    f"log_std_max={np.max(log_std_error):.3e} critic_max={np.max(critic_error):.3e} "
    f"critic_median={np.median(critic_error):.3e} log_prob_max={np.max(log_prob_error):.3e} "
    f"log_prob_median={np.median(log_prob_error):.3e}"
)

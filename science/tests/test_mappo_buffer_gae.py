"""Rollout schema/batching and hand-computed terminal/truncation GAE tests."""
from __future__ import annotations

from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rl.mappo.buffer import RolloutBuffer  # noqa: E402
from rl.mappo.gae import generalized_advantage_estimation  # noqa: E402
from rl.mappo.torch_backend import torch_generalized_advantage_estimation  # noqa: E402

rewards = np.asarray([[1.0], [2.0]])
values = np.asarray([[0.5], [0.4]])
next_values = np.asarray([[0.4], [0.3]])
zeros = np.zeros_like(rewards)
advantages, returns = generalized_advantage_estimation(rewards, values, next_values, zeros, zeros, 0.9, 0.8)
assert np.allclose(advantages[:, 0], [2.2064, 1.87])
assert np.allclose(returns, advantages + values)

terminal = zeros.copy()
terminal[-1] = 1.0
terminal_advantages, _ = generalized_advantage_estimation(rewards, values, next_values, terminal, zeros, 0.9, 0.8)
assert np.allclose(terminal_advantages[:, 0], [2.012, 1.6])

truncation = zeros.copy()
truncation[0] = 1.0
truncated_advantages, _ = generalized_advantage_estimation(rewards, values, next_values, zeros, truncation, 0.9, 0.8)
assert np.allclose(truncated_advantages[:, 0], [0.86, 1.87])

buffer = RolloutBuffer(2)
for time in range(2):
    nominal = np.full((2, 2), 10.0 + time)
    provisional = nominal - 1.0
    executed = nominal - 2.0
    buffer.add(
        observations=np.full((2, 11), time), centralized_state=np.full(13, time),
        nominal_actions=nominal, old_log_probs=np.asarray([-1.0, -2.0]),
        provisional_actions=provisional, executed_actions=executed,
        rewards=np.asarray([1.0, 2.0]), terminals=np.zeros(2, dtype=bool),
        truncations=np.asarray([time == 1] * 2), values=np.zeros(2), next_values=np.ones(2),
        supervisor_modes=np.asarray(["normal_actor", "normal_actor"]),
        intervention_norms=np.asarray([1.0, 2.0]), rho=np.asarray([3.0, 4.0]),
        rho_H_cert=np.asarray([5.0, 6.0]), critical_vehicle_ids=np.asarray([1, 2]),
        critical_components=np.asarray(["a", "b"]), message_validity=np.asarray([True, False]),
    )
buffer.compute_gae(0.9, 0.8)
flat = buffer.flattened()
assert flat.observations.shape == (4, 11) and flat.centralized_states.shape == (4, 13)
assert np.all(flat.nominal_actions - flat.provisional_actions == 1.0)
assert np.all(flat.nominal_actions - flat.executed_actions == 2.0)
batches = buffer.minibatches(3, np.random.default_rng(5))
assert sum(len(batch.old_log_probs) for batch in batches) == 4

torch_advantages, torch_returns = torch_generalized_advantage_estimation(
    rewards, values, next_values, zeros, zeros, 0.9, 0.8
)
gae_errors = np.concatenate([
    np.abs(torch_advantages.numpy() - advantages).reshape(-1),
    np.abs(torch_returns.numpy() - returns).reshape(-1),
])
torch_terminal, _ = torch_generalized_advantage_estimation(
    rewards, values, next_values, terminal, zeros, 0.9, 0.8
)
torch_truncated, _ = torch_generalized_advantage_estimation(
    rewards, values, next_values, zeros, truncation, 0.9, 0.8
)
assert np.allclose(torch_terminal.numpy(), terminal_advantages, rtol=0.0, atol=1e-14)
assert np.allclose(torch_truncated.numpy(), truncated_advantages, rtol=0.0, atol=1e-14)
assert np.max(gae_errors) < 1e-14

print(
    "PASS: MAPPO rollout/action schema and NumPy/PyTorch GAE equivalence "
    f"gae_max={np.max(gae_errors):.3e} gae_median={np.median(gae_errors):.3e}"
)

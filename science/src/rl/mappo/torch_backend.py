"""Genuine PyTorch/autograd MAPPO backend with the NumPy core as reference.

This module deliberately stops at the nominal-policy boundary.  The safety
filter, QP, predictive monitor, supervisor and final re-verification remain
ordinary non-differentiable environment code; no DiffQP backward is present.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import random
from typing import Callable

import numpy as np
import torch
from torch import Tensor, nn

from controllers.debug_nominal import ConstantCommandController
from envs.heavy_platoon_env import HeavyPlatoonEnv

from .actor import RunningNormalizer, SharedActor as NumpySharedActor, local_observation_vector
from .buffer import RolloutBatch, RolloutBuffer
from .checkpoint import configuration_hash, initialize_reproducibility, library_versions
from .critic import CentralizedCritic as NumpyCentralizedCritic, centralized_state_vector
from .reward import DebugReward, DebugRewardWeights
from .trainer import PPOConfig, PPOUpdateMetrics


TORCH_DTYPE = torch.float64


@dataclass(frozen=True)
class TorchSquashedGaussianSample:
    action: Tensor
    raw_action: Tensor
    log_prob: Tensor


@dataclass(frozen=True)
class DetachedPolicySample:
    action: np.ndarray
    raw_action: np.ndarray
    log_prob: np.ndarray


class TorchSquashedGaussian:
    """Affine tanh-squashed diagonal Gaussian with an ``rsample`` path."""

    def __init__(self, lower: Tensor, upper: Tensor, epsilon: float = 1e-7) -> None:
        self.lower = lower
        self.upper = upper
        if lower.shape != upper.shape or bool(torch.any(upper <= lower).item()):
            raise ValueError("action bounds must have equal shapes and upper > lower")
        self.scale = (upper - lower) / 2.0
        self.shift = (upper + lower) / 2.0
        self.epsilon = float(epsilon)

    def _validate(self, mean: Tensor, log_std: Tensor) -> None:
        if mean.shape[-1] != self.lower.numel() or log_std.shape[-1] != self.lower.numel():
            raise ValueError("distribution/action dimensions do not match")
        if not bool(torch.all(torch.isfinite(mean)).item()) or not bool(torch.all(torch.isfinite(log_std)).item()):
            raise FloatingPointError("non-finite Gaussian parameters")

    def transform(self, raw_action: Tensor) -> Tensor:
        return self.shift + self.scale * torch.tanh(raw_action)

    def inverse(self, action: Tensor) -> tuple[Tensor, Tensor]:
        normalized = (action - self.shift) / self.scale
        if bool(torch.any(normalized < -1.0 - 1e-10).item()) or bool(torch.any(normalized > 1.0 + 1e-10).item()):
            raise ValueError("action lies outside affine tanh bounds")
        clipped = torch.clamp(normalized, -1.0 + self.epsilon, 1.0 - self.epsilon)
        return torch.atanh(clipped), clipped

    def log_prob_from_raw(self, mean: Tensor, log_std: Tensor, raw_action: Tensor) -> Tensor:
        self._validate(mean, log_std)
        squashed = torch.tanh(raw_action)
        variance_term = ((raw_action - mean) / torch.exp(log_std)).square()
        normal = -0.5 * (variance_term + 2.0 * log_std + np.log(2.0 * np.pi))
        log_jacobian = torch.log(self.scale) + torch.log(
            torch.clamp(1.0 - squashed.square(), min=self.epsilon)
        )
        return torch.sum(normal - log_jacobian, dim=-1)

    def log_prob(self, mean: Tensor, log_std: Tensor, action: Tensor) -> Tensor:
        raw, _ = self.inverse(action)
        return self.log_prob_from_raw(mean, log_std, raw)

    def rsample(
        self,
        mean: Tensor,
        log_std: Tensor,
        generator: torch.Generator | None = None,
        noise: Tensor | None = None,
    ) -> TorchSquashedGaussianSample:
        self._validate(mean, log_std)
        if noise is None:
            noise = torch.randn(mean.shape, dtype=mean.dtype, device=mean.device, generator=generator)
        raw = mean + torch.exp(log_std) * noise
        return TorchSquashedGaussianSample(
            self.transform(raw), raw, self.log_prob_from_raw(mean, log_std, raw)
        )

    def deterministic(self, mean: Tensor) -> Tensor:
        return self.transform(mean)


class TorchSharedActor(nn.Module):
    """Parameter-shared decentralized actor, numerically mapped from NumPy."""

    def __init__(
        self,
        observation_dim: int,
        hidden_dim: int,
        action_lower: np.ndarray,
        action_upper: np.ndarray,
        rng: np.random.Generator,
    ) -> None:
        super().__init__()
        self.observation_dim = int(observation_dim)
        self.hidden_dim = int(hidden_dim)
        self.action_dim = len(action_lower)
        self.normalizer = RunningNormalizer(self.observation_dim)
        self.register_buffer("action_lower", torch.as_tensor(action_lower, dtype=TORCH_DTYPE).clone())
        self.register_buffer("action_upper", torch.as_tensor(action_upper, dtype=TORCH_DTYPE).clone())
        self.W1 = nn.Parameter(torch.as_tensor(
            rng.normal(0.0, 1.0 / np.sqrt(observation_dim), (observation_dim, hidden_dim)), dtype=TORCH_DTYPE
        ))
        self.b1 = nn.Parameter(torch.zeros(hidden_dim, dtype=TORCH_DTYPE))
        self.Wmu = nn.Parameter(torch.as_tensor(
            rng.normal(0.0, 0.05, (hidden_dim, self.action_dim)), dtype=TORCH_DTYPE
        ))
        self.bmu = nn.Parameter(torch.zeros(self.action_dim, dtype=TORCH_DTYPE))
        self.log_std = nn.Parameter(torch.full((self.action_dim,), -0.5, dtype=TORCH_DTYPE))

    @property
    def distribution(self) -> TorchSquashedGaussian:
        return TorchSquashedGaussian(self.action_lower, self.action_upper)

    def _as_tensor(self, values: np.ndarray | Tensor) -> Tensor:
        if isinstance(values, Tensor):
            return values.to(dtype=TORCH_DTYPE, device=self.W1.device)
        return torch.as_tensor(values, dtype=TORCH_DTYPE, device=self.W1.device)

    def forward(self, observations: np.ndarray | Tensor) -> tuple[Tensor, Tensor]:
        raw = self._as_tensor(observations)
        if raw.ndim == 1:
            raw = raw.unsqueeze(0)
        if raw.shape[1] != self.observation_dim:
            raise ValueError("actor observation dimension mismatch")
        normalizer_mean = torch.as_tensor(self.normalizer.mean, dtype=TORCH_DTYPE, device=raw.device)
        normalizer_variance = torch.as_tensor(self.normalizer.variance, dtype=TORCH_DTYPE, device=raw.device)
        x = (raw - normalizer_mean) / torch.sqrt(normalizer_variance + 1e-8)
        hidden = torch.tanh(x @ self.W1 + self.b1)
        mean = hidden @ self.Wmu + self.bmu
        bounded_log_std = torch.clamp(self.log_std, -5.0, 2.0).expand_as(mean)
        return mean, bounded_log_std

    def distribution_parameters(self, observations: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        with torch.no_grad():
            mean, log_std = self(observations)
        return mean.cpu().numpy(), log_std.cpu().numpy()

    def log_prob_tensor(self, observations: np.ndarray | Tensor, actions: np.ndarray | Tensor) -> Tensor:
        mean, log_std = self(observations)
        return self.distribution.log_prob(mean, log_std, self._as_tensor(actions))

    def sample_tensor(
        self, observations: np.ndarray | Tensor, generator: torch.Generator | None = None
    ) -> TorchSquashedGaussianSample:
        mean, log_std = self(observations)
        return self.distribution.rsample(mean, log_std, generator=generator)

    def sample_numpy(
        self, observations: np.ndarray, generator: torch.Generator | None = None
    ) -> DetachedPolicySample:
        with torch.no_grad():
            sample = self.sample_tensor(observations, generator)
        return DetachedPolicySample(
            sample.action.cpu().numpy(), sample.raw_action.cpu().numpy(), sample.log_prob.cpu().numpy()
        )

    def deterministic_numpy(self, observations: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            mean, _ = self(observations)
            action = self.distribution.deterministic(mean)
        return action.cpu().numpy()

    def load_numpy_reference(self, reference: NumpySharedActor) -> None:
        expected = (self.observation_dim, self.hidden_dim, self.action_dim)
        actual = (reference.observation_dim, reference.hidden_dim, reference.action_dim)
        if actual != expected:
            raise ValueError("NumPy actor architecture mismatch")
        with torch.no_grad():
            for name in ("W1", "b1", "Wmu", "bmu", "log_std"):
                getattr(self, name).copy_(torch.as_tensor(reference.parameters[name], dtype=TORCH_DTYPE))
        self.normalizer.load_state_dict(reference.normalizer.state_dict())

    @classmethod
    def from_numpy(cls, reference: NumpySharedActor, rng: np.random.Generator | None = None) -> "TorchSharedActor":
        instance = cls(
            reference.observation_dim,
            reference.hidden_dim,
            reference.distribution.lower,
            reference.distribution.upper,
            np.random.default_rng(0) if rng is None else rng,
        )
        instance.load_numpy_reference(reference)
        return instance

    def checksum(self) -> str:
        digest = sha256()
        for name, value in sorted(self.state_dict().items()):
            digest.update(name.encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()


class TorchCentralizedCritic(nn.Module):
    """Fixed-N centralized critic with per-agent values."""

    def __init__(
        self,
        state_dim: int,
        hidden_dim: int,
        controlled_truck_count: int,
        rng: np.random.Generator,
    ) -> None:
        super().__init__()
        if controlled_truck_count < 1:
            raise ValueError("critic requires a fixed positive N")
        self.state_dim = int(state_dim)
        self.hidden_dim = int(hidden_dim)
        self.controlled_truck_count = int(controlled_truck_count)
        self.W1 = nn.Parameter(torch.as_tensor(
            rng.normal(0.0, 1.0 / np.sqrt(state_dim), (state_dim, hidden_dim)), dtype=TORCH_DTYPE
        ))
        self.b1 = nn.Parameter(torch.zeros(hidden_dim, dtype=TORCH_DTYPE))
        self.Wout = nn.Parameter(torch.as_tensor(
            rng.normal(0.0, 0.05, (hidden_dim, controlled_truck_count)), dtype=TORCH_DTYPE
        ))
        self.bout = nn.Parameter(torch.zeros(controlled_truck_count, dtype=TORCH_DTYPE))

    def forward(self, states: np.ndarray | Tensor) -> Tensor:
        x = torch.as_tensor(states, dtype=TORCH_DTYPE, device=self.W1.device)
        if x.ndim == 1:
            x = x.unsqueeze(0)
        if x.shape[1] != self.state_dim:
            raise ValueError("critic state dimension mismatch; fixed N cannot be silently padded")
        return torch.tanh(x @ self.W1 + self.b1) @ self.Wout + self.bout

    def values_numpy(self, states: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return self(states).cpu().numpy()

    def load_numpy_reference(self, reference: NumpyCentralizedCritic) -> None:
        expected = (self.state_dim, self.hidden_dim, self.controlled_truck_count)
        actual = (reference.state_dim, reference.hidden_dim, reference.controlled_truck_count)
        if actual != expected:
            raise ValueError("NumPy critic architecture mismatch")
        with torch.no_grad():
            for name in ("W1", "b1", "Wout", "bout"):
                getattr(self, name).copy_(torch.as_tensor(reference.parameters[name], dtype=TORCH_DTYPE))

    @classmethod
    def from_numpy(
        cls, reference: NumpyCentralizedCritic, rng: np.random.Generator | None = None
    ) -> "TorchCentralizedCritic":
        instance = cls(
            reference.state_dim,
            reference.hidden_dim,
            reference.controlled_truck_count,
            np.random.default_rng(0) if rng is None else rng,
        )
        instance.load_numpy_reference(reference)
        return instance

    def checksum(self) -> str:
        digest = sha256()
        for name, value in sorted(self.state_dict().items()):
            digest.update(name.encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()


def torch_generalized_advantage_estimation(
    rewards: np.ndarray | Tensor,
    values: np.ndarray | Tensor,
    next_values: np.ndarray | Tensor,
    terminals: np.ndarray | Tensor,
    truncations: np.ndarray | Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple[Tensor, Tensor]:
    arrays = [torch.as_tensor(item, dtype=TORCH_DTYPE) for item in (rewards, values, next_values, terminals, truncations)]
    if len({tuple(item.shape) for item in arrays}) != 1:
        raise ValueError("GAE arrays must have identical shapes")
    if not 0.0 <= gamma <= 1.0 or not 0.0 <= gae_lambda <= 1.0:
        raise ValueError("gamma and lambda must lie in [0,1]")
    rewards_t, values_t, next_values_t, terminals_t, truncations_t = arrays
    bootstrap_mask = 1.0 - terminals_t
    continuation_mask = 1.0 - torch.maximum(terminals_t, truncations_t)
    deltas = rewards_t + gamma * next_values_t * bootstrap_mask - values_t
    advantages = torch.zeros_like(rewards_t)
    running = torch.zeros_like(rewards_t[0])
    for index in range(rewards_t.shape[0] - 1, -1, -1):
        running = deltas[index] + gamma * gae_lambda * continuation_mask[index] * running
        advantages[index] = running
    return advantages, advantages + values_t


def torch_probability_ratio(new_log_prob: Tensor, old_log_prob: Tensor) -> Tensor:
    difference = new_log_prob - old_log_prob
    if not bool(torch.all(torch.isfinite(difference)).item()):
        raise FloatingPointError("non-finite PPO log-probability difference")
    return torch.exp(torch.clamp(difference, -60.0, 60.0))


def torch_clipped_surrogate(
    ratio: Tensor, advantages: Tensor, clip_epsilon: float
) -> tuple[Tensor, Tensor]:
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
    return torch.minimum(unclipped, clipped), torch.abs(ratio - 1.0) > clip_epsilon


@dataclass(frozen=True)
class TorchPPOLosses:
    old_log_probability: Tensor
    new_log_probability: Tensor
    ratio: Tensor
    surrogate: Tensor
    policy_loss: Tensor
    entropy: Tensor
    actor_loss: Tensor
    selected_values: Tensor
    value_loss: Tensor
    critic_loss: Tensor
    clipped_mask: Tensor


def torch_ppo_losses(
    actor: TorchSharedActor,
    critic: TorchCentralizedCritic,
    batch: RolloutBatch,
    config: PPOConfig,
) -> TorchPPOLosses:
    old_log_prob = torch.as_tensor(batch.old_log_probs, dtype=TORCH_DTYPE)
    advantages = torch.as_tensor(batch.advantages, dtype=TORCH_DTYPE)
    new_log_prob = actor.log_prob_tensor(batch.observations, batch.nominal_actions)
    ratio = torch_probability_ratio(new_log_prob, old_log_prob)
    surrogate, clipped_mask = torch_clipped_surrogate(ratio, advantages, config.clip_epsilon)
    policy_loss = -torch.mean(surrogate)
    entropy = -torch.mean(new_log_prob)
    actor_loss = policy_loss - config.entropy_coefficient * entropy
    all_values = critic(batch.centralized_states)
    row_indices = torch.arange(len(batch.agent_indices), dtype=torch.int64)
    agent_indices = torch.as_tensor(batch.agent_indices, dtype=torch.int64)
    selected_values = all_values[row_indices, agent_indices]
    targets = torch.as_tensor(batch.returns, dtype=TORCH_DTYPE)
    value_loss = 0.5 * torch.mean((selected_values - targets).square())
    critic_loss = config.value_coefficient * value_loss
    return TorchPPOLosses(
        old_log_prob, new_log_prob, ratio, surrogate, policy_loss, entropy,
        actor_loss, selected_values, value_loss, critic_loss, clipped_mask,
    )


class TorchMAPPOTrainer:
    """PPO trainer whose updates are exclusively ``backward`` + ``step``."""

    def __init__(self, actor: TorchSharedActor, critic: TorchCentralizedCritic, config: PPOConfig) -> None:
        self.actor = actor
        self.critic = critic
        self.config = config
        self.actor_optimizer = torch.optim.Adam(actor.parameters(), lr=config.actor_learning_rate)
        self.critic_optimizer = torch.optim.Adam(critic.parameters(), lr=config.critic_learning_rate)
        self.training_step = 0
        self.last_actor_gradient_norm = 0.0
        self.last_critic_gradient_norm = 0.0

    @staticmethod
    def _validate_gradients(module: nn.Module, label: str) -> None:
        gradients = [parameter.grad for parameter in module.parameters() if parameter.requires_grad]
        if not gradients or any(gradient is None for gradient in gradients):
            raise AssertionError(f"{label} autograd did not populate all expected gradients")
        if not all(bool(torch.all(torch.isfinite(gradient)).item()) for gradient in gradients if gradient is not None):
            raise FloatingPointError(f"{label} produced non-finite gradients")
        if not any(float(torch.linalg.vector_norm(gradient).item()) > 0.0 for gradient in gradients if gradient is not None):
            raise AssertionError(f"{label} gradients are all zero")

    def update_minibatch(self, batch: RolloutBatch) -> tuple[float, ...]:
        losses = torch_ppo_losses(self.actor, self.critic, batch, self.config)

        self.actor_optimizer.zero_grad(set_to_none=True)
        losses.actor_loss.backward()
        self._validate_gradients(self.actor, "actor")
        actor_norm = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.maximum_gradient_norm)
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad(set_to_none=True)
        losses.critic_loss.backward()
        self._validate_gradients(self.critic, "critic")
        critic_norm = torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.maximum_gradient_norm)
        self.critic_optimizer.step()

        self.last_actor_gradient_norm = float(actor_norm.item())
        self.last_critic_gradient_norm = float(critic_norm.item())
        fields = (
            float(losses.policy_loss.detach().item()),
            float(losses.value_loss.detach().item()),
            float(losses.entropy.detach().item()),
            float(torch.mean(losses.old_log_probability - losses.new_log_probability).detach().item()),
            float(torch.mean(losses.clipped_mask.to(TORCH_DTYPE)).detach().item()),
            self.last_actor_gradient_norm,
            self.last_critic_gradient_norm,
        )
        if not np.all(np.isfinite(fields)):
            raise FloatingPointError("non-finite PPO loss or diagnostic")
        self.training_step += 1
        return fields

    def update(self, buffer: RolloutBuffer, rng: np.random.Generator) -> PPOUpdateMetrics:
        if buffer.advantages is None:
            raise ValueError("GAE must be computed before PPO update")
        flat = buffer.advantages.reshape(-1)
        buffer.advantages = (buffer.advantages - np.mean(flat)) / (np.std(flat) + 1e-8)
        accumulated: list[tuple[float, ...]] = []
        for _ in range(self.config.epochs):
            for batch in buffer.minibatches(self.config.minibatch_size, rng):
                accumulated.append(self.update_minibatch(batch))
        means = np.mean(np.asarray(accumulated, dtype=np.float64), axis=0)
        return PPOUpdateMetrics(*map(float, means), len(accumulated))


@dataclass(frozen=True)
class TorchRolloutSummary:
    steps: int
    safety_cycles: int
    total_reward: float
    nominal_executed_difference_count: int
    data_provenance: str = "DEBUG"
    paper_eligible: bool = False
    training_mode: str = "debug"


def collect_torch_rollout(
    env: HeavyPlatoonEnv,
    actor: TorchSharedActor,
    critic: TorchCentralizedCritic,
    reward_function: DebugReward,
    horizon: int,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    update_normalization: bool = True,
    torch_generator: torch.Generator | None = None,
) -> tuple[RolloutBuffer, TorchRolloutSummary]:
    if horizon < 1:
        raise ValueError("rollout horizon must be positive")
    if critic.controlled_truck_count != env.controlled_truck_count:
        raise ValueError("critic registered N does not match environment N")
    buffer = RolloutBuffer(env.controlled_truck_count)
    total_reward = 0.0
    distinct = 0
    if update_normalization:
        actor.normalizer.update(np.stack([
            local_observation_vector(item) for item in env.current_local_observations()
        ]))
    for step_index in range(horizon):
        observations = np.stack([local_observation_vector(item) for item in env.current_local_observations()])
        central_state = centralized_state_vector(env.centralized_training_state())
        values = critic.values_numpy(central_state)[0]
        sample = actor.sample_numpy(observations, torch_generator)
        env.nominal_controllers = [
            ConstantCommandController(float(action[0]), float(action[1])) for action in sample.action
        ]
        result = env.step()
        next_state = centralized_state_vector(env.centralized_training_state())
        next_values = critic.values_numpy(next_state)[0]
        provisional = np.asarray([record.provisional_action for record in result.logs], dtype=np.float64)
        executed = np.asarray([record.final_action for record in result.logs], dtype=np.float64)
        nominal = np.asarray([record.nominal_action for record in result.logs], dtype=np.float64)
        if not np.allclose(nominal, sample.action, rtol=0.0, atol=1e-9):
            raise AssertionError("safety controller did not receive u_RL")
        rewards = np.asarray([reward_function(record).total for record in result.logs])
        total_reward += float(np.sum(rewards))
        distinct += int(np.sum(np.any(np.abs(nominal - executed) > 1e-9, axis=1)))
        buffer.add(
            observations=observations,
            centralized_state=central_state,
            nominal_actions=sample.action.copy(),
            old_log_probs=sample.log_prob.copy(),
            provisional_actions=provisional,
            executed_actions=executed,
            rewards=rewards,
            terminals=np.zeros(env.controlled_truck_count, dtype=bool),
            truncations=np.full(env.controlled_truck_count, step_index == horizon - 1),
            values=values,
            next_values=next_values,
            supervisor_modes=np.asarray([record.supervisor_mode for record in result.logs]),
            intervention_norms=np.linalg.norm(executed - nominal, axis=1),
            rho=np.asarray([record.rho for record in result.logs]),
            rho_H_cert=np.asarray([record.rho_H_cert for record in result.logs]),
            critical_vehicle_ids=np.asarray([record.critical_vehicle_id for record in result.logs]),
            critical_components=np.asarray([record.critical_component for record in result.logs]),
            message_validity=np.asarray([record.message_valid for record in result.logs]),
        )
    advantages, returns = torch_generalized_advantage_estimation(
        np.stack([step["rewards"] for step in buffer.steps]),
        np.stack([step["values"] for step in buffer.steps]),
        np.stack([step["next_values"] for step in buffer.steps]),
        np.stack([step["terminals"] for step in buffer.steps]),
        np.stack([step["truncations"] for step in buffer.steps]),
        gamma,
        gae_lambda,
    )
    buffer.advantages = advantages.cpu().numpy()
    buffer.returns = returns.cpu().numpy()
    return buffer, TorchRolloutSummary(horizon, horizon * env.controlled_truck_count, total_reward, distinct)


@dataclass(frozen=True)
class TorchEvaluationResult:
    total_reward: float
    nominal_actions: tuple[tuple[tuple[float, float], ...], ...]
    provisional_actions: tuple[tuple[tuple[float, float], ...], ...]
    executed_actions: tuple[tuple[tuple[float, float], ...], ...]
    state_trajectory: tuple[tuple[float, ...], ...]
    data_provenance: str = "DEBUG"
    paper_eligible: bool = False
    training_mode: str = "debug"


def evaluate_torch_deterministic(
    actor: TorchSharedActor,
    environment_factory: Callable[[], HeavyPlatoonEnv],
    reward_function: DebugReward,
    steps: int,
) -> TorchEvaluationResult:
    if steps < 1:
        raise ValueError("evaluation steps must be positive")
    normalizer_before = actor.normalizer.state_dict()
    checksum_before = actor.checksum()
    env = environment_factory()
    nominal_history: list[tuple[tuple[float, float], ...]] = []
    provisional_history: list[tuple[tuple[float, float], ...]] = []
    executed_history: list[tuple[tuple[float, float], ...]] = []
    states: list[tuple[float, ...]] = [tuple(map(float, centralized_state_vector(env.centralized_training_state())))]
    total_reward = 0.0
    for _ in range(steps):
        observations = np.stack([local_observation_vector(item) for item in env.current_local_observations()])
        nominal = actor.deterministic_numpy(observations)
        env.nominal_controllers = [
            ConstantCommandController(float(action[0]), float(action[1])) for action in nominal
        ]
        result = env.step()
        nominal_history.append(tuple(tuple(map(float, action)) for action in nominal))
        provisional_history.append(tuple(tuple(map(float, record.provisional_action)) for record in result.logs))
        executed_history.append(tuple(tuple(map(float, record.final_action)) for record in result.logs))
        states.append(tuple(map(float, centralized_state_vector(env.centralized_training_state()))))
        total_reward += sum(reward_function(record).total for record in result.logs)
    if checksum_before != actor.checksum():
        raise AssertionError("deterministic evaluation modified actor parameters")
    normalizer_after = actor.normalizer.state_dict()
    if normalizer_before["count"] != normalizer_after["count"]:
        raise AssertionError("evaluation updated normalization sample count")
    for key in ("mean", "m2"):
        if not np.array_equal(normalizer_before[key], normalizer_after[key]):
            raise AssertionError("evaluation updated normalization statistics")
    return TorchEvaluationResult(
        float(total_reward), tuple(nominal_history), tuple(provisional_history),
        tuple(executed_history), tuple(states),
    )


def save_torch_checkpoint(
    path: Path,
    actor: TorchSharedActor,
    critic: TorchCentralizedCritic,
    trainer: TorchMAPPOTrainer,
    environment_step: int,
    rng: np.random.Generator,
    configuration: object,
    reproducibility: dict[str, object],
    qp_gradient_mode: str = "NONE",
) -> dict[str, object]:
    if qp_gradient_mode not in {"NONE", "DIFFQP", "STOP_GRADIENT"}:
        raise ValueError("unregistered QP gradient mode")
    payload: dict[str, object] = {
        "schema_version": 2,
        "backend": "pytorch_autograd",
        "actor_state_dict": actor.state_dict(),
        "actor_normalizer": actor.normalizer.state_dict(),
        "critic_state_dict": critic.state_dict(),
        "actor_optimizer_state": trainer.actor_optimizer.state_dict(),
        "critic_optimizer_state": trainer.critic_optimizer.state_dict(),
        "training_step": trainer.training_step,
        "environment_step": int(environment_step),
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "generator_state": rng.bit_generator.state,
        "torch_cpu_rng_state": torch.get_rng_state(),
        "torch_cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "config_hash": configuration_hash(configuration),
        "reproducibility": reproducibility,
        "library_versions": library_versions(),
        "data_provenance": "DEBUG",
        "paper_eligible": False,
        "training_mode": "debug",
        "diffqp_backward_active": qp_gradient_mode == "DIFFQP",
        "qp_gradient_mode": qp_gradient_mode,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return payload


def load_torch_checkpoint(
    path: Path,
    actor: TorchSharedActor,
    critic: TorchCentralizedCritic,
    trainer: TorchMAPPOTrainer,
    rng: np.random.Generator,
    expected_configuration: object | None = None,
) -> dict[str, object]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if payload.get("backend") != "pytorch_autograd":
        raise ValueError("checkpoint is not a PyTorch/autograd checkpoint")
    if payload.get("paper_eligible") is not False:
        raise ValueError("checkpoint violates DEBUG-only provenance")
    gradient_mode = payload.get("qp_gradient_mode", "NONE")
    if gradient_mode not in {"NONE", "DIFFQP", "STOP_GRADIENT"}:
        raise ValueError("checkpoint has an unregistered QP gradient mode")
    if bool(payload.get("diffqp_backward_active")) != (gradient_mode == "DIFFQP"):
        raise ValueError("checkpoint QP gradient-mode metadata is inconsistent")
    if expected_configuration is not None and payload["config_hash"] != configuration_hash(expected_configuration):
        raise ValueError("checkpoint configuration hash mismatch")
    actor.load_state_dict(payload["actor_state_dict"])
    actor.normalizer.load_state_dict(payload["actor_normalizer"])
    critic.load_state_dict(payload["critic_state_dict"])
    trainer.actor_optimizer.load_state_dict(payload["actor_optimizer_state"])
    trainer.critic_optimizer.load_state_dict(payload["critic_optimizer_state"])
    trainer.training_step = int(payload["training_step"])
    random.setstate(payload["python_random_state"])
    np.random.set_state(payload["numpy_random_state"])
    rng.bit_generator.state = payload["generator_state"]
    torch.set_rng_state(payload["torch_cpu_rng_state"])
    if torch.cuda.is_available() and payload["torch_cuda_rng_state_all"] is not None:
        torch.cuda.set_rng_state_all(payload["torch_cuda_rng_state_all"])
    return payload


@dataclass(frozen=True)
class TorchDebugTrainingSmokeResult:
    initial_actor_checksum: str
    final_actor_checksum: str
    initial_critic_checksum: str
    final_critic_checksum: str
    update_metrics: tuple[PPOUpdateMetrics, ...]
    rollout_returns: tuple[float, ...]
    checkpoint_path: str
    deterministic_max_discrepancy: float
    data_provenance: str = "DEBUG"
    paper_eligible: bool = False
    training_mode: str = "debug"


def _evaluation_max_discrepancy(a: TorchEvaluationResult, b: TorchEvaluationResult) -> float:
    arrays = (
        (a.nominal_actions, b.nominal_actions),
        (a.provisional_actions, b.provisional_actions),
        (a.executed_actions, b.executed_actions),
        (a.state_trajectory, b.state_trajectory),
    )
    return max(float(np.max(np.abs(np.asarray(left) - np.asarray(right)))) for left, right in arrays)


def run_torch_debug_training_smoke(
    checkpoint_path: Path, seed: int = 20260917
) -> TorchDebugTrainingSmokeResult:
    """Short N=3 DEBUG-only run through the complete frozen safety stack."""
    from envs.scenario import build_synthetic_debug_scenario

    rng, reproducibility = initialize_reproducibility(seed, seed + 1, seed + 2, seed + 3, seed + 4)
    scenario = build_synthetic_debug_scenario(3, "flat_steady")
    env = HeavyPlatoonEnv(scenario)
    observation_dim = local_observation_vector(env.current_local_observations()[0]).size
    state_dim = centralized_state_vector(env.centralized_training_state()).size
    upper = np.asarray([
        min(item.friction_force_limit_N for item in scenario.parameters),
        min(item.auxiliary_force_limit_N for item in scenario.parameters),
    ])
    actor = TorchSharedActor(observation_dim, 16, np.zeros(2), upper, rng)
    critic = TorchCentralizedCritic(state_dim, 16, 3, rng)
    config = PPOConfig(epochs=2, minibatch_size=6)
    trainer = TorchMAPPOTrainer(actor, critic, config)
    reward = DebugReward(
        DebugRewardWeights(0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01), 18.0, 65.0
    )
    initial_actor = actor.checksum()
    initial_critic = critic.checksum()
    metrics: list[PPOUpdateMetrics] = []
    returns: list[float] = []
    for _ in range(2):
        rollout, summary = collect_torch_rollout(env, actor, critic, reward, 3)
        metrics.append(trainer.update(rollout, rng))
        returns.append(summary.total_reward)
    final_actor = actor.checksum()
    final_critic = critic.checksum()
    if initial_actor == final_actor or initial_critic == final_critic:
        raise AssertionError("PyTorch DEBUG smoke did not update both networks")
    configuration = {"ppo": config, "scenario": "flat_steady", "N": 3, "horizon": 3}
    save_torch_checkpoint(checkpoint_path, actor, critic, trainer, 6, rng, configuration, reproducibility)

    clone_rng = np.random.default_rng(0)
    clone_actor = TorchSharedActor(observation_dim, 16, np.zeros(2), upper, clone_rng)
    clone_critic = TorchCentralizedCritic(state_dim, 16, 3, clone_rng)
    clone_trainer = TorchMAPPOTrainer(clone_actor, clone_critic, config)
    load_torch_checkpoint(checkpoint_path, clone_actor, clone_critic, clone_trainer, clone_rng, configuration)
    if clone_actor.checksum() != final_actor or clone_critic.checksum() != final_critic:
        raise AssertionError("PyTorch checkpoint parameter reload is not exact")
    factory = lambda: HeavyPlatoonEnv(build_synthetic_debug_scenario(3, "flat_steady"))
    first = evaluate_torch_deterministic(clone_actor, factory, reward, 2)
    second = evaluate_torch_deterministic(clone_actor, factory, reward, 2)
    discrepancy = _evaluation_max_discrepancy(first, second)
    if discrepancy != 0.0:
        raise AssertionError("PyTorch deterministic evaluation is not repeatable")
    return TorchDebugTrainingSmokeResult(
        initial_actor, final_actor, initial_critic, final_critic,
        tuple(metrics), tuple(returns), str(checkpoint_path), discrepancy,
    )


DIFFQP_BACKWARD_ACTIVE = False

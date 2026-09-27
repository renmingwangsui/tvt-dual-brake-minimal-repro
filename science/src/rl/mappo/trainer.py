"""NumPy MAPPO/PPO update for the Phase 2C-1 auditable DEBUG pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np

from .actor import SharedActor
from .buffer import RolloutBuffer
from .critic import CentralizedCritic


def ppo_probability_ratio(new_log_prob: np.ndarray, old_log_prob: np.ndarray) -> np.ndarray:
    difference = np.asarray(new_log_prob, dtype=np.float64) - np.asarray(old_log_prob, dtype=np.float64)
    if not np.all(np.isfinite(difference)):
        raise FloatingPointError("non-finite PPO log-probability difference")
    return np.exp(np.clip(difference, -60.0, 60.0))


def clipped_surrogate(
    ratio: np.ndarray, advantages: np.ndarray, clip_epsilon: float
) -> tuple[np.ndarray, np.ndarray]:
    ratio = np.asarray(ratio, dtype=np.float64)
    advantages = np.asarray(advantages, dtype=np.float64)
    unclipped = ratio * advantages
    clipped = np.clip(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
    return np.minimum(unclipped, clipped), np.abs(ratio - 1.0) > clip_epsilon


def global_gradient_norm(gradients: dict[str, np.ndarray]) -> float:
    return float(np.sqrt(sum(float(np.sum(np.asarray(value) ** 2)) for value in gradients.values())))


def clip_gradients(
    gradients: dict[str, np.ndarray], maximum_norm: float
) -> tuple[dict[str, np.ndarray], float]:
    norm = global_gradient_norm(gradients)
    if not np.isfinite(norm):
        raise FloatingPointError("non-finite gradient norm")
    factor = min(1.0, maximum_norm / max(norm, 1e-12))
    return {name: value * factor for name, value in gradients.items()}, norm


class AdamOptimizer:
    """Small explicit Adam implementation whose full state is checkpointable."""

    def __init__(
        self,
        parameters: dict[str, np.ndarray],
        learning_rate: float,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
    ) -> None:
        if learning_rate <= 0.0:
            raise ValueError("Adam learning rate must be positive")
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.step_count = 0
        self.first = {name: np.zeros_like(value) for name, value in parameters.items()}
        self.second = {name: np.zeros_like(value) for name, value in parameters.items()}

    def step(self, parameters: dict[str, np.ndarray], gradients: dict[str, np.ndarray]) -> None:
        if set(parameters) != set(gradients):
            raise ValueError("gradient/parameter keys do not match")
        self.step_count += 1
        for name, parameter in parameters.items():
            gradient = np.asarray(gradients[name], dtype=np.float64)
            if gradient.shape != parameter.shape or not np.all(np.isfinite(gradient)):
                raise FloatingPointError(f"invalid gradient for {name}")
            self.first[name] = self.beta1 * self.first[name] + (1.0 - self.beta1) * gradient
            self.second[name] = self.beta2 * self.second[name] + (1.0 - self.beta2) * gradient**2
            corrected_first = self.first[name] / (1.0 - self.beta1**self.step_count)
            corrected_second = self.second[name] / (1.0 - self.beta2**self.step_count)
            parameter -= self.learning_rate * corrected_first / (np.sqrt(corrected_second) + self.epsilon)
            if not np.all(np.isfinite(parameter)):
                raise FloatingPointError(f"non-finite parameter after Adam update: {name}")

    def state_dict(self) -> dict[str, object]:
        return {
            "learning_rate": self.learning_rate,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "epsilon": self.epsilon,
            "step_count": self.step_count,
            "first": {name: value.copy() for name, value in self.first.items()},
            "second": {name: value.copy() for name, value in self.second.items()},
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        self.learning_rate = float(state["learning_rate"])
        self.beta1 = float(state["beta1"])
        self.beta2 = float(state["beta2"])
        self.epsilon = float(state["epsilon"])
        self.step_count = int(state["step_count"])
        for name in self.first:
            self.first[name][...] = np.asarray(state["first"][name], dtype=np.float64)
            self.second[name][...] = np.asarray(state["second"][name], dtype=np.float64)


@dataclass(frozen=True)
class PPOConfig:
    clip_epsilon: float = 0.2
    entropy_coefficient: float = 0.001
    value_coefficient: float = 0.5
    actor_learning_rate: float = 3e-4
    critic_learning_rate: float = 5e-4
    maximum_gradient_norm: float = 1.0
    epochs: int = 2
    minibatch_size: int = 16
    data_provenance: str = "synthetic_debug"
    paper_eligible: bool = False
    training_mode: str = "debug"

    def __post_init__(self) -> None:
        if self.data_provenance != "synthetic_debug" or self.paper_eligible or self.training_mode != "debug":
            raise ValueError("Phase 2C-1 PPO configuration must remain DEBUG-only")
        if self.epochs < 1 or self.minibatch_size < 1 or self.maximum_gradient_norm <= 0.0:
            raise ValueError("invalid PPO update configuration")


@dataclass(frozen=True)
class PPOUpdateMetrics:
    policy_loss: float
    value_loss: float
    entropy: float
    approximate_kl: float
    clip_fraction: float
    actor_gradient_norm: float
    critic_gradient_norm: float
    minibatch_updates: int
    data_provenance: str = "synthetic_debug"
    paper_eligible: bool = False
    training_mode: str = "debug"


class MAPPOTrainer:
    def __init__(self, actor: SharedActor, critic: CentralizedCritic, config: PPOConfig) -> None:
        self.actor = actor
        self.critic = critic
        self.config = config
        # Exactly one actor optimizer is shared by samples from every truck.
        self.actor_optimizer = AdamOptimizer(actor.parameters, config.actor_learning_rate)
        self.critic_optimizer = AdamOptimizer(critic.parameters, config.critic_learning_rate)
        self.training_step = 0

    def update(self, buffer: RolloutBuffer, rng: np.random.Generator) -> PPOUpdateMetrics:
        if buffer.advantages is None:
            raise ValueError("GAE must be computed before PPO update")
        flat_advantages = buffer.advantages.reshape(-1)
        advantage_mean = float(np.mean(flat_advantages))
        advantage_std = float(np.std(flat_advantages))
        buffer.advantages = (buffer.advantages - advantage_mean) / (advantage_std + 1e-8)
        accumulated: list[tuple[float, ...]] = []
        for _epoch in range(self.config.epochs):
            for batch in buffer.minibatches(self.config.minibatch_size, rng):
                new_log_prob = self.actor.log_prob(batch.observations, batch.nominal_actions)
                ratio = ppo_probability_ratio(new_log_prob, batch.old_log_probs)
                surrogate, clipped_mask = clipped_surrogate(ratio, batch.advantages, self.config.clip_epsilon)
                policy_loss = -float(np.mean(surrogate))
                entropy = -float(np.mean(new_log_prob))

                unclipped = ratio * batch.advantages
                clipped = np.clip(
                    ratio, 1.0 - self.config.clip_epsilon, 1.0 + self.config.clip_epsilon
                ) * batch.advantages
                active_unclipped = unclipped <= clipped
                derivative = np.where(
                    active_unclipped,
                    -ratio * batch.advantages / len(ratio),
                    0.0,
                )
                # Entropy uses the sampled negative-log-probability estimator.
                derivative += self.config.entropy_coefficient / len(ratio)
                actor_gradients = self.actor.log_prob_gradients(
                    batch.observations, batch.nominal_actions, derivative
                )
                actor_gradients, actor_norm = clip_gradients(
                    actor_gradients, self.config.maximum_gradient_norm
                )
                self.actor_optimizer.step(self.actor.parameters, actor_gradients)

                value_loss, critic_gradients, _values = self.critic.loss_and_gradients(
                    batch.centralized_states, batch.agent_indices, batch.returns
                )
                critic_gradients = {
                    name: self.config.value_coefficient * value
                    for name, value in critic_gradients.items()
                }
                critic_gradients, critic_norm = clip_gradients(
                    critic_gradients, self.config.maximum_gradient_norm
                )
                self.critic_optimizer.step(self.critic.parameters, critic_gradients)
                approximate_kl = float(np.mean(batch.old_log_probs - new_log_prob))
                clip_fraction = float(np.mean(clipped_mask))
                fields = (
                    policy_loss,
                    value_loss,
                    entropy,
                    approximate_kl,
                    clip_fraction,
                    actor_norm,
                    critic_norm,
                )
                if not np.all(np.isfinite(fields)):
                    raise FloatingPointError("non-finite PPO loss or diagnostic")
                accumulated.append(fields)
                self.training_step += 1
        means = np.mean(np.asarray(accumulated, dtype=np.float64), axis=0)
        return PPOUpdateMetrics(*map(float, means), len(accumulated))


@dataclass(frozen=True)
class DebugTrainingSmokeResult:
    initial_actor_checksum: str
    final_actor_checksum: str
    initial_critic_checksum: str
    final_critic_checksum: str
    update_metrics: tuple[PPOUpdateMetrics, ...]
    rollout_returns: tuple[float, ...]
    checkpoint_path: str
    deterministic_repeatable: bool
    data_provenance: str = "synthetic_debug"
    paper_eligible: bool = False
    training_mode: str = "debug"


def run_debug_training_smoke(checkpoint_path: Path, seed: int = 20260917) -> DebugTrainingSmokeResult:
    """Run a short N=3 software-validation job, never a paper experiment."""
    from envs.heavy_platoon_env import HeavyPlatoonEnv
    from envs.scenario import build_synthetic_debug_scenario
    from .actor import local_observation_vector
    from .checkpoint import initialize_reproducibility, load_checkpoint, save_checkpoint
    from .critic import centralized_state_vector
    from .evaluation import evaluate_deterministic
    from .reward import DebugReward, DebugRewardWeights
    from .rollout import collect_rollout

    rng, reproducibility = initialize_reproducibility(seed, seed + 1, seed + 2, seed + 3, seed + 4)
    scenario = build_synthetic_debug_scenario(3, "flat_steady")
    env = HeavyPlatoonEnv(scenario)
    observation_dim = local_observation_vector(env.current_local_observations()[0]).size
    state_dim = centralized_state_vector(env.centralized_training_state()).size
    upper = np.asarray([
        min(item.friction_force_limit_N for item in scenario.parameters),
        min(item.auxiliary_force_limit_N for item in scenario.parameters),
    ])
    actor = SharedActor(observation_dim, 16, np.zeros(2), upper, rng)
    critic = CentralizedCritic(state_dim, 16, 3, rng)
    config = PPOConfig(epochs=2, minibatch_size=6)
    trainer = MAPPOTrainer(actor, critic, config)
    reward = DebugReward(
        DebugRewardWeights(0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01),
        target_speed_mps=18.0,
        target_gap_m=65.0,
    )
    initial_actor = actor.checksum()
    initial_critic = critic.checksum()
    metrics: list[PPOUpdateMetrics] = []
    returns: list[float] = []
    for _ in range(2):
        rollout, summary = collect_rollout(env, actor, critic, reward, 3, rng)
        metrics.append(trainer.update(rollout, rng))
        returns.append(summary.total_reward)
    final_actor = actor.checksum()
    final_critic = critic.checksum()
    if initial_actor == final_actor or initial_critic == final_critic:
        raise AssertionError("DEBUG training smoke did not update both networks")
    configuration = {"ppo": config, "scenario": "flat_steady", "N": 3, "horizon": 3}
    save_checkpoint(
        checkpoint_path,
        actor,
        critic,
        trainer.actor_optimizer,
        trainer.critic_optimizer,
        trainer.training_step,
        6,
        rng,
        configuration,
        reproducibility,
    )

    clone_rng = np.random.default_rng(0)
    clone_actor = SharedActor(observation_dim, 16, np.zeros(2), upper, clone_rng)
    clone_critic = CentralizedCritic(state_dim, 16, 3, clone_rng)
    clone_trainer = MAPPOTrainer(clone_actor, clone_critic, config)
    load_checkpoint(
        checkpoint_path,
        clone_actor,
        clone_critic,
        clone_trainer.actor_optimizer,
        clone_trainer.critic_optimizer,
        clone_rng,
        configuration,
    )
    if clone_actor.checksum() != final_actor or clone_critic.checksum() != final_critic:
        raise AssertionError("checkpoint parameter reload is not exact")
    factory = lambda: HeavyPlatoonEnv(build_synthetic_debug_scenario(3, "flat_steady"))
    evaluation_a = evaluate_deterministic(clone_actor, factory, reward, 2)
    evaluation_b = evaluate_deterministic(clone_actor, factory, reward, 2)
    repeatable = evaluation_a == evaluation_b
    if not repeatable:
        raise AssertionError("deterministic evaluation is not repeatable")
    return DebugTrainingSmokeResult(
        initial_actor, final_actor, initial_critic, final_critic,
        tuple(metrics), tuple(returns), str(checkpoint_path), repeatable,
    )

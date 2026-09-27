"""Parameter-shared decentralized NumPy actor with manual PPO gradients."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import numpy as np

from envs.observations import LocalActorObservation
from .distributions import SquashedGaussian, SquashedGaussianSample


def local_observation_vector(observation: LocalActorObservation) -> np.ndarray:
    return np.asarray([
        observation.own_speed_mps,
        observation.own_friction_force_N,
        observation.own_auxiliary_force_N,
        observation.own_temperature_K,
        observation.gap_m,
        observation.relative_speed_mps,
        observation.previous_friction_command_N,
        observation.previous_auxiliary_command_N,
        float(observation.gear_id),
        observation.message_age_s,
        observation.received_safety_intent,
    ], dtype=np.float64)


@dataclass
class RunningNormalizer:
    dimension: int
    count: float = 0.0
    mean: np.ndarray | None = None
    m2: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.dimension < 1:
            raise ValueError("normalizer dimension must be positive")
        if self.mean is None:
            self.mean = np.zeros(self.dimension, dtype=np.float64)
        if self.m2 is None:
            self.m2 = np.zeros(self.dimension, dtype=np.float64)

    def update(self, values: np.ndarray) -> None:
        batch = np.atleast_2d(np.asarray(values, dtype=np.float64))
        if batch.shape[1] != self.dimension:
            raise ValueError("normalizer input dimension mismatch")
        for row in batch:
            self.count += 1.0
            delta = row - self.mean
            self.mean += delta / self.count
            self.m2 += delta * (row - self.mean)

    @property
    def variance(self) -> np.ndarray:
        return self.m2 / max(1.0, self.count - 1.0)

    def normalize(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        return (array - self.mean) / np.sqrt(self.variance + 1e-8)

    def state_dict(self) -> dict[str, object]:
        return {"dimension": self.dimension, "count": self.count, "mean": self.mean.copy(), "m2": self.m2.copy()}

    def load_state_dict(self, state: dict[str, object]) -> None:
        if int(state["dimension"]) != self.dimension:
            raise ValueError("normalizer checkpoint dimension mismatch")
        self.count = float(state["count"])
        self.mean = np.asarray(state["mean"], dtype=np.float64).copy()
        self.m2 = np.asarray(state["m2"], dtype=np.float64).copy()


class SharedActor:
    def __init__(
        self,
        observation_dim: int,
        hidden_dim: int,
        action_lower: np.ndarray,
        action_upper: np.ndarray,
        rng: np.random.Generator,
    ) -> None:
        self.observation_dim = observation_dim
        self.hidden_dim = hidden_dim
        self.action_dim = len(action_lower)
        self.distribution = SquashedGaussian(action_lower, action_upper)
        self.normalizer = RunningNormalizer(observation_dim)
        self.parameters: dict[str, np.ndarray] = {
            "W1": rng.normal(0.0, 1.0 / np.sqrt(observation_dim), (observation_dim, hidden_dim)),
            "b1": np.zeros(hidden_dim),
            "Wmu": rng.normal(0.0, 0.05, (hidden_dim, self.action_dim)),
            "bmu": np.zeros(self.action_dim),
            "log_std": np.full(self.action_dim, -0.5),
        }

    def _forward(self, observations: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        raw_obs = np.atleast_2d(np.asarray(observations, dtype=np.float64))
        if raw_obs.shape[1] != self.observation_dim:
            raise ValueError("actor observation dimension mismatch")
        x = self.normalizer.normalize(raw_obs)
        hidden = np.tanh(x @ self.parameters["W1"] + self.parameters["b1"])
        mean = hidden @ self.parameters["Wmu"] + self.parameters["bmu"]
        log_std = np.broadcast_to(np.clip(self.parameters["log_std"], -5.0, 2.0), mean.shape)
        return x, hidden, mean

    def distribution_parameters(self, observations: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        _x, _hidden, mean = self._forward(observations)
        return mean, np.broadcast_to(np.clip(self.parameters["log_std"], -5.0, 2.0), mean.shape)

    def sample(self, observations: np.ndarray, rng: np.random.Generator) -> SquashedGaussianSample:
        mean, log_std = self.distribution_parameters(observations)
        return self.distribution.sample(mean, log_std, rng)

    def deterministic(self, observations: np.ndarray) -> np.ndarray:
        mean, _log_std = self.distribution_parameters(observations)
        return self.distribution.deterministic(mean)

    def log_prob(self, observations: np.ndarray, actions: np.ndarray) -> np.ndarray:
        mean, log_std = self.distribution_parameters(observations)
        return self.distribution.log_prob(mean, log_std, actions)

    def log_prob_gradients(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        loss_derivative_wrt_log_prob: np.ndarray,
    ) -> dict[str, np.ndarray]:
        x, hidden, mean = self._forward(observations)
        log_std = np.broadcast_to(np.clip(self.parameters["log_std"], -5.0, 2.0), mean.shape)
        raw_action, _ = self.distribution.inverse(actions)
        standardized_sq = ((raw_action - mean) / np.exp(log_std)) ** 2
        dlogp_dmean = (raw_action - mean) / np.exp(2.0 * log_std)
        dlogp_dlogstd = standardized_sq - 1.0
        coefficient = np.asarray(loss_derivative_wrt_log_prob, dtype=np.float64)[:, None]
        dmean = coefficient * dlogp_dmean
        dhidden = dmean @ self.parameters["Wmu"].T
        dpre = dhidden * (1.0 - hidden**2)
        return {
            "Wmu": hidden.T @ dmean,
            "bmu": np.sum(dmean, axis=0),
            "W1": x.T @ dpre,
            "b1": np.sum(dpre, axis=0),
            "log_std": np.sum(coefficient * dlogp_dlogstd, axis=0),
        }

    def state_dict(self) -> dict[str, object]:
        return {
            "parameters": {name: value.copy() for name, value in self.parameters.items()},
            "normalizer": self.normalizer.state_dict(),
            "observation_dim": self.observation_dim,
            "hidden_dim": self.hidden_dim,
            "lower": self.distribution.lower.copy(),
            "upper": self.distribution.upper.copy(),
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        if int(state["observation_dim"]) != self.observation_dim or int(state["hidden_dim"]) != self.hidden_dim:
            raise ValueError("actor checkpoint architecture mismatch")
        for name in self.parameters:
            self.parameters[name][...] = np.asarray(state["parameters"][name], dtype=np.float64)
        self.normalizer.load_state_dict(state["normalizer"])

    def checksum(self) -> str:
        digest = sha256()
        for name in sorted(self.parameters):
            digest.update(name.encode())
            digest.update(self.parameters[name].tobytes())
        return digest.hexdigest()

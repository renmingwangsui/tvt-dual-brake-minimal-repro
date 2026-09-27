"""Exact affine tanh-squashed diagonal Gaussian utilities."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


LOG_2PI = float(np.log(2.0 * np.pi))


@dataclass(frozen=True)
class SquashedGaussianSample:
    action: np.ndarray
    raw_action: np.ndarray
    log_prob: np.ndarray


class SquashedGaussian:
    def __init__(self, lower: np.ndarray, upper: np.ndarray, epsilon: float = 1e-7) -> None:
        self.lower = np.asarray(lower, dtype=np.float64)
        self.upper = np.asarray(upper, dtype=np.float64)
        if self.lower.shape != self.upper.shape or np.any(self.upper <= self.lower):
            raise ValueError("action bounds must have equal shapes and upper > lower")
        self.scale = (self.upper - self.lower) / 2.0
        self.shift = (self.upper + self.lower) / 2.0
        self.epsilon = epsilon

    def _validate(self, mean: np.ndarray, log_std: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mean = np.asarray(mean, dtype=np.float64)
        log_std = np.asarray(log_std, dtype=np.float64)
        if mean.shape[-1] != self.lower.size or log_std.shape[-1] != self.lower.size:
            raise ValueError("distribution/action dimensions do not match")
        if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(log_std)):
            raise FloatingPointError("non-finite Gaussian parameters")
        return mean, log_std

    def transform(self, raw_action: np.ndarray) -> np.ndarray:
        return self.shift + self.scale * np.tanh(np.asarray(raw_action, dtype=np.float64))

    def inverse(self, action: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        action = np.asarray(action, dtype=np.float64)
        normalized = (action - self.shift) / self.scale
        if np.any(normalized < -1.0 - 1e-10) or np.any(normalized > 1.0 + 1e-10):
            raise ValueError("action lies outside affine tanh bounds")
        clipped = np.clip(normalized, -1.0 + self.epsilon, 1.0 - self.epsilon)
        return np.arctanh(clipped), clipped

    def log_prob_from_raw(self, mean: np.ndarray, log_std: np.ndarray, raw_action: np.ndarray) -> np.ndarray:
        mean, log_std = self._validate(mean, log_std)
        raw = np.asarray(raw_action, dtype=np.float64)
        squashed = np.tanh(raw)
        variance_term = ((raw - mean) / np.exp(log_std)) ** 2
        normal = -0.5 * (variance_term + 2.0 * log_std + LOG_2PI)
        log_jacobian = np.log(self.scale) + np.log(np.maximum(self.epsilon, 1.0 - squashed**2))
        return np.sum(normal - log_jacobian, axis=-1)

    def log_prob(self, mean: np.ndarray, log_std: np.ndarray, action: np.ndarray) -> np.ndarray:
        raw, _squashed = self.inverse(action)
        return self.log_prob_from_raw(mean, log_std, raw)

    def sample(
        self, mean: np.ndarray, log_std: np.ndarray, rng: np.random.Generator
    ) -> SquashedGaussianSample:
        mean, log_std = self._validate(mean, log_std)
        raw = mean + np.exp(log_std) * rng.standard_normal(mean.shape)
        action = self.transform(raw)
        return SquashedGaussianSample(action, raw, self.log_prob_from_raw(mean, log_std, raw))

    def deterministic(self, mean: np.ndarray) -> np.ndarray:
        return self.transform(np.asarray(mean, dtype=np.float64))

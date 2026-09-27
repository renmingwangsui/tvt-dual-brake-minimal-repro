"""Generalized advantage estimation with distinct terminal and truncation masks."""
from __future__ import annotations

import numpy as np


def generalized_advantage_estimation(
    rewards: np.ndarray,
    values: np.ndarray,
    next_values: np.ndarray,
    terminals: np.ndarray,
    truncations: np.ndarray,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    arrays = [np.asarray(item, dtype=np.float64) for item in (rewards, values, next_values, terminals, truncations)]
    if len({item.shape for item in arrays}) != 1:
        raise ValueError("GAE arrays must have identical shapes")
    if not 0.0 <= gamma <= 1.0 or not 0.0 <= gae_lambda <= 1.0:
        raise ValueError("gamma and lambda must lie in [0,1]")
    rewards, values, next_values, terminals, truncations = arrays
    bootstrap_mask = 1.0 - terminals
    continuation_mask = 1.0 - np.maximum(terminals, truncations)
    deltas = rewards + gamma * next_values * bootstrap_mask - values
    advantages = np.zeros_like(rewards)
    running = np.zeros(rewards.shape[1:] or (), dtype=np.float64)
    for index in range(rewards.shape[0] - 1, -1, -1):
        running = deltas[index] + gamma * gae_lambda * continuation_mask[index] * running
        advantages[index] = running
    return advantages, advantages + values

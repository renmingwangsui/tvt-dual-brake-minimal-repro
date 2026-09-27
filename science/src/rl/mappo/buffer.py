"""MAPPO rollout storage preserving nominal, provisional and executed actions."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .gae import generalized_advantage_estimation


@dataclass(frozen=True)
class RolloutBatch:
    observations: np.ndarray
    centralized_states: np.ndarray
    agent_indices: np.ndarray
    nominal_actions: np.ndarray
    provisional_actions: np.ndarray
    executed_actions: np.ndarray
    old_log_probs: np.ndarray
    advantages: np.ndarray
    returns: np.ndarray
    supervisor_modes: np.ndarray
    intervention_norms: np.ndarray
    rho: np.ndarray
    rho_H_cert: np.ndarray


class RolloutBuffer:
    def __init__(self, controlled_truck_count: int) -> None:
        self.n = controlled_truck_count
        self.steps: list[dict[str, object]] = []
        self.advantages: np.ndarray | None = None
        self.returns: np.ndarray | None = None

    def add(self, **record: object) -> None:
        required = {
            "observations", "centralized_state", "nominal_actions", "old_log_probs",
            "provisional_actions", "executed_actions", "rewards", "terminals",
            "truncations", "values", "next_values", "supervisor_modes",
            "intervention_norms", "rho", "rho_H_cert", "critical_vehicle_ids",
            "critical_components", "message_validity",
        }
        missing = required.difference(record)
        if missing:
            raise ValueError("rollout record missing: " + ", ".join(sorted(missing)))
        if np.asarray(record["observations"]).shape[0] != self.n:
            raise ValueError("rollout step must contain exactly N actor observations")
        self.steps.append(record)

    def compute_gae(self, gamma: float, gae_lambda: float) -> tuple[np.ndarray, np.ndarray]:
        if not self.steps:
            raise ValueError("cannot compute GAE on an empty buffer")
        stack = lambda key: np.stack([np.asarray(step[key]) for step in self.steps])
        self.advantages, self.returns = generalized_advantage_estimation(
            stack("rewards"), stack("values"), stack("next_values"),
            stack("terminals"), stack("truncations"), gamma, gae_lambda,
        )
        return self.advantages, self.returns

    def flattened(self) -> RolloutBatch:
        if self.advantages is None or self.returns is None:
            raise ValueError("compute GAE before requesting batches")
        time_steps = len(self.steps)
        stack = lambda key: np.stack([np.asarray(step[key]) for step in self.steps])
        central = stack("centralized_state")
        repeated_central = np.repeat(central[:, None, :], self.n, axis=1)
        flatten = lambda value: value.reshape((time_steps * self.n,) + value.shape[2:])
        return RolloutBatch(
            observations=flatten(stack("observations")),
            centralized_states=flatten(repeated_central),
            agent_indices=np.tile(np.arange(self.n), time_steps),
            nominal_actions=flatten(stack("nominal_actions")),
            provisional_actions=flatten(stack("provisional_actions")),
            executed_actions=flatten(stack("executed_actions")),
            old_log_probs=stack("old_log_probs").reshape(-1),
            advantages=self.advantages.reshape(-1),
            returns=self.returns.reshape(-1),
            supervisor_modes=stack("supervisor_modes").reshape(-1),
            intervention_norms=stack("intervention_norms").reshape(-1),
            rho=stack("rho").reshape(-1),
            rho_H_cert=stack("rho_H_cert").reshape(-1),
        )

    def minibatches(self, batch_size: int, rng: np.random.Generator) -> tuple[RolloutBatch, ...]:
        batch = self.flattened()
        total = len(batch.old_log_probs)
        indices = rng.permutation(total)
        result: list[RolloutBatch] = []
        for start in range(0, total, batch_size):
            selected = indices[start : start + batch_size]
            result.append(RolloutBatch(**{
                name: getattr(batch, name)[selected]
                for name in batch.__dataclass_fields__
            }))
        return tuple(result)

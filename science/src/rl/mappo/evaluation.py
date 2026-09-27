"""Deterministic, normalization-frozen evaluation for DEBUG policies."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
import numpy as np

from controllers.debug_nominal import ConstantCommandController
from envs.heavy_platoon_env import HeavyPlatoonEnv
from .actor import SharedActor, local_observation_vector
from .reward import DebugReward


@dataclass(frozen=True)
class EvaluationResult:
    total_reward: float
    actions: tuple[tuple[tuple[float, float], ...], ...]
    executed_actions: tuple[tuple[tuple[float, float], ...], ...]
    safety_cycles: int
    data_provenance: str = "synthetic_debug"
    paper_eligible: bool = False
    training_mode: str = "debug"


def evaluate_deterministic(
    actor: SharedActor,
    environment_factory: Callable[[], HeavyPlatoonEnv],
    reward_function: DebugReward,
    steps: int,
) -> EvaluationResult:
    if steps < 1:
        raise ValueError("evaluation steps must be positive")
    normalizer_before = actor.normalizer.state_dict()
    parameters_before = actor.checksum()
    env = environment_factory()
    action_history: list[tuple[tuple[float, float], ...]] = []
    executed_history: list[tuple[tuple[float, float], ...]] = []
    total_reward = 0.0
    for _ in range(steps):
        observations = np.stack([
            local_observation_vector(item) for item in env.current_local_observations()
        ])
        actions = actor.deterministic(observations)
        env.nominal_controllers = [
            ConstantCommandController(float(action[0]), float(action[1])) for action in actions
        ]
        result = env.step()
        action_history.append(tuple(tuple(map(float, action)) for action in actions))
        executed_history.append(tuple(record.final_action for record in result.logs))
        total_reward += sum(reward_function(record).total for record in result.logs)
    normalizer_after = actor.normalizer.state_dict()
    if parameters_before != actor.checksum():
        raise AssertionError("deterministic evaluation modified actor parameters")
    for name in ("mean", "m2"):
        if not np.array_equal(normalizer_before[name], normalizer_after[name]):
            raise AssertionError("evaluation updated running normalization statistics")
    if normalizer_before["count"] != normalizer_after["count"]:
        raise AssertionError("evaluation updated normalization sample count")
    return EvaluationResult(
        float(total_reward), tuple(action_history), tuple(executed_history), steps * env.controlled_truck_count
    )

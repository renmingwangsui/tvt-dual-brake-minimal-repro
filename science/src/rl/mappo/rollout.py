"""Rollout collection through the complete non-differentiable Phase 2B safety path."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from controllers.debug_nominal import ConstantCommandController
from envs.heavy_platoon_env import HeavyPlatoonEnv
from .actor import SharedActor, local_observation_vector
from .buffer import RolloutBuffer
from .critic import CentralizedCritic, centralized_state_vector
from .reward import DebugReward


@dataclass(frozen=True)
class RolloutSummary:
    steps: int
    safety_cycles: int
    total_reward: float
    nominal_executed_difference_count: int
    data_provenance: str = "synthetic_debug"
    paper_eligible: bool = False
    training_mode: str = "debug"


def collect_rollout(
    env: HeavyPlatoonEnv,
    actor: SharedActor,
    critic: CentralizedCritic,
    reward_function: DebugReward,
    horizon: int,
    rng: np.random.Generator,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    update_normalization: bool = True,
) -> tuple[RolloutBuffer, RolloutSummary]:
    if horizon < 1:
        raise ValueError("rollout horizon must be positive")
    if critic.controlled_truck_count != env.controlled_truck_count:
        raise ValueError("critic registered N does not match environment N")
    buffer = RolloutBuffer(env.controlled_truck_count)
    total_reward = 0.0
    distinct = 0
    if update_normalization:
        initial_observations = np.stack([
            local_observation_vector(item) for item in env.current_local_observations()
        ])
        actor.normalizer.update(initial_observations)
    for step_index in range(horizon):
        local_objects = env.current_local_observations()
        observations = np.stack([local_observation_vector(item) for item in local_objects])
        central_state = centralized_state_vector(env.centralized_training_state())
        values = critic.forward(central_state)[0]
        sample = actor.sample(observations, rng)

        # The actor output is detached into ordinary scalar controllers.  The
        # existing Phase 2B QP/prediction/supervisor/re-verification path runs
        # unmodified and supplies the actual commands used by the simulator.
        env.nominal_controllers = [
            ConstantCommandController(float(action[0]), float(action[1]))
            for action in sample.action
        ]
        result = env.step()
        next_state = centralized_state_vector(env.centralized_training_state())
        next_values = critic.forward(next_state)[0]
        provisional = np.asarray([record.provisional_action for record in result.logs], dtype=np.float64)
        executed = np.asarray([record.final_action for record in result.logs], dtype=np.float64)
        nominal = np.asarray([record.nominal_action for record in result.logs], dtype=np.float64)
        if not np.allclose(nominal, sample.action, rtol=0.0, atol=1e-9):
            raise AssertionError("safety controller did not receive the sampled nominal action")
        rewards = np.asarray([reward_function(record).total for record in result.logs])
        total_reward += float(np.sum(rewards))
        distinct += int(np.sum(np.any(np.abs(nominal - executed) > 1e-9, axis=1)))
        truncations = np.full(env.controlled_truck_count, step_index == horizon - 1)
        terminals = np.zeros(env.controlled_truck_count, dtype=bool)
        buffer.add(
            observations=observations,
            centralized_state=central_state,
            nominal_actions=sample.action.copy(),
            old_log_probs=sample.log_prob.copy(),
            provisional_actions=provisional,
            executed_actions=executed,
            rewards=rewards,
            terminals=terminals,
            truncations=truncations,
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
    buffer.compute_gae(gamma, gae_lambda)
    return buffer, RolloutSummary(horizon, horizon * env.controlled_truck_count, total_reward, distinct)

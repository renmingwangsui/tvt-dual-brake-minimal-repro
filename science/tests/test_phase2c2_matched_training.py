"""Matched closed-loop forward execution and N=3 two-mode DEBUG training."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
import tempfile

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from controllers.debug_nominal import ConstantCommandController  # noqa: E402
from envs.heavy_platoon_env import HeavyPlatoonEnv  # noqa: E402
from envs.scenario import build_synthetic_debug_scenario  # noqa: E402
from rl.mappo.actor import local_observation_vector  # noqa: E402
from rl.mappo.critic import centralized_state_vector  # noqa: E402
from rl.mappo.diffqp_training import run_phase2c2_debug_training_smoke  # noqa: E402
from rl.mappo.torch_backend import TorchSharedActor  # noqa: E402
from safety.diffqp import CanonicalSafetyQP, QPGradientMode, solve_differentiable_qp  # noqa: E402


@dataclass(frozen=True)
class FrozenTrajectory:
    u_rl: np.ndarray
    u_star0: np.ndarray
    u_final: np.ndarray
    states: np.ndarray
    rho: np.ndarray
    rho_h: np.ndarray
    supervisor_modes: tuple[str, ...]
    qp_statuses: tuple[str, ...]
    wrapper_to_controller_error_max: float
    ppo_likelihood_finite: bool


scenario = build_synthetic_debug_scenario(3, "flat_steady")
probe_env = HeavyPlatoonEnv(scenario)
obs_dim = local_observation_vector(probe_env.current_local_observations()[0]).size
upper = np.asarray([
    min(item.friction_force_limit_N for item in scenario.parameters),
    min(item.auxiliary_force_limit_N for item in scenario.parameters),
])
reference_actor = TorchSharedActor(obs_dim, 16, np.zeros(2), upper, np.random.default_rng(501))
initial_observations = np.stack([
    local_observation_vector(item) for item in probe_env.current_local_observations()
])
reference_actor.normalizer.update(initial_observations)
checkpoint_state = {name: value.detach().clone() for name, value in reference_actor.state_dict().items()}
checkpoint_normalizer = reference_actor.normalizer.state_dict()


def run_frozen_trajectory(mode: QPGradientMode) -> FrozenTrajectory:
    local_scenario = build_synthetic_debug_scenario(3, "flat_steady")
    env = HeavyPlatoonEnv(local_scenario)
    actor = TorchSharedActor(obs_dim, 16, np.zeros(2), upper, np.random.default_rng(999))
    actor.load_state_dict(checkpoint_state)
    actor.normalizer.load_state_dict(checkpoint_normalizer)
    nominal_history = []
    provisional_history = []
    final_history = []
    state_history = [centralized_state_vector(env.centralized_training_state())]
    rho_history = []
    rho_h_history = []
    supervisor_modes = []
    qp_statuses = []
    wrapper_errors = []
    likelihood_values = []
    for _ in range(3):
        observations = np.stack([
            local_observation_vector(item) for item in env.current_local_observations()
        ])
        nominal = actor.deterministic_numpy(observations)
        likelihood_values.extend(actor.log_prob_tensor(observations, nominal).detach().numpy())
        env.nominal_controllers = [
            ConstantCommandController(float(action[0]), float(action[1])) for action in nominal
        ]
        step = env.step()
        wrapped = []
        for index, controller_result in enumerate(step.controller_results):
            nominal_tensor = torch.tensor(nominal[index], dtype=torch.float64, requires_grad=True)
            result = solve_differentiable_qp(
                nominal_tensor,
                CanonicalSafetyQP(controller_result.hard_rows),
                mode,
            )
            wrapped.append(result.action.detach().numpy())
            wrapper_errors.extend(np.abs(
                result.action.detach().numpy()
                - np.asarray(controller_result.two_pass.provisional_command, dtype=np.float64)
            ))
        nominal_history.append(np.asarray([record.nominal_action for record in step.logs]))
        provisional_history.append(np.asarray(wrapped))
        final_history.append(np.asarray([record.final_action for record in step.logs]))
        state_history.append(centralized_state_vector(env.centralized_training_state()))
        rho_history.append(np.asarray([record.rho for record in step.logs]))
        rho_h_history.append(np.asarray([record.rho_H_cert for record in step.logs]))
        supervisor_modes.extend(record.supervisor_mode for record in step.logs)
        qp_statuses.extend(result.qp_status for result in step.controller_results)
    assert "all_commands_verified" in env.execution_trace
    assert all(record.final_residuals_valid for record in env.logs)
    return FrozenTrajectory(
        u_rl=np.asarray(nominal_history),
        u_star0=np.asarray(provisional_history),
        u_final=np.asarray(final_history),
        states=np.asarray(state_history),
        rho=np.asarray(rho_history),
        rho_h=np.asarray(rho_h_history),
        supervisor_modes=tuple(supervisor_modes),
        qp_statuses=tuple(qp_statuses),
        wrapper_to_controller_error_max=float(max(wrapper_errors)),
        ppo_likelihood_finite=bool(np.all(np.isfinite(likelihood_values))),
    )


diff_trajectory = run_frozen_trajectory(QPGradientMode.DIFFQP)
stop_trajectory = run_frozen_trajectory(QPGradientMode.STOP_GRADIENT)
forward_errors = np.concatenate([
    np.abs(diff_trajectory.u_rl - stop_trajectory.u_rl).reshape(-1),
    np.abs(diff_trajectory.u_star0 - stop_trajectory.u_star0).reshape(-1),
    np.abs(diff_trajectory.u_final - stop_trajectory.u_final).reshape(-1),
    np.abs(diff_trajectory.rho - stop_trajectory.rho).reshape(-1),
    np.abs(diff_trajectory.rho_h - stop_trajectory.rho_h).reshape(-1),
])
state_errors = np.abs(diff_trajectory.states - stop_trajectory.states)
assert np.max(forward_errors) == 0.0
assert np.max(state_errors) == 0.0
assert diff_trajectory.supervisor_modes == stop_trajectory.supervisor_modes
assert diff_trajectory.qp_statuses == stop_trajectory.qp_statuses
assert diff_trajectory.wrapper_to_controller_error_max <= 1e-9
assert stop_trajectory.wrapper_to_controller_error_max <= 1e-9
assert diff_trajectory.ppo_likelihood_finite and stop_trajectory.ppo_likelihood_finite


with tempfile.TemporaryDirectory() as directory:
    directory_path = Path(directory)
    diff_checkpoint = directory_path / "diffqp_debug.pt"
    stop_checkpoint = directory_path / "stop_gradient_debug.pt"
    diff_training = run_phase2c2_debug_training_smoke(
        QPGradientMode.DIFFQP, diff_checkpoint, seed=20260923
    )
    stop_training = run_phase2c2_debug_training_smoke(
        QPGradientMode.STOP_GRADIENT, stop_checkpoint, seed=20260923
    )
    diff_payload = torch.load(diff_checkpoint, map_location="cpu", weights_only=False)
    stop_payload = torch.load(stop_checkpoint, map_location="cpu", weights_only=False)
    assert diff_payload["qp_gradient_mode"] == "DIFFQP"
    assert diff_payload["diffqp_backward_active"] is True
    assert stop_payload["qp_gradient_mode"] == "STOP_GRADIENT"
    assert stop_payload["diffqp_backward_active"] is False
    assert diff_payload["config_hash"] == stop_payload["config_hash"]
    assert diff_checkpoint.with_suffix(".qp_debug.json").is_file()
    assert stop_checkpoint.with_suffix(".qp_debug.json").is_file()

assert diff_training.data_provenance == "DEBUG" and diff_training.paper_eligible is False
assert stop_training.data_provenance == "DEBUG" and stop_training.paper_eligible is False
assert diff_training.safety_stack_verified and stop_training.safety_stack_verified
assert diff_training.qp_actions == stop_training.qp_actions
assert diff_training.qp_intervention_max > 0.0
assert stop_training.qp_intervention_max == diff_training.qp_intervention_max
assert np.isfinite(diff_training.qp_mediated_gradient_norm)
assert diff_training.qp_mediated_gradient_norm > 0.0
assert stop_training.qp_mediated_gradient_norm == 0.0
assert diff_training.qp_actor_parameter_gradient_norm > 0.0
assert stop_training.qp_actor_parameter_gradient_norm == 0.0
assert diff_training.actor_gradient_norm > 0.0
assert stop_training.actor_gradient_norm > 0.0
assert diff_training.fallback_count == 0 and stop_training.fallback_count == 0

print(
    "PASS: Phase 2C-2 matched frozen trajectory and N=3 DEBUG training "
    f"forward_action_max={np.max(forward_errors):.3e} state_max={np.max(state_errors):.3e} "
    f"controller_wrapper_max={max(diff_trajectory.wrapper_to_controller_error_max, stop_trajectory.wrapper_to_controller_error_max):.3e} "
    f"diffqp_qp_gradient_norm={diff_training.qp_mediated_gradient_norm:.6e} "
    f"stopgrad_qp_gradient_norm={stop_training.qp_mediated_gradient_norm:.6e} "
    f"diffqp_qp_actor_gradient_norm={diff_training.qp_actor_parameter_gradient_norm:.6e} "
    f"diffqp_actor_gradient_norm={diff_training.actor_gradient_norm:.6e} "
    f"stopgrad_actor_gradient_norm={stop_training.actor_gradient_norm:.6e}"
)

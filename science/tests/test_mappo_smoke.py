"""N=3 DEBUG safety-forward training smoke and executed-transition regression."""
from __future__ import annotations

from pathlib import Path
import json
import sys
import tempfile
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from envs.heavy_platoon_env import HeavyPlatoonEnv  # noqa: E402
from envs.scenario import build_synthetic_debug_scenario  # noqa: E402
from rl.mappo import DIFFQP_BACKWARD_ACTIVE  # noqa: E402
from rl.mappo.actor import SharedActor, local_observation_vector  # noqa: E402
from rl.mappo.critic import CentralizedCritic, centralized_state_vector  # noqa: E402
from rl.mappo.reward import DebugReward, DebugRewardWeights  # noqa: E402
from rl.mappo.rollout import collect_rollout  # noqa: E402
from rl.mappo.trainer import run_debug_training_smoke  # noqa: E402
from rl.mappo.torch_backend import (  # noqa: E402
    TorchCentralizedCritic,
    TorchSharedActor,
    collect_torch_rollout,
    run_torch_debug_training_smoke,
)
from simulator.truck_dynamics import analytical_first_order_response  # noqa: E402

assert DIFFQP_BACKWARD_ACTIVE is False
rng = np.random.default_rng(91)
scenario = build_synthetic_debug_scenario(3, "flat_steady")
env = HeavyPlatoonEnv(scenario)
initial_states = tuple(env.states)
obs_dim = local_observation_vector(env.current_local_observations()[0]).size
state_dim = centralized_state_vector(env.centralized_training_state()).size
actor = SharedActor(obs_dim, 10, np.zeros(2), np.asarray([120_000.0, 80_000.0]), rng)
critic = CentralizedCritic(state_dim, 10, 3, rng)
reward = DebugReward(DebugRewardWeights(0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01), 18.0, 65.0)
buffer, summary = collect_rollout(env, actor, critic, reward, 1, rng)
step = buffer.steps[0]
assert summary.safety_cycles == 3 and summary.nominal_executed_difference_count > 0
assert len(env.execution_trace) > 0 and "all_commands_verified" in env.execution_trace
assert all(record.final_residuals_valid for record in env.logs[-3:])

for index, (initial, final_state, parameters) in enumerate(zip(initial_states, env.states, scenario.parameters)):
    executed = float(step["executed_actions"][index, 0])
    nominal = float(step["nominal_actions"][index, 0])
    expected_executed = analytical_first_order_response(initial.friction_force_N, executed, scenario.control_step_s, parameters.tau_f_s)
    expected_nominal = analytical_first_order_response(initial.friction_force_N, nominal, scenario.control_step_s, parameters.tau_f_s)
    assert abs(final_state.friction_force_N - expected_executed) < 0.05
    if abs(executed - nominal) > 1e-9:
        assert abs(final_state.friction_force_N - expected_executed) < abs(final_state.friction_force_N - expected_nominal)

artifact_dir = ROOT / "artifacts" / "debug"
artifact_dir.mkdir(parents=True, exist_ok=True)
checkpoint = artifact_dir / "phase2c1_mappo_smoke.pkl"
result = run_debug_training_smoke(checkpoint)
assert result.data_provenance == "synthetic_debug" and result.paper_eligible is False
assert result.initial_actor_checksum != result.final_actor_checksum
assert result.initial_critic_checksum != result.final_critic_checksum
assert result.deterministic_repeatable and checkpoint.is_file()
assert len(result.update_metrics) == 2 and len(result.rollout_returns) == 2
assert all(np.all(np.isfinite([
    metric.policy_loss, metric.value_loss, metric.entropy, metric.approximate_kl,
    metric.clip_fraction, metric.actor_gradient_norm, metric.critic_gradient_norm,
])) for metric in result.update_metrics)

report = {
    "data_provenance": result.data_provenance,
    "paper_eligible": result.paper_eligible,
    "training_mode": result.training_mode,
    "diffqp_backward_active": False,
    "initial_actor_checksum": result.initial_actor_checksum,
    "final_actor_checksum": result.final_actor_checksum,
    "initial_critic_checksum": result.initial_critic_checksum,
    "final_critic_checksum": result.final_critic_checksum,
    "updates": len(result.update_metrics),
    "returns_finite": bool(np.all(np.isfinite(result.rollout_returns))),
    "claim": "software-validation smoke only; no convergence or performance claim",
}
(artifact_dir / "phase2c1_mappo_smoke.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

# PyTorch rollout keeps u_RL likelihood separate while the unchanged safety
# stack produces u_star0 and u_final for the environment transition.
torch_env = HeavyPlatoonEnv(build_synthetic_debug_scenario(3, "flat_steady"))
torch_initial_states = tuple(torch_env.states)
torch_actor = TorchSharedActor.from_numpy(actor)
torch_critic = TorchCentralizedCritic.from_numpy(critic)
torch_buffer, torch_summary = collect_torch_rollout(
    torch_env, torch_actor, torch_critic, reward, 1
)
torch_step = torch_buffer.steps[0]
recomputed_log_prob = torch_actor.log_prob_tensor(
    torch_step["observations"], torch_step["nominal_actions"]
).detach().numpy()
assert np.allclose(recomputed_log_prob, torch_step["old_log_probs"], rtol=0.0, atol=1e-12)
assert torch_summary.safety_cycles == 3
assert torch_summary.nominal_executed_difference_count > 0
assert "all_commands_verified" in torch_env.execution_trace
assert all(record.final_residuals_valid for record in torch_env.logs[-3:])
for index, (initial, final_state, parameters) in enumerate(
    zip(torch_initial_states, torch_env.states, torch_env.scenario.parameters)
):
    u_rl = float(torch_step["nominal_actions"][index, 0])
    u_final = float(torch_step["executed_actions"][index, 0])
    expected_final = analytical_first_order_response(
        initial.friction_force_N, u_final, torch_env.scenario.control_step_s, parameters.tau_f_s
    )
    expected_nominal = analytical_first_order_response(
        initial.friction_force_N, u_rl, torch_env.scenario.control_step_s, parameters.tau_f_s
    )
    assert abs(final_state.friction_force_N - expected_final) < 0.05
    if abs(u_final - u_rl) > 1e-9:
        assert abs(final_state.friction_force_N - expected_final) < abs(
            final_state.friction_force_N - expected_nominal
        )

with tempfile.TemporaryDirectory() as directory:
    torch_checkpoint = Path(directory) / "phase2c15_torch_smoke.pt"
    torch_result = run_torch_debug_training_smoke(torch_checkpoint)
    assert torch_checkpoint.is_file()
assert torch_result.data_provenance == "DEBUG"
assert torch_result.paper_eligible is False
assert torch_result.initial_actor_checksum != torch_result.final_actor_checksum
assert torch_result.initial_critic_checksum != torch_result.final_critic_checksum
assert torch_result.deterministic_max_discrepancy == 0.0
assert len(torch_result.update_metrics) == 2 and len(torch_result.rollout_returns) == 2
assert all(metric.actor_gradient_norm > 0.0 and metric.critic_gradient_norm > 0.0
           for metric in torch_result.update_metrics)

print(
    "PASS: N=3 DEBUG NumPy reference and PyTorch-autograd training through complete safety stack; "
    "paper_eligible=false deterministic_max_discrepancy=0.000e+00"
)

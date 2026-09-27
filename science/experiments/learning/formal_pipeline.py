"""Frozen matched DiffQP/StopGradient learning pipeline.

The module is deliberately separate from the audited forward controller.  It
consumes the controller's exact hard rows and never alters physical, QP,
certificate, supervisor, or reward configuration after execution starts.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from hashlib import sha256
import itertools
import json
import math
import os
from pathlib import Path
import platform
import random
import resource
import shutil
import statistics
import subprocess
import time
from typing import Any, Iterable, Sequence

import numpy as np
import torch

from controllers.debug_nominal import ConstantCommandController
from envs.heavy_platoon_env import HeavyPlatoonEnv, Phase2BLogRecord
from envs.leader_profile import EmergencyBrakingPulseLeader
from envs.road_profile import PiecewiseLinearGrade
from experiments.model_based.paper_pipeline import (
    EXPECTED_CONFIG_HASH,
    FROZEN_CONFIG,
    build_controller,
    build_scenario,
    build_vehicle,
    make_env,
    records_by_id,
    validate_frozen_inputs,
)
from network.v2v_channel import ChannelConfig
from rl.mappo.actor import local_observation_vector
from rl.mappo.buffer import RolloutBatch, RolloutBuffer
from rl.mappo.critic import centralized_state_vector
from rl.mappo.torch_backend import (
    TORCH_DTYPE,
    TorchCentralizedCritic,
    TorchSharedActor,
    torch_generalized_advantage_estimation,
    torch_ppo_losses,
)
from safety.diffqp import CanonicalSafetyQP, QPGradientMode, solve_differentiable_qp


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = ROOT / "configs/learning/formal_learning_protocol_v1.yaml"
EXPECTED_PROTOCOL_HASH = "9e4efec951d171460a7d3f48c2ba492a53297215b6b37041cd028a875e0e333e"
RESULT_ROOT = ROOT / "results/formal_learning"
INVALIDATED_RESULT_ROOT = ROOT / "results/formal_learning_invalidated_wallclock_bug"
EXPECTED_INVALIDATED_MANIFEST_HASH = "a497709905fe4e33fb42141b57a0734ead37b17fbaa7251197d0d12fc548bb03"
FORWARD_CONTROLLER_FILES = (
    "src/safety_core.py",
    "src/controllers/integrated_safety_controller.py",
    "src/controllers/paper_safety_controller.py",
    "src/safety/diffqp.py",
)


def file_hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def canonical_hash(value: Any) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def dump_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")


def load_protocol() -> dict[str, Any]:
    if file_hash(PROTOCOL_PATH) != EXPECTED_PROTOCOL_HASH:
        raise RuntimeError("formal protocol hash changed after pre-registration")
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if protocol.get("phase") != "2C-3" or protocol.get("frozen_before_final_results") is not True:
        raise RuntimeError("formal protocol is not frozen")
    seeds = list(map(int, protocol["training_seeds"]))
    if len(seeds) < 10 or len(seeds) != len(set(seeds)):
        raise RuntimeError("at least ten unique registered training seeds are required")
    if protocol["primary_methods"] != ["SAFE-MAPPO-DIFFQP", "SAFE-MAPPO-STOPGRAD"]:
        raise RuntimeError("primary matched methods changed")
    if protocol["forward_qp"] != "accepted exact two-dimensional (u_f,u_a) hard projection; no jerk-slack decision":
        raise RuntimeError("canonical forward-QP declaration changed")
    if int(protocol["batch_size_agent_transitions"]) != int(protocol["rollout_length"]) * int(protocol["controlled_truck_count"]):
        raise RuntimeError("registered batch size does not match rollout x agents")
    if int(protocol["checkpoint_updates"][-1]) != int(protocol["number_of_updates"]):
        raise RuntimeError("final checkpoint is not the final update")
    return protocol


def git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def require_clean_code_tree() -> None:
    completed = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=ROOT, text=True, capture_output=True, check=True,
    )
    if completed.stdout.strip():
        raise RuntimeError("formal execution requires a clean tracked code tree")


def code_snapshot_hash() -> str:
    files = sorted(
        path for directory in ("src", "experiments", "analysis", "scripts")
        for path in (ROOT / directory).rglob("*.py")
    )
    return canonical_hash({str(path.relative_to(ROOT)).replace("\\", "/"): file_hash(path) for path in files})


def forward_controller_hash() -> str:
    return canonical_hash({name: file_hash(ROOT / name) for name in FORWARD_CONTROLLER_FILES})


def frozen_evidence_hash() -> str:
    manifests = sorted((ROOT / "results/paper_candidate").glob("*/result_manifest.json"))
    if not manifests:
        raise RuntimeError("frozen Phase 2M evidence manifests are missing")
    return canonical_hash({str(path.relative_to(ROOT)): file_hash(path) for path in manifests})


def formal_actor_checksum(actor: TorchSharedActor) -> str:
    digest = sha256(actor.checksum().encode())
    normalizer = actor.normalizer.state_dict()
    digest.update(str(normalizer["dimension"]).encode())
    digest.update(repr(float(normalizer["count"])).encode())
    digest.update(np.asarray(normalizer["mean"], dtype=np.float64).tobytes())
    digest.update(np.asarray(normalizer["m2"], dtype=np.float64).tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class FormalPPOConfig:
    clip_epsilon: float
    entropy_coefficient: float
    value_coefficient: float
    actor_learning_rate: float
    critic_learning_rate: float
    maximum_gradient_norm: float
    epochs: int
    minibatch_size: int


@dataclass(frozen=True)
class FormalReward:
    protocol: dict[str, Any]
    action_upper: np.ndarray

    def terms(self, record: Phase2BLogRecord) -> dict[str, float]:
        spec = self.protocol["reward_definition"]
        scales = spec["normalization"]
        weights = spec["weights"]
        speed_error = (record.physical_state.speed_mps - float(spec["target_speed_mps"])) / float(scales["speed_error_mps"])
        gap_error = (record.augmented_state.gap_m - float(spec["target_gap_m"])) / float(scales["gap_error_m"])
        nominal = np.asarray(record.nominal_action, dtype=np.float64)
        provisional = np.asarray(record.provisional_action, dtype=np.float64)
        final = np.asarray(record.final_action, dtype=np.float64)
        normalized_final = final / self.action_upper
        normalized_intervention = (provisional - nominal) / self.action_upper
        negative_certificate = min(0.0, record.rho_H_cert / float(scales["rho_scale"]))
        components = {
            "speed_tracking": -float(weights["speed_tracking"]) * speed_error**2,
            "spacing_tracking": -float(weights["spacing_tracking"]) * gap_error**2,
            "control_effort": -float(weights["control_effort"]) * float(np.sum(normalized_final**2)),
            "friction_usage": -float(weights["friction_usage"]) * float(normalized_final[0]),
            "auxiliary_usage": -float(weights["auxiliary_usage"]) * float(normalized_final[1]),
            "provisional_qp_intervention": -float(weights["provisional_qp_intervention"]) * float(np.sum(normalized_intervention**2)),
            "negative_predictive_certificate": -float(weights["negative_predictive_certificate"]) * negative_certificate**2,
        }
        return {**components, "total": float(sum(components.values()))}


def _scenario(inputs: Any, name: str, seed: int) -> HeavyPlatoonEnv:
    protocol = inputs.protocol
    grade = float(protocol["M2"]["grade_rad"])
    t_hot = float(protocol["M3"]["initial_temperature_K"][1])
    if name == "nominal_descent":
        scenario, controllers = build_scenario(inputs, 3, name, speed_mps=15.0, temperature_K=350.0, grade_rad=grade)
    elif name == "hot_brake_descent":
        scenario, controllers = build_scenario(inputs, 3, name, speed_mps=15.0, temperature_K=t_hot, grade_rad=grade)
    elif name == "grade_transition":
        scenario, controllers = build_scenario(inputs, 3, name, speed_mps=15.0, temperature_K=350.0, grade_rad=0.0)
        road = PiecewiseLinearGrade((-400.0, -220.0, -120.0, 100.0), (0.0, -0.03, grade, grade))
        scenario = replace(scenario, road=road)
    elif name == "leader_braking":
        predecessor = records_by_id(inputs)["predecessor_emergency_bounds"]["nominal_value"]
        deceleration = abs(float(predecessor["acceleration_lower_mps2"]))
        leader = EmergencyBrakingPulseLeader(0.0, 15.0, 0.3, 0.6, deceleration)
        scenario, controllers = build_scenario(inputs, 3, name, speed_mps=15.0, temperature_K=350.0, grade_rad=0.0, leader=leader)
    elif name == "communication_delay_loss":
        channel = ChannelConfig(
            delay_model="random_bounded", maximum_random_delay_s=0.2,
            packet_loss_probability=0.10, burst_loss_probability=0.05,
            burst_length=2, random_seed=int(seed),
        )
        scenario, controllers = build_scenario(inputs, 3, name, speed_mps=15.0, temperature_K=350.0, grade_rad=grade, channel=channel)
    elif name == "low_speed_operation":
        scenario, controllers = build_scenario(inputs, 3, name, speed_mps=0.5, temperature_K=350.0, grade_rad=0.0)
    elif name == "heterogeneous_platoon":
        scenario, _ = build_scenario(inputs, 3, name, speed_mps=15.0, temperature_K=350.0, grade_rad=grade)
        source = records_by_id(inputs)
        masses = (
            float(source["vehicle_mass"]["uncertainty_lower"]),
            float(source["vehicle_mass"]["nominal_value"]),
            float(source["vehicle_mass"]["uncertainty_upper"]),
        )
        tau_f = (
            float(source["friction_actuator_lag"]["uncertainty_lower"]),
            float(source["friction_actuator_lag"]["nominal_value"]),
            float(source["friction_actuator_lag"]["uncertainty_upper"]),
        )
        tau_a = (
            float(source["auxiliary_actuator_lag"]["uncertainty_lower"]),
            float(source["auxiliary_actuator_lag"]["nominal_value"]),
            float(source["auxiliary_actuator_lag"]["uncertainty_upper"]),
        )
        vehicles = tuple(build_vehicle(inputs, mass_kg=masses[i], tau_f_s=tau_f[i], tau_a_s=tau_a[i]) for i in range(3))
        drive = tuple(v.mass_kg * v.gravity_mps2 * v.rolling_resistance_coefficient for v in vehicles)
        scenario = replace(scenario, parameters=vehicles, drive_force_N=drive)
        controllers = [build_controller(inputs, vehicle) for vehicle in vehicles]
    else:
        raise ValueError(f"unregistered formal scenario: {name}")
    return make_env(inputs, scenario, controllers)


def _nominal_feasible(rows: Sequence[Any], action: Sequence[float], tolerance: float) -> bool:
    u_f, u_a = map(float, action)
    return all(row.h - row.g_f * u_f - row.g_a * u_a >= -tolerance for row in rows)


def _aggregate_cycles(
    cycles: list[tuple[Phase2BLogRecord, Sequence[Any], dict[str, float], dict[str, float]]],
    protocol: dict[str, Any],
) -> dict[str, Any]:
    threshold = float(protocol["intervention_frequency_threshold_N"])
    tolerance = float(protocol["hard_feasibility_tolerance"])
    dt = 0.1
    modes = Counter(record.supervisor_mode for record, _rows, _reward, _timing in cycles)
    final_norms = [float(np.linalg.norm(np.asarray(r.final_action) - np.asarray(r.nominal_action))) for r, *_ in cycles]
    provisional_norms = [float(np.linalg.norm(np.asarray(r.provisional_action) - np.asarray(r.nominal_action))) for r, *_ in cycles]
    feasible = [_nominal_feasible(rows, record.nominal_action, tolerance) for record, rows, _reward, _timing in cycles]
    return {
        "episode_return": float(sum(reward["total"] for _r, _rows, reward, _timing in cycles)),
        "nominal_hard_feasible_rate": float(np.mean(feasible)),
        "qp_intervention_frequency": float(np.mean(np.asarray(final_norms) > threshold)),
        "qp_intervention_norm_mean_N": float(np.mean(final_norms)),
        "provisional_qp_intervention_norm_mean_N": float(np.mean(provisional_norms)),
        "minimum_rho": float(min(r.rho for r, *_ in cycles)),
        "minimum_rho_H_cert": float(min(r.rho_H_cert for r, *_ in cycles)),
        "certificate_loss_count": int(sum(r.rho_H_cert < 0.0 for r, *_ in cycles)),
        "backup_mode_occupancy": float(sum(r.supervisor_mode in {"certified_backup", "minimal_risk"} for r, *_ in cycles) / len(cycles)),
        "supervisor_mode_occupancy": {key: value / len(cycles) for key, value in sorted(modes.items())},
        "mean_absolute_speed_tracking_error_mps": float(np.mean([abs(r.physical_state.speed_mps - 14.0) for r, *_ in cycles])),
        "mean_absolute_gap_tracking_error_m": float(np.mean([abs(r.augmented_state.gap_m - 65.0) for r, *_ in cycles])),
        "peak_brake_temperature_K": float(max(r.physical_state.temperature_K for r, *_ in cycles)),
        "friction_brake_energy_J": float(sum(r.final_action[0] * max(0.0, r.physical_state.speed_mps) * dt for r, *_ in cycles)),
        "auxiliary_brake_energy_J": float(sum(r.final_action[1] * max(0.0, r.physical_state.speed_mps) * dt for r, *_ in cycles)),
        "mission_completion": float(all(r.augmented_state.gap_m > 0.0 and r.final_residuals_valid for r, *_ in cycles)),
        "forward_qp_time_s": float(sum(timing.get("qp1", timing.get("qp_solve", 0.0)) for _r, _rows, _reward, timing in cycles)),
        "controller_time_s": float(sum(timing.get("total_controller", 0.0) for _r, _rows, _reward, timing in cycles)),
        "action_count": len(cycles),
    }


def collect_rollout(
    env: HeavyPlatoonEnv,
    actor: TorchSharedActor,
    critic: TorchCentralizedCritic,
    reward: FormalReward,
    protocol: dict[str, Any],
    torch_generator: torch.Generator,
) -> tuple[RolloutBuffer, dict[str, Any]]:
    horizon = int(protocol["rollout_length"])
    buffer = RolloutBuffer(env.controlled_truck_count)
    cycles: list[tuple[Phase2BLogRecord, Sequence[Any], dict[str, float], dict[str, float]]] = []
    for step_index in range(horizon):
        observations = np.stack([local_observation_vector(item) for item in env.current_local_observations()])
        actor.normalizer.update(observations)
        central_state = centralized_state_vector(env.centralized_training_state())
        values = critic.values_numpy(central_state)[0]
        sample = actor.sample_numpy(observations, torch_generator)
        env.nominal_controllers = [ConstantCommandController(float(action[0]), float(action[1])) for action in sample.action]
        result = env.step()
        next_state = centralized_state_vector(env.centralized_training_state())
        next_values = critic.values_numpy(next_state)[0]
        nominal = np.asarray([record.nominal_action for record in result.logs], dtype=np.float64)
        provisional = np.asarray([record.provisional_action for record in result.logs], dtype=np.float64)
        executed = np.asarray([record.final_action for record in result.logs], dtype=np.float64)
        if not np.allclose(nominal, sample.action, rtol=0.0, atol=1e-9):
            raise AssertionError("formal safety controller did not receive sampled u_RL")
        reward_terms = [reward.terms(record) for record in result.logs]
        rewards = np.asarray([item["total"] for item in reward_terms], dtype=np.float64)
        hard_rows = tuple(item.hard_rows for item in result.controller_results)
        qp_status = tuple(item.qp_status for item in result.controller_results)
        for index, record in enumerate(result.logs):
            cycles.append((record, hard_rows[index], reward_terms[index], result.controller_results[index].timings_s))
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
            hard_rows=hard_rows,
            qp_status=qp_status,
        )
    advantages, returns = torch_generalized_advantage_estimation(
        np.stack([step["rewards"] for step in buffer.steps]),
        np.stack([step["values"] for step in buffer.steps]),
        np.stack([step["next_values"] for step in buffer.steps]),
        np.stack([step["terminals"] for step in buffer.steps]),
        np.stack([step["truncations"] for step in buffer.steps]),
        float(protocol["gamma"]), float(protocol["gae_lambda"]),
    )
    buffer.advantages = advantages.cpu().numpy()
    buffer.returns = returns.cpu().numpy()
    return buffer, _aggregate_cycles(cycles, protocol)


def _select_batch(batch: RolloutBatch, indices: np.ndarray) -> RolloutBatch:
    return RolloutBatch(**{name: getattr(batch, name)[indices] for name in batch.__dataclass_fields__})


def _parameter_norm(gradients: Iterable[torch.Tensor | None]) -> float:
    return float(math.sqrt(sum(float(torch.sum(item.square()).item()) for item in gradients if item is not None)))


def update_policy(
    actor: TorchSharedActor,
    critic: TorchCentralizedCritic,
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    buffer: RolloutBuffer,
    mode: QPGradientMode,
    config: FormalPPOConfig,
    protocol: dict[str, Any],
    rng: np.random.Generator,
) -> dict[str, Any]:
    if buffer.advantages is None:
        raise ValueError("GAE is required before formal PPO update")
    flat_advantages = buffer.advantages.reshape(-1)
    buffer.advantages = (buffer.advantages - np.mean(flat_advantages)) / (np.std(flat_advantages) + 1e-8)
    batch = buffer.flattened()
    rows_flat = [tuple(rows) for step in buffer.steps for rows in step["hard_rows"]]
    status_flat = [status for step in buffer.steps for status in step["qp_status"]]
    upper = actor.action_upper.detach()
    coefficient = float(protocol["intervention_objective_coefficient"])
    accumulated: list[dict[str, float]] = []
    fallback_reasons: Counter[str] = Counter()
    active_sets: Counter[str] = Counter()
    valid_count = 0
    qp_count = 0
    licq_failures = 0
    strict_failures = 0
    ill_conditioned = 0
    qp_forward_s = 0.0
    qp_backward_s = 0.0
    update_start = time.perf_counter()
    for _epoch in range(config.epochs):
        permutation = rng.permutation(len(batch.old_log_probs))
        for start in range(0, len(permutation), config.minibatch_size):
            indices = permutation[start : start + config.minibatch_size]
            mini = _select_batch(batch, indices)
            losses = torch_ppo_losses(actor, critic, mini, config)  # type: ignore[arg-type]
            mean, _ = actor(mini.observations)
            nominal = actor.distribution.deterministic(mean)
            qp_actions: list[torch.Tensor] = []
            forward_start = time.perf_counter()
            for local_index, flat_index in enumerate(indices):
                qp_count += 1
                if status_flat[int(flat_index)] != "optimal":
                    qp_actions.append(torch.as_tensor(mini.provisional_actions[local_index], dtype=TORCH_DTYPE))
                    fallback_reasons["EMPTY_HARD_POLYTOPE"] += 1
                    continue
                qp = CanonicalSafetyQP(tuple(rows_flat[int(flat_index)]))
                result = solve_differentiable_qp(nominal[local_index], qp, mode)
                qp_actions.append(result.action)
                active_sets["+".join(result.record.active_constraint_names) or "INTERIOR"] += 1
                if result.record.diffqp_gradient_valid:
                    valid_count += 1
                else:
                    fallback_reasons[result.record.fallback_reason] += 1
                licq_failures += int(not result.record.licq_satisfied)
                strict_failures += int(not result.record.strict_complementarity_satisfied)
                ill_conditioned += int(result.record.fallback_reason == "ILL_CONDITIONED_KKT")
            qp_forward_s += time.perf_counter() - forward_start
            projected = torch.stack(qp_actions)
            normalized_delta = (projected - nominal) / upper
            qp_loss = coefficient * torch.mean(torch.sum(normalized_delta.square(), dim=1))
            path_loss = coefficient * torch.mean(torch.sum(((projected - nominal.detach()) / upper).square(), dim=1))
            if path_loss.requires_grad:
                mediated = torch.autograd.grad(path_loss, tuple(actor.parameters()), retain_graph=True, allow_unused=True)
                mediated_norm = _parameter_norm(mediated)
            else:
                mediated_norm = 0.0

            actor_optimizer.zero_grad(set_to_none=True)
            backward_start = time.perf_counter()
            qp_loss.backward(retain_graph=True)
            qp_backward_s += time.perf_counter() - backward_start
            losses.actor_loss.backward()
            if any(parameter.grad is None or not torch.all(torch.isfinite(parameter.grad)) for parameter in actor.parameters()):
                raise FloatingPointError("invalid formal actor gradient")
            actor_norm = float(torch.nn.utils.clip_grad_norm_(actor.parameters(), config.maximum_gradient_norm).item())
            actor_optimizer.step()

            critic_optimizer.zero_grad(set_to_none=True)
            losses.critic_loss.backward()
            if any(parameter.grad is None or not torch.all(torch.isfinite(parameter.grad)) for parameter in critic.parameters()):
                raise FloatingPointError("invalid formal critic gradient")
            critic_norm = float(torch.nn.utils.clip_grad_norm_(critic.parameters(), config.maximum_gradient_norm).item())
            critic_optimizer.step()
            accumulated.append({
                "policy_loss": float(losses.policy_loss.detach().item()),
                "value_loss": float(losses.value_loss.detach().item()),
                "entropy": float(losses.entropy.detach().item()),
                "intervention_objective": float(qp_loss.detach().item()),
                "approximate_kl": float(torch.mean(losses.old_log_probability - losses.new_log_probability).detach().item()),
                "clip_fraction": float(torch.mean(losses.clipped_mask.to(TORCH_DTYPE)).detach().item()),
                "actor_gradient_norm": actor_norm,
                "critic_gradient_norm": critic_norm,
                "qp_mediated_actor_gradient_norm": mediated_norm,
            })
    update_s = time.perf_counter() - update_start
    if mode is QPGradientMode.STOP_GRADIENT and any(item["qp_mediated_actor_gradient_norm"] != 0.0 for item in accumulated):
        raise AssertionError("StopGradient leaked a QP-mediated actor gradient")
    mean_metrics = {key: float(np.mean([item[key] for item in accumulated])) for key in accumulated[0]}
    return {
        **mean_metrics,
        "minibatch_updates": len(accumulated),
        "qp_gradient_valid_fraction": valid_count / qp_count,
        "qp_fallback_count": int(qp_count - valid_count),
        "qp_fallback_fraction": (qp_count - valid_count) / qp_count,
        "qp_fallback_reasons": dict(sorted(fallback_reasons.items())),
        "active_set_frequency": dict(sorted(active_sets.items())),
        "licq_failure_frequency": licq_failures / qp_count,
        "strict_complementarity_failure_frequency": strict_failures / qp_count,
        "ill_conditioned_kkt_frequency": ill_conditioned / qp_count,
        "training_qp_forward_time_s": qp_forward_s,
        "qp_objective_backward_time_s": qp_backward_s,
        "ppo_update_time_s": update_s,
    }


def evaluate_actor(
    actor: TorchSharedActor,
    inputs: Any,
    reward: FormalReward,
    protocol: dict[str, Any],
    scenario_name: str,
    evaluation_seed: int,
) -> dict[str, Any]:
    checksum = actor.checksum()
    normalizer = actor.normalizer.state_dict()
    env = _scenario(inputs, scenario_name, evaluation_seed)
    cycles: list[tuple[Phase2BLogRecord, Sequence[Any], dict[str, float], dict[str, float]]] = []
    for _ in range(int(protocol["evaluation_steps_per_scenario"])):
        observations = np.stack([local_observation_vector(item) for item in env.current_local_observations()])
        nominal = actor.deterministic_numpy(observations)
        env.nominal_controllers = [ConstantCommandController(float(action[0]), float(action[1])) for action in nominal]
        result = env.step()
        for index, record in enumerate(result.logs):
            cycles.append((record, result.controller_results[index].hard_rows, reward.terms(record), result.controller_results[index].timings_s))
    if actor.checksum() != checksum:
        raise AssertionError("formal evaluation modified actor parameters")
    after = actor.normalizer.state_dict()
    if normalizer["count"] != after["count"] or not np.array_equal(normalizer["mean"], after["mean"]) or not np.array_equal(normalizer["m2"], after["m2"]):
        raise AssertionError("formal evaluation modified observation normalization")
    return {
        "scenario": scenario_name,
        "evaluation_seed": int(evaluation_seed),
        **_aggregate_cycles(cycles, protocol),
    }


def _checkpoint(
    path: Path,
    actor: TorchSharedActor,
    critic: TorchCentralizedCritic,
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    update: int,
    metadata: dict[str, Any],
) -> str:
    torch.save({
        "schema_version": 1,
        "training_mode": "FORMAL",
        "paper_eligible": True,
        "update": int(update),
        "actor_state_dict": actor.state_dict(),
        "actor_normalizer_state": actor.normalizer.state_dict(),
        "critic_state_dict": critic.state_dict(),
        "actor_optimizer_state": actor_optimizer.state_dict(),
        "critic_optimizer_state": critic_optimizer.state_dict(),
        **metadata,
    }, path)
    return file_hash(path)


def _aggregate_evaluations(records: list[dict[str, Any]]) -> dict[str, float]:
    scalar_keys = (
        "episode_return", "nominal_hard_feasible_rate", "qp_intervention_frequency",
        "qp_intervention_norm_mean_N", "provisional_qp_intervention_norm_mean_N",
        "minimum_rho", "minimum_rho_H_cert", "certificate_loss_count",
        "backup_mode_occupancy", "mean_absolute_speed_tracking_error_mps",
        "mean_absolute_gap_tracking_error_m", "peak_brake_temperature_K",
        "friction_brake_energy_J", "auxiliary_brake_energy_J", "mission_completion",
        "forward_qp_time_s", "controller_time_s",
    )
    result = {key: float(np.mean([float(item[key]) for item in records])) for key in scalar_keys}
    result["minimum_rho"] = float(min(float(item["minimum_rho"]) for item in records))
    result["minimum_rho_H_cert"] = float(min(float(item["minimum_rho_H_cert"]) for item in records))
    result["peak_brake_temperature_K"] = float(max(float(item["peak_brake_temperature_K"]) for item in records))
    return result


def run_one(
    method: str,
    seed: int,
    inputs: Any,
    protocol: dict[str, Any],
    metadata: dict[str, Any],
    raw_root: Path,
) -> dict[str, Any]:
    mode = QPGradientMode.DIFFQP if method == "SAFE-MAPPO-DIFFQP" else QPGradientMode.STOP_GRADIENT
    run_id = f"{method.lower().replace('safe-mappo-', '')}-seed-{seed}"
    final_dir = raw_root / run_id
    if final_dir.exists():
        raise FileExistsError(f"formal run already exists: {run_id}")
    temporary = raw_root / f".{run_id}.incomplete"
    temporary.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    torch_generator = torch.Generator(device="cpu")
    torch_generator.manual_seed(seed + 1_000_000)
    probe = _scenario(inputs, protocol["training_scenarios"][0], seed)
    observation_dim = local_observation_vector(probe.current_local_observations()[0]).size
    state_dim = centralized_state_vector(probe.centralized_training_state()).size
    upper = np.asarray([
        min(item.friction_force_limit_N for item in probe.scenario.parameters),
        min(item.auxiliary_force_limit_N for item in probe.scenario.parameters),
    ], dtype=np.float64)
    actor = TorchSharedActor(observation_dim, 32, np.zeros(2), upper, rng)
    critic = TorchCentralizedCritic(state_dim, 32, 3, rng)
    config = FormalPPOConfig(
        float(protocol["clip_epsilon"]), float(protocol["entropy_coefficient"]),
        float(protocol["value_coefficient"]), float(protocol["actor_learning_rate"]),
        float(protocol["critic_learning_rate"]), float(protocol["gradient_clipping_norm"]),
        int(protocol["ppo_epochs"]), int(protocol["minibatch_size"]),
    )
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=config.actor_learning_rate)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=config.critic_learning_rate)
    reward = FormalReward(protocol, upper)
    initial_actor = formal_actor_checksum(actor)
    initial_critic = critic.checksum()
    checkpoint_hashes: dict[str, str] = {}
    update_records: list[dict[str, Any]] = []
    evaluation_records: list[dict[str, Any]] = []
    training_wall_clock_s = 0.0
    common_checkpoint_metadata = {
        **metadata, "run_id": run_id, "method": method, "seed": int(seed),
        "initial_actor_checksum": initial_actor, "initial_critic_checksum": initial_critic,
        "qp_gradient_mode": mode.value,
    }
    checkpoints = set(map(int, protocol["checkpoint_updates"]))
    for update in range(0, int(protocol["number_of_updates"]) + 1):
        if update in checkpoints:
            checkpoint_path = temporary / f"checkpoint_update_{update:03d}.pt"
            checkpoint_hashes[str(update)] = _checkpoint(
                checkpoint_path, actor, critic, actor_optimizer, critic_optimizer,
                update, common_checkpoint_metadata,
            )
            for evaluation_seed in protocol["evaluation_seeds"]:
                for scenario_name in protocol["evaluation_scenarios"]:
                    evaluation_records.append({
                        "run_id": run_id, "method": method, "training_seed": int(seed),
                        "checkpoint_update": update,
                        **evaluate_actor(actor, inputs, reward, protocol, scenario_name, int(evaluation_seed)),
                    })
        if update == int(protocol["number_of_updates"]):
            break
        scenario_index = (update + int(seed)) % len(protocol["training_scenarios"])
        scenario_name = protocol["training_scenarios"][scenario_index]
        training_start = time.perf_counter()
        env = _scenario(inputs, scenario_name, seed * 1000 + update)
        rollout, rollout_metrics = collect_rollout(env, actor, critic, reward, protocol, torch_generator)
        update_metrics = update_policy(
            actor, critic, actor_optimizer, critic_optimizer, rollout,
            mode, config, protocol, rng,
        )
        update_records.append({
            "run_id": run_id, "method": method, "seed": int(seed),
            "update": update + 1,
            "environment_steps": (update + 1) * int(protocol["rollout_length"]),
            "scenario": scenario_name,
            **rollout_metrics, **update_metrics,
        })
        training_wall_clock_s += time.perf_counter() - training_start
    update_log = temporary / "updates.jsonl"
    evaluation_log = temporary / "evaluations.jsonl"
    update_log.write_text("".join(json.dumps(item, sort_keys=True) + "\n" for item in update_records), encoding="utf-8", newline="\n")
    evaluation_log.write_text("".join(json.dumps(item, sort_keys=True) + "\n" for item in evaluation_records), encoding="utf-8", newline="\n")
    checkpoint_summary: dict[str, dict[str, float]] = {}
    for checkpoint in sorted(checkpoints):
        selected = [item for item in evaluation_records if item["checkpoint_update"] == checkpoint]
        checkpoint_summary[str(checkpoint)] = _aggregate_evaluations(selected)
    final_metrics = checkpoint_summary[str(max(checkpoints))]
    x = np.asarray(sorted(checkpoints), dtype=np.float64) * int(protocol["rollout_length"])
    y = np.asarray([checkpoint_summary[str(item)]["episode_return"] for item in sorted(checkpoints)], dtype=np.float64)
    final_steps = float(protocol["training_horizon_environment_steps"])
    final_metrics["evaluation_return_auc"] = float(np.trapezoid(y, x) / final_steps)
    diagnostics = {
        "qp_gradient_valid_fraction": float(np.mean([item["qp_gradient_valid_fraction"] for item in update_records])),
        "qp_fallback_fraction": float(np.mean([item["qp_fallback_fraction"] for item in update_records])),
        "qp_mediated_actor_gradient_norm": float(np.mean([item["qp_mediated_actor_gradient_norm"] for item in update_records])),
        "training_qp_forward_time_s": float(sum(item["training_qp_forward_time_s"] for item in update_records)),
        "qp_objective_backward_time_s": float(sum(item["qp_objective_backward_time_s"] for item in update_records)),
        "ppo_update_time_s": float(sum(item["ppo_update_time_s"] for item in update_records)),
        "training_wall_clock_time_s": training_wall_clock_s,
        "peak_resident_memory_MiB": float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0),
        "fallback_reasons": dict(sum((Counter(item["qp_fallback_reasons"]) for item in update_records), Counter())),
        "active_set_frequency": dict(sum((Counter(item["active_set_frequency"]) for item in update_records), Counter())),
        "licq_failure_frequency": float(np.mean([item["licq_failure_frequency"] for item in update_records])),
        "strict_complementarity_failure_frequency": float(np.mean([item["strict_complementarity_failure_frequency"] for item in update_records])),
        "ill_conditioned_kkt_frequency": float(np.mean([item["ill_conditioned_kkt_frequency"] for item in update_records])),
    }
    run_manifest = {
        "schema_version": 1, "status": "PASS", "run_id": run_id,
        "method": method, "seed": int(seed), "training_mode": "FORMAL",
        "data_provenance": "literature_calibrated", "paper_eligible": True,
        **metadata,
        "initial_actor_checksum": initial_actor, "initial_critic_checksum": initial_critic,
        "final_actor_checksum": formal_actor_checksum(actor), "final_critic_checksum": critic.checksum(),
        "checkpoint_hashes": checkpoint_hashes,
        "updates_sha256": file_hash(update_log), "evaluations_sha256": file_hash(evaluation_log),
        "checkpoint_summary": checkpoint_summary, "final_metrics": final_metrics,
        "diagnostics": diagnostics,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    dump_json(temporary / "run_manifest.json", run_manifest)
    temporary.rename(final_dir)
    for path in final_dir.iterdir():
        path.chmod(0o444)
    return {
        "run_id": run_id, "method": method, "seed": int(seed), "status": "PASS",
        "run_manifest": f"raw/{run_id}/run_manifest.json",
        "run_manifest_sha256": file_hash(final_dir / "run_manifest.json"),
        "paper_eligible": True,
    }


def _bootstrap_ci(differences: np.ndarray, repetitions: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    samples = rng.choice(differences, size=(repetitions, len(differences)), replace=True)
    means = np.mean(samples, axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _sign_flip_p(differences: np.ndarray) -> float:
    observed = abs(float(np.mean(differences)))
    count = 0
    total = 0
    for signs in itertools.product((-1.0, 1.0), repeat=len(differences)):
        total += 1
        if abs(float(np.mean(differences * np.asarray(signs)))) >= observed - 1e-15:
            count += 1
    return count / total


def _holm(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values, key=p_values.get)
    adjusted: dict[str, float] = {}
    running = 0.0
    m = len(ordered)
    for rank, key in enumerate(ordered):
        running = max(running, (m - rank) * p_values[key])
        adjusted[key] = min(1.0, running)
    return adjusted


def paired_statistics(
    manifests: list[dict[str, Any]], protocol: dict[str, Any]
) -> dict[str, Any]:
    by_method_seed = {(item["method"], int(item["seed"])): item for item in manifests}
    metric_map = {
        "final_evaluation_return": "episode_return",
        "evaluation_return_auc": "evaluation_return_auc",
        "final_qp_intervention_frequency": "qp_intervention_frequency",
        "final_qp_intervention_norm": "qp_intervention_norm_mean_N",
        "final_nominal_hard_feasible_rate": "nominal_hard_feasible_rate",
    }
    raw: dict[str, dict[str, Any]] = {}
    p_values: dict[str, float] = {}
    for offset, (outcome, key) in enumerate(metric_map.items()):
        diffqp = np.asarray([
            by_method_seed[("SAFE-MAPPO-DIFFQP", int(seed))]["final_metrics"][key]
            for seed in protocol["training_seeds"]
        ], dtype=np.float64)
        stopgrad = np.asarray([
            by_method_seed[("SAFE-MAPPO-STOPGRAD", int(seed))]["final_metrics"][key]
            for seed in protocol["training_seeds"]
        ], dtype=np.float64)
        differences = diffqp - stopgrad
        standard = float(np.std(differences, ddof=1))
        ci = _bootstrap_ci(
            differences, int(protocol["statistical_analysis"]["bootstrap_replicates"]),
            int(protocol["statistical_analysis"]["bootstrap_seed"]) + offset,
        )
        p_value = _sign_flip_p(differences)
        p_values[outcome] = p_value
        raw[outcome] = {
            "metric_key": key,
            "n_pairs": len(differences),
            "diffqp_values": diffqp.tolist(), "stopgrad_values": stopgrad.tolist(),
            "paired_differences": differences.tolist(),
            "diffqp_mean": float(np.mean(diffqp)), "diffqp_std": float(np.std(diffqp, ddof=1)),
            "stopgrad_mean": float(np.mean(stopgrad)), "stopgrad_std": float(np.std(stopgrad, ddof=1)),
            "paired_mean": float(np.mean(differences)), "paired_std": standard,
            "paired_median": float(np.median(differences)),
            "paired_bootstrap_95_ci": list(ci),
            "cohen_dz": float(np.mean(differences) / standard) if standard > 0.0 else (math.copysign(math.inf, float(np.mean(differences))) if np.mean(differences) != 0.0 else 0.0),
            "unadjusted_p": p_value,
        }
    adjusted = _holm(p_values)
    for key, value in adjusted.items():
        raw[key]["holm_adjusted_p"] = value
    return raw


def _claim_status(statistic: dict[str, Any], direction: str, alpha: float) -> str:
    mean = float(statistic["paired_mean"])
    low, high = map(float, statistic["paired_bootstrap_95_ci"])
    favorable = mean > 0.0 if direction == "higher" else mean < 0.0
    favorable_ci = low > 0.0 if direction == "higher" else high < 0.0
    adverse_ci = high < 0.0 if direction == "higher" else low > 0.0
    if favorable and favorable_ci and float(statistic["holm_adjusted_p"]) <= alpha:
        return "SUPPORTED"
    if favorable and favorable_ci:
        return "PARTIALLY_SUPPORTED"
    if adverse_ci and float(statistic["holm_adjusted_p"]) <= alpha:
        return "NOT_SUPPORTED"
    return "INCONCLUSIVE"


def _svg_axes(title: str, x_label: str, y_label: str, body: str, width: int = 760, height: int = 460) -> str:
    namespace = "http" + "://www.w3.org/2000/svg"
    return f'''<svg xmlns="{namespace}" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/><text x="{width/2}" y="28" text-anchor="middle" font-family="sans-serif" font-size="18">{title}</text>
<line x1="70" y1="390" x2="720" y2="390" stroke="black"/><line x1="70" y1="50" x2="70" y2="390" stroke="black"/>
<text x="395" y="445" text-anchor="middle" font-family="sans-serif" font-size="13">{x_label}</text>
<text x="18" y="220" text-anchor="middle" transform="rotate(-90 18 220)" font-family="sans-serif" font-size="13">{y_label}</text>{body}</svg>'''


def _curve_svg(
    title: str, y_label: str, updates: list[dict[str, Any]], key: str, output: Path
) -> None:
    methods = (("SAFE-MAPPO-DIFFQP", "#1565c0"), ("SAFE-MAPPO-STOPGRAD", "#d84315"))
    xs = sorted({float(item["environment_steps"]) for item in updates})
    grouped: dict[tuple[str, float], list[float]] = {}
    for item in updates:
        grouped.setdefault((item["method"], float(item["environment_steps"])), []).append(float(item[key]))
    values = [value for group in grouped.values() for value in group]
    y_min, y_max = min(values), max(values)
    if math.isclose(y_min, y_max):
        y_min -= 1.0
        y_max += 1.0
    x_min, x_max = min(xs), max(xs)
    scale_x = lambda x: 70.0 + 650.0 * (x - x_min) / max(1.0, x_max - x_min)
    scale_y = lambda y: 390.0 - 340.0 * (y - y_min) / (y_max - y_min)
    body: list[str] = []
    for method, color in methods:
        points = []
        for x in xs:
            group = np.asarray(grouped[(method, x)], dtype=np.float64)
            mean, std = float(np.mean(group)), float(np.std(group, ddof=1))
            px, py = scale_x(x), scale_y(mean)
            points.append(f"{px:.2f},{py:.2f}")
            body.append(f'<line x1="{px:.2f}" y1="{scale_y(mean-std):.2f}" x2="{px:.2f}" y2="{scale_y(mean+std):.2f}" stroke="{color}" opacity="0.45"/>')
        body.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="2"/>')
    body.append('<text x="90" y="70" font-family="sans-serif" font-size="12" fill="#1565c0">DiffQP</text><text x="170" y="70" font-family="sans-serif" font-size="12" fill="#d84315">StopGrad</text>')
    output.write_text(_svg_axes(title, "environment steps", y_label, "".join(body)), encoding="utf-8", newline="\n")


def generate_figures(
    result_root: Path, manifests: list[dict[str, Any]], statistics_data: dict[str, Any]
) -> list[str]:
    figure_root = result_root / "figures"
    figure_root.mkdir(parents=True, exist_ok=True)
    updates: list[dict[str, Any]] = []
    for manifest in manifests:
        update_file = result_root / "raw" / manifest["run_id"] / "updates.jsonl"
        updates.extend(json.loads(line) for line in update_file.read_text(encoding="utf-8").splitlines())
    specifications = (
        ("learning_figure_1_return.svg", "Return vs environment steps", "rollout return", "episode_return"),
        ("learning_figure_2_intervention_frequency.svg", "QP intervention frequency vs steps", "frequency", "qp_intervention_frequency"),
        ("learning_figure_3_intervention_norm.svg", "QP intervention magnitude vs steps", "mean norm (N)", "qp_intervention_norm_mean_N"),
        ("learning_figure_4_nominal_feasible.svg", "Nominal hard-feasible rate vs steps", "feasible fraction", "nominal_hard_feasible_rate"),
    )
    for filename, title, label, key in specifications:
        _curve_svg(title, label, updates, key, figure_root / filename)
    paired = statistics_data["final_evaluation_return"]
    differences = paired["paired_differences"]
    low, high = paired["paired_bootstrap_95_ci"]
    body = []
    lo = min(differences + [low, 0.0]); hi = max(differences + [high, 0.0])
    span = max(1e-12, hi - lo)
    sy = lambda y: 390.0 - 340.0 * (y - lo) / span
    body.append(f'<line x1="70" y1="{sy(0):.2f}" x2="720" y2="{sy(0):.2f}" stroke="#777" stroke-dasharray="4 4"/>')
    for index, value in enumerate(differences):
        body.append(f'<circle cx="{120+55*index}" cy="{sy(value):.2f}" r="5" fill="#1565c0"/>')
    body.append(f'<line x1="670" y1="{sy(low):.2f}" x2="670" y2="{sy(high):.2f}" stroke="#d84315" stroke-width="4"/>')
    (figure_root / "learning_figure_5_paired_final.svg").write_text(
        _svg_axes("Paired final return differences and 95% CI", "paired seed", "DiffQP - StopGrad", "".join(body)), encoding="utf-8", newline="\n"
    )
    diffqp = [item["diagnostics"] for item in manifests if item["method"] == "SAFE-MAPPO-DIFFQP"]
    valid = float(np.mean([item["qp_gradient_valid_fraction"] for item in diffqp]))
    fallback = float(np.mean([item["qp_fallback_fraction"] for item in diffqp]))
    bars = f'<rect x="180" y="{390-300*valid:.2f}" width="120" height="{300*valid:.2f}" fill="#1565c0"/><rect x="460" y="{390-300*fallback:.2f}" width="120" height="{300*fallback:.2f}" fill="#d84315"/><text x="240" y="415" text-anchor="middle" font-family="sans-serif">valid</text><text x="520" y="415" text-anchor="middle" font-family="sans-serif">fallback</text>'
    (figure_root / "learning_figure_6_diffqp_diagnostics.svg").write_text(
        _svg_axes("DiffQP backward validity", "category", "fraction", bars), encoding="utf-8", newline="\n"
    )
    return [str(path.relative_to(result_root)).replace("\\", "/") for path in sorted(figure_root.glob("*.svg"))]


def _tex_escape(value: Any) -> str:
    return str(value).replace("_", "\\_").replace("%", "\\%")


def _write_table(path: Path, headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> None:
    columns = "l" + "r" * (len(headers) - 1)
    lines = [f"\\begin{{tabular}}{{{columns}}}", "\\toprule", " & ".join(map(_tex_escape, headers)) + " \\\\", "\\midrule"]
    lines.extend(" & ".join(_tex_escape(value) for value in row) + " \\\\" for row in rows)
    lines.extend(("\\bottomrule", "\\end{tabular}"))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def generate_tables(
    result_root: Path, protocol: dict[str, Any], manifests: list[dict[str, Any]], statistics_data: dict[str, Any]
) -> list[str]:
    table_root = result_root / "tables"
    table_root.mkdir(parents=True, exist_ok=True)
    _write_table(table_root / "table_L1_protocol.tex", ("Item", "Registered value"), (
        ("Training seeds", len(protocol["training_seeds"])), ("Updates", protocol["number_of_updates"]),
        ("Rollout length", protocol["rollout_length"]), ("PPO epochs", protocol["ppo_epochs"]),
        ("Minibatch", protocol["minibatch_size"]), ("Checkpoints", ",".join(map(str, protocol["checkpoint_updates"]))),
    ))
    outcome_rows = []
    for outcome, stats in statistics_data.items():
        outcome_rows.append((outcome, f'{stats["diffqp_mean"]:.6g}', f'{stats["diffqp_std"]:.6g}', f'{stats["stopgrad_mean"]:.6g}', f'{stats["stopgrad_std"]:.6g}'))
    _write_table(table_root / "table_L2_final_metrics.tex", ("Metric", "DiffQP mean", "DiffQP SD", "StopGrad mean", "StopGrad SD"), outcome_rows)
    _write_table(table_root / "table_L3_paired_statistics.tex", ("Metric", "paired mean", "95% CI", "Cohen dz", "Holm p"), [
        (name, f'{item["paired_mean"]:.6g}', f'[{item["paired_bootstrap_95_ci"][0]:.6g}, {item["paired_bootstrap_95_ci"][1]:.6g}]', f'{item["cohen_dz"]:.6g}', f'{item["holm_adjusted_p"]:.6g}')
        for name, item in statistics_data.items()
    ])
    diffqp = [item["diagnostics"] for item in manifests if item["method"] == "SAFE-MAPPO-DIFFQP"]
    _write_table(table_root / "table_L4_diffqp_diagnostics.tex", ("Diagnostic", "Mean"), (
        ("gradient-valid fraction", f'{np.mean([x["qp_gradient_valid_fraction"] for x in diffqp]):.6g}'),
        ("fallback fraction", f'{np.mean([x["qp_fallback_fraction"] for x in diffqp]):.6g}'),
        ("QP-mediated actor-gradient norm", f'{np.mean([x["qp_mediated_actor_gradient_norm"] for x in diffqp]):.6g}'),
        ("LICQ failure frequency", f'{np.mean([x["licq_failure_frequency"] for x in diffqp]):.6g}'),
        ("strict-complementarity failure", f'{np.mean([x["strict_complementarity_failure_frequency"] for x in diffqp]):.6g}'),
    ))
    rows = []
    for method in protocol["primary_methods"]:
        selected = [item["diagnostics"] for item in manifests if item["method"] == method]
        rows.append((method, f'{np.mean([x["training_qp_forward_time_s"] for x in selected]):.6g}', f'{np.mean([x["qp_objective_backward_time_s"] for x in selected]):.6g}', f'{np.mean([x["ppo_update_time_s"] for x in selected]):.6g}', f'{np.mean([x["training_wall_clock_time_s"] for x in selected]):.6g}', f'{np.max([x["peak_resident_memory_MiB"] for x in selected]):.6g}'))
    _write_table(table_root / "table_L5_computational_overhead.tex", ("Method", "QP forward s", "QP backward s", "PPO update s", "wall-clock s", "peak RSS MiB"), rows)
    return [str(path.relative_to(result_root)).replace("\\", "/") for path in sorted(table_root.glob("*.tex"))]


def verify_integrity(result_root: Path = RESULT_ROOT, require_results: bool = True) -> dict[str, Any]:
    protocol = load_protocol()
    manifest_path = result_root / "result_manifest.json"
    if not manifest_path.exists():
        if require_results:
            raise FileNotFoundError("formal result manifest is missing")
        return {"status": "NOT_RUN", "registered_seed_count": len(protocol["training_seeds"])}
    campaign = json.loads(manifest_path.read_text(encoding="utf-8"))
    if campaign["protocol_sha256"] != file_hash(PROTOCOL_PATH):
        raise AssertionError("formal protocol hash mismatch")
    if campaign["physical_model_sha256"] != file_hash(FROZEN_CONFIG) or campaign["physical_model_sha256"] != EXPECTED_CONFIG_HASH:
        raise AssertionError("frozen physical-model hash mismatch")
    expected = {(method, int(seed)) for method in protocol["primary_methods"] for seed in protocol["training_seeds"]}
    actual = [(item["method"], int(item["seed"])) for item in campaign["active_runs"]]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise AssertionError("missing or duplicate formal method/seed run")
    manifests: list[dict[str, Any]] = []
    for item in campaign["active_runs"]:
        if item["status"] != "PASS" or item.get("paper_eligible") is not True:
            raise AssertionError("failed or paper-ineligible active run")
        run_manifest_path = result_root / item["run_manifest"]
        if file_hash(run_manifest_path) != item["run_manifest_sha256"]:
            raise AssertionError("run-manifest hash mismatch")
        run = json.loads(run_manifest_path.read_text(encoding="utf-8"))
        if run.get("training_mode") != "FORMAL" or run.get("data_provenance") != "literature_calibrated" or run.get("paper_eligible") is not True:
            raise AssertionError("DEBUG or invalid provenance in formal run")
        if run["protocol_sha256"] != campaign["protocol_sha256"] or run["training_config_sha256"] != campaign["training_config_sha256"]:
            raise AssertionError("run protocol/config hash mismatch")
        if run["forward_controller_sha256"] != campaign["forward_controller_sha256"]:
            raise AssertionError("run forward-controller hash mismatch")
        run_dir = run_manifest_path.parent
        if file_hash(run_dir / "updates.jsonl") != run["updates_sha256"] or file_hash(run_dir / "evaluations.jsonl") != run["evaluations_sha256"]:
            raise AssertionError("immutable raw-log hash mismatch")
        for update, expected_hash in run["checkpoint_hashes"].items():
            if file_hash(run_dir / f"checkpoint_update_{int(update):03d}.pt") != expected_hash:
                raise AssertionError("checkpoint hash mismatch")
        manifests.append(run)
    by_pair = {(run["method"], int(run["seed"])): run for run in manifests}
    for seed in protocol["training_seeds"]:
        left = by_pair[("SAFE-MAPPO-DIFFQP", int(seed))]
        right = by_pair[("SAFE-MAPPO-STOPGRAD", int(seed))]
        if left["initial_actor_checksum"] != right["initial_actor_checksum"] or left["initial_critic_checksum"] != right["initial_critic_checksum"]:
            raise AssertionError("pairwise initialization mismatch")
        if left["forward_controller_sha256"] != right["forward_controller_sha256"]:
            raise AssertionError("pairwise forward-controller mismatch")
    figures = campaign.get("figures", [])
    tables = campaign.get("tables", [])
    if len(figures) != 6 or len(tables) != 5 or any(not (result_root / path).exists() for path in figures + tables):
        raise AssertionError("formal figures/tables incomplete")
    for relative, expected_hash in {**campaign.get("figure_hashes", {}), **campaign.get("table_hashes", {})}.items():
        if file_hash(result_root / relative) != expected_hash:
            raise AssertionError("generated figure/table hash mismatch")
    if campaign["frozen_evidence_sha256"] != frozen_evidence_hash():
        raise AssertionError("frozen Phase 2M evidence changed")
    invalidated_count = len(campaign.get("invalidated_runs", []))
    return {
        "status": "PASS", "run_count": len(manifests), "failed_runs": 0,
        "invalidated_runs": invalidated_count, "pairwise_initialization_match": True,
        "forward_controller_hash_match": True,
    }


def _integrity_markdown(campaign: dict[str, Any], verification: dict[str, Any]) -> str:
    return f"""# Formal learning integrity report

- Status: **{verification['status']}**
- Protocol SHA-256: `{campaign['protocol_sha256']}`
- Training configuration SHA-256: `{campaign['training_config_sha256']}`
- Git commit: `{campaign['git_commit']}`
- Frozen physical-model SHA-256: `{campaign['physical_model_sha256']}`
- Forward-controller SHA-256: `{campaign['forward_controller_sha256']}`
- Active formal runs: {verification['run_count']}
- Failed runs: {verification['failed_runs']}
- Invalidated runs: {verification['invalidated_runs']}
- Pairwise actor/critic initialization match: {verification['pairwise_initialization_match']}
- Forward-controller hash match: {verification['forward_controller_hash_match']}
- DEBUG runs admitted: no
- Missing or duplicate seeds: no
- Raw logs and checkpoints: SHA-256 verified and read-only

The campaign uses only the registered Phase 2C-3 methods and seeds. This
report does not modify manuscript claims or any frozen Phase 2M evidence.
"""


def run_campaign(result_root: Path = RESULT_ROOT) -> dict[str, Any]:
    protocol = load_protocol()
    require_clean_code_tree()
    if result_root.exists():
        raise FileExistsError(f"formal output already exists and will not be overwritten: {result_root}")
    invalidated_runs: list[dict[str, Any]] = []
    invalidated_manifest = INVALIDATED_RESULT_ROOT / "result_manifest.json"
    if invalidated_manifest.exists():
        if file_hash(invalidated_manifest) != EXPECTED_INVALIDATED_MANIFEST_HASH:
            raise RuntimeError("invalidated timing-defect campaign hash mismatch")
        prior_verification = verify_integrity(INVALIDATED_RESULT_ROOT, require_results=True)
        prior = json.loads(invalidated_manifest.read_text(encoding="utf-8"))
        if prior_verification["run_count"] != 20:
            raise RuntimeError("invalidated campaign is incomplete")
        invalidated_runs = [
            {
                "run_id": item["run_id"], "method": item["method"], "seed": int(item["seed"]),
                "reason": "IMPLEMENTATION_ERROR_TRAINING_WALL_CLOCK_INCLUDED_CHECKPOINT_EVALUATION",
                "prior_result_manifest_sha256": EXPECTED_INVALIDATED_MANIFEST_HASH,
            }
            for item in prior["active_runs"]
        ]
    inputs = validate_frozen_inputs()
    if inputs.config_hash != EXPECTED_CONFIG_HASH:
        raise RuntimeError("frozen literature model failed the accepted hash gate")
    result_root.mkdir(parents=True)
    raw_root = result_root / "raw"
    raw_root.mkdir()
    metadata = {
        "git_commit": git_commit(),
        "protocol_sha256": file_hash(PROTOCOL_PATH),
        "training_config_sha256": canonical_hash(protocol),
        "physical_model_sha256": inputs.config_hash,
        "qp_config_sha256": canonical_hash({
            "forward": "accepted_2d_u_f_u_a", "controller": forward_controller_hash(),
        }),
        "forward_controller_sha256": forward_controller_hash(),
        "code_snapshot_sha256": code_snapshot_hash(),
        "frozen_evidence_sha256": frozen_evidence_hash(),
    }
    active_runs: list[dict[str, Any]] = []
    failed_runs: list[dict[str, Any]] = []
    for seed in protocol["training_seeds"]:
        for method in protocol["primary_methods"]:
            try:
                active_runs.append(run_one(method, int(seed), inputs, protocol, metadata, raw_root))
            except Exception as exc:
                failed_runs.append({"method": method, "seed": int(seed), "error": f"{type(exc).__name__}: {exc}"})
                dump_json(result_root / "failed_runs.json", failed_runs)
                raise
    manifests = [json.loads((result_root / item["run_manifest"]).read_text(encoding="utf-8")) for item in active_runs]
    statistics_data = paired_statistics(manifests, protocol)
    dump_json(result_root / "paired_statistics.json", statistics_data)
    alpha = float(protocol["statistical_analysis"]["alpha"])
    claims = {
        "return": _claim_status(statistics_data["final_evaluation_return"], "higher", alpha),
        "sample_efficiency": _claim_status(statistics_data["evaluation_return_auc"], "higher", alpha),
        "intervention_frequency": _claim_status(statistics_data["final_qp_intervention_frequency"], "lower", alpha),
        "intervention_magnitude": _claim_status(statistics_data["final_qp_intervention_norm"], "lower", alpha),
        "nominal_feasibility": _claim_status(statistics_data["final_nominal_hard_feasible_rate"], "higher", alpha),
        "safety": "NOT_SUPPORTED",
    }
    dump_json(result_root / "claim_decisions.json", claims)
    figures = generate_figures(result_root, manifests, statistics_data)
    tables = generate_tables(result_root, protocol, manifests, statistics_data)
    diffqp_diag = [item["diagnostics"] for item in manifests if item["method"] == "SAFE-MAPPO-DIFFQP"]
    stop_diag = [item["diagnostics"] for item in manifests if item["method"] == "SAFE-MAPPO-STOPGRAD"]
    campaign = {
        "schema_version": 1, "phase": "2C-3", "status": "PASS",
        **metadata,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "platform": {"python": platform.python_version(), "torch": torch.__version__, "os": platform.platform(), "device": "cpu"},
        "primary_methods": protocol["primary_methods"],
        "registered_training_seeds": protocol["training_seeds"],
        "registered_evaluation_seeds": protocol["evaluation_seeds"],
        "active_runs": active_runs, "failed_runs": failed_runs, "invalidated_runs": invalidated_runs,
        "statistics_file": "paired_statistics.json", "statistics_sha256": file_hash(result_root / "paired_statistics.json"),
        "claims_file": "claim_decisions.json", "claims_sha256": file_hash(result_root / "claim_decisions.json"),
        "figures": figures, "tables": tables,
        "figure_hashes": {path: file_hash(result_root / path) for path in figures},
        "table_hashes": {path: file_hash(result_root / path) for path in tables},
        "diffqp_gradient_valid_fraction": float(np.mean([item["qp_gradient_valid_fraction"] for item in diffqp_diag])),
        "diffqp_fallback_fraction": float(np.mean([item["qp_fallback_fraction"] for item in diffqp_diag])),
        "diffqp_training_wall_clock_mean_s": float(np.mean([item["training_wall_clock_time_s"] for item in diffqp_diag])),
        "stopgrad_training_wall_clock_mean_s": float(np.mean([item["training_wall_clock_time_s"] for item in stop_diag])),
        "frozen_model_evidence_changed": False,
    }
    dump_json(result_root / "result_manifest.json", campaign)
    verification = verify_integrity(result_root, require_results=True)
    integrity_path = ROOT / "docs/formal_learning_integrity.md"
    integrity_path.write_text(_integrity_markdown(campaign, verification), encoding="utf-8", newline="\n")
    return {**verification, "result_manifest_sha256": file_hash(result_root / "result_manifest.json"), "claims": claims, "statistics": statistics_data}


def main() -> int:
    result = run_campaign()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

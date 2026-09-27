"""Pre-registered native reproduction of Zhou et al., T-ITS 2025.

This module is intentionally independent of the project's heavy-duty safety
controller.  It implements the source paper's native mixed-autonomy plant,
reward, conformal predictor, cooperative barrier filter, and MAPPO training.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import subprocess
from typing import Any

import numpy as np
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = ROOT / "configs/external_baselines/zhou_tits_2025_protocol_v1.yaml"
RESULT_ROOT = ROOT / "results/external_baselines/zhou_tits_2025"
CAV_INDICES = (2, 4)
HDV_INDICES = (1, 3, 5, 6, 7)
DTYPE = torch.float64


def file_hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return sha256(encoded).hexdigest()


def dump_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def load_protocol() -> dict[str, Any]:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if protocol["campaign_id"] != "PHASE_2M_EXT_ZHOU":
        raise RuntimeError("external baseline protocol identity mismatch")
    if protocol["registered_before_native_execution"] is not True:
        raise RuntimeError("native reproduction protocol was not pre-registered")
    if protocol["result_driven_retuning_permitted"] is not False:
        raise RuntimeError("result-driven retuning must remain prohibited")
    return protocol


def git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


@dataclass
class NativeState:
    spacing_m: np.ndarray
    speed_mps: np.ndarray
    leader_speed_mps: float
    leader_acceleration_mps2: float
    acceleration_mps2: np.ndarray

    def copy(self) -> "NativeState":
        return NativeState(
            self.spacing_m.copy(),
            self.speed_mps.copy(),
            float(self.leader_speed_mps),
            float(self.leader_acceleration_mps2),
            self.acceleration_mps2.copy(),
        )


def desired_velocity(spacing_m: np.ndarray | float, env: dict[str, Any]) -> np.ndarray:
    spacing = np.asarray(spacing_m, dtype=np.float64)
    s_st = float(env["s_st_m"])
    s_go = float(env["s_go_m"])
    v_max = float(env["v_max_mps"])
    middle = 0.5 * v_max * (1.0 - np.cos(np.pi * (spacing - s_st) / (s_go - s_st)))
    return np.where(spacing <= s_st, 0.0, np.where(spacing >= s_go, v_max, middle))


def fvd_acceleration(
    spacing_m: np.ndarray | float,
    own_speed_mps: np.ndarray | float,
    predecessor_speed_mps: np.ndarray | float,
    env: dict[str, Any],
) -> np.ndarray:
    return (
        float(env["alpha"]) * (desired_velocity(spacing_m, env) - np.asarray(own_speed_mps))
        + float(env["beta"]) * (np.asarray(predecessor_speed_mps) - np.asarray(own_speed_mps))
    )


def predecessor_speeds(state: NativeState) -> np.ndarray:
    return np.concatenate(([state.leader_speed_mps], state.speed_mps[:-1]))


def normalized_global_state(state: NativeState) -> np.ndarray:
    return np.column_stack(
        ((state.spacing_m - 20.0) / 15.0, (state.speed_mps - 15.0) / 15.0)
    ).reshape(-1)


def actor_observations(state: NativeState) -> np.ndarray:
    base = normalized_global_state(state)
    return np.stack(
        (np.concatenate((base, [1.0, 0.0])), np.concatenate((base, [0.0, 1.0])))
    )


def cbf_values(state: NativeState, tau_s: float) -> np.ndarray:
    return state.spacing_m - tau_s * state.speed_mps


def zhou_reward(state: NativeState, spec: dict[str, Any]) -> tuple[float, dict[str, float]]:
    v = state.speed_mps
    reference = v[0]
    global_reward = -(v[1] - reference) ** 2
    for vehicle_index in (3, 5, 6, 7):
        global_reward -= (v[vehicle_index - 1] - reference) ** 2
    local_total = 0.0
    predecessor = predecessor_speeds(state)
    for vehicle_index in CAV_INDICES:
        k = vehicle_index - 1
        headway = state.spacing_m[k] / max(v[k], 1e-6)
        efficiency = -1.0 if headway >= float(spec["time_headway_threshold_s"]) else 0.0
        closing_speed = v[k] - predecessor[k]
        ttc = state.spacing_m[k] / closing_speed if closing_speed > 0.0 else math.inf
        safety = (
            math.log(max(ttc, 1e-12) / float(spec["ttc_threshold_s"]))
            if 0.0 <= ttc <= float(spec["ttc_threshold_s"])
            else 0.0
        )
        local_total += float(spec["efficiency_weight"]) * efficiency + float(spec["safety_weight"]) * safety
    total = float(spec["global_weight"]) * global_reward + float(spec["local_weight"]) * local_total
    return float(total), {"global": float(global_reward), "local": float(local_total)}


class NativePlatoonEnv:
    def __init__(self, protocol: dict[str, Any]) -> None:
        self.protocol = protocol
        self.env = protocol["native_environment"]
        self.dt = float(self.env["dt_s"])
        self.state = self.reset()

    def reset(self) -> NativeState:
        self.state = NativeState(
            np.full(7, float(self.env["equilibrium_spacing_m"]), dtype=np.float64),
            np.full(7, float(self.env["equilibrium_speed_mps"]), dtype=np.float64),
            float(self.env["equilibrium_speed_mps"]),
            0.0,
            np.zeros(7, dtype=np.float64),
        )
        return self.state.copy()

    def _leader_update(self, scenario: str, time_s: float, rng: np.random.Generator) -> tuple[float, float]:
        old_speed = self.state.leader_speed_mps
        if scenario == "training":
            new_speed = max(0.0, 15.0 + float(rng.normal(0.0, 0.2)))
            return new_speed, (new_speed - old_speed) / self.dt
        if scenario == "scenario1":
            cfg = self.protocol["native_evaluation"]["scenario1"]
            start = float(cfg["brake_start_s"])
            brake_end = start + float(cfg["brake_duration_s"])
            recovery_end = brake_end + float(cfg["recovery_duration_s"])
            if start <= time_s < brake_end:
                acceleration = float(cfg["brake_acceleration_mps2"])
            elif brake_end <= time_s < recovery_end:
                lost_speed = -float(cfg["brake_acceleration_mps2"]) * float(cfg["brake_duration_s"])
                acceleration = lost_speed / float(cfg["recovery_duration_s"])
            else:
                acceleration = 0.0
        elif scenario == "sine":
            cfg = self.protocol["native_evaluation"]["sine"]
            acceleration = float(cfg["acceleration_amplitude_mps2"]) * math.sin(
                float(cfg["angular_frequency_rad_s"]) * time_s
            )
        else:
            acceleration = 0.0
        return max(0.0, old_speed + self.dt * acceleration), acceleration

    def step(
        self,
        safe_cav_acceleration: np.ndarray,
        scenario: str,
        time_s: float,
        rng: np.random.Generator,
    ) -> NativeState:
        old = self.state
        new_leader_speed, leader_acceleration = self._leader_update(scenario, time_s, rng)
        predecessor = predecessor_speeds(old)
        acceleration = np.zeros(7, dtype=np.float64)
        for vehicle_index in HDV_INDICES:
            k = vehicle_index - 1
            acceleration[k] = float(
                fvd_acceleration(old.spacing_m[k], old.speed_mps[k], predecessor[k], self.env)
            )
        acceleration[1] = float(safe_cav_acceleration[0])
        acceleration[3] = float(safe_cav_acceleration[1])
        if scenario == "scenario2":
            cfg = self.protocol["native_evaluation"]["scenario2"]
            if float(cfg["start_s"]) <= time_s < float(cfg["start_s"]) + float(cfg["duration_disturbance_s"]):
                acceleration[int(cfg["hdv_index"]) - 1] = float(cfg["acceleration_mps2"])
        new_spacing = old.spacing_m + self.dt * (predecessor - old.speed_mps)
        new_speed = np.maximum(0.0, old.speed_mps + self.dt * acceleration)
        self.state = NativeState(new_spacing, new_speed, new_leader_speed, leader_acceleration, acceleration)
        return self.state.copy()


class AccelerationPredictor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(4, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 1)
        )
        self.register_buffer("input_mean", torch.zeros(4, dtype=DTYPE))
        self.register_buffer("input_scale", torch.ones(4, dtype=DTYPE))
        self.register_buffer("target_mean", torch.zeros((), dtype=DTYPE))
        self.register_buffer("target_scale", torch.ones((), dtype=DTYPE))
        self.to(dtype=DTYPE)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        normalized = self.network((features - self.input_mean) / self.input_scale).squeeze(-1)
        return normalized * self.target_scale + self.target_mean

    def predict(self, features: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            values = self(torch.as_tensor(features, dtype=DTYPE))
        return values.cpu().numpy()


def _predictor_dataset(
    sample_count: int, rng: np.random.Generator, env: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    features = np.empty((sample_count, len(HDV_INDICES), 4), dtype=np.float64)
    spacing = rng.uniform(5.0, 35.0, size=(sample_count, len(HDV_INDICES)))
    own_speed = rng.uniform(0.0, 30.0, size=(sample_count, len(HDV_INDICES)))
    predecessor_speed = rng.uniform(0.0, 30.0, size=(sample_count, len(HDV_INDICES)))
    previous_acceleration = rng.uniform(-5.0, 5.0, size=(sample_count, len(HDV_INDICES)))
    features[..., 0] = spacing
    features[..., 1] = own_speed
    features[..., 2] = predecessor_speed
    features[..., 3] = previous_acceleration
    targets = fvd_acceleration(spacing, own_speed, predecessor_speed, env)
    return features, np.asarray(targets, dtype=np.float64)


def train_conformal_predictor(
    protocol: dict[str, Any], output_dir: Path
) -> tuple[AccelerationPredictor, dict[str, Any]]:
    spec = protocol["conformal_prediction"]
    rng = np.random.default_rng(int(spec["seed"]))
    torch.manual_seed(int(spec["seed"]))
    train_x, train_y = _predictor_dataset(int(spec["training_samples"]), rng, protocol["native_environment"])
    cal_x, cal_y = _predictor_dataset(int(spec["calibration_samples"]), rng, protocol["native_environment"])
    test_x, test_y = _predictor_dataset(int(spec["test_samples"]), rng, protocol["native_environment"])
    model = AccelerationPredictor()
    flat_x = torch.as_tensor(train_x.reshape(-1, 4), dtype=DTYPE)
    flat_y = torch.as_tensor(train_y.reshape(-1), dtype=DTYPE)
    model.input_mean.copy_(flat_x.mean(dim=0))
    model.input_scale.copy_(flat_x.std(dim=0).clamp_min(1e-6))
    model.target_mean.copy_(flat_y.mean())
    model.target_scale.copy_(flat_y.std().clamp_min(1e-6))
    optimizer = torch.optim.Adam(model.parameters(), lr=float(spec["learning_rate"]))
    for _ in range(int(spec["epochs"])):
        optimizer.zero_grad()
        normalized_error = (model(flat_x) - flat_y) / model.target_scale
        loss = torch.mean(normalized_error ** 2)
        loss.backward()
        optimizer.step()
    cal_prediction = model.predict(cal_x.reshape(-1, 4)).reshape(cal_y.shape)
    cal_scores = np.max(np.abs(cal_y - cal_prediction), axis=1)
    n_cal = len(cal_scores)
    p = int(math.ceil((n_cal + 1) * (1.0 - float(spec["failure_probability"]))))
    augmented = np.concatenate((np.sort(cal_scores), [math.inf]))
    conformal_bound = float(augmented[p - 1])
    test_prediction = model.predict(test_x.reshape(-1, 4)).reshape(test_y.shape)
    test_scores = np.max(np.abs(test_y - test_prediction), axis=1)
    metrics = {
        "train_normalized_mse": float(loss.detach().cpu()),
        "train_mse": float(loss.detach().cpu() * model.target_scale.detach().cpu() ** 2),
        "calibration_sample_count": n_cal,
        "quantile_index_one_based": p,
        "conformal_bound_mps2": conformal_bound,
        "test_joint_coverage": float(np.mean(test_scores <= conformal_bound)),
        "test_mean_nonconformity_mps2": float(np.mean(test_scores)),
        "sets_are_disjoint_by_generation": True,
    }
    torch.save({"model": model.state_dict(), "metrics": metrics}, output_dir / "conformal_predictor.pt")
    dump_json(output_dir / "conformal_metrics.json", metrics)
    return model, metrics


class CooperativeSafetyLayer:
    def __init__(
        self,
        protocol: dict[str, Any],
        predictor: AccelerationPredictor,
        conformal_bound_mps2: float,
    ) -> None:
        self.protocol = protocol
        self.spec = protocol["cooperative_cbf"]
        self.predictor = predictor
        self.bound = float(conformal_bound_mps2)

    def _predicted_hdv_acceleration(self, state: NativeState) -> dict[int, float]:
        predecessor = predecessor_speeds(state)
        rows = []
        for i in HDV_INDICES:
            k = i - 1
            rows.append([state.spacing_m[k], state.speed_mps[k], predecessor[k], state.acceleration_mps2[k]])
        prediction = self.predictor.predict(np.asarray(rows, dtype=np.float64))
        return {i: float(prediction[k]) for k, i in enumerate(HDV_INDICES)}

    def apply(self, nominal_action: np.ndarray, state: NativeState) -> tuple[np.ndarray, dict[str, Any]]:
        tau = float(self.spec["tau_s"])
        coupling = float(self.spec["coupling_k"])
        gain = float(self.spec["class_k_gain"])
        lower = float(self.spec["acceleration_min_mps2"])
        upper = float(self.spec["acceleration_max_mps2"])
        h = cbf_values(state, tau)
        predecessor = predecessor_speeds(state)
        action = np.clip(np.asarray(nominal_action, dtype=np.float64), lower, upper)
        cav_upper = np.full(2, upper, dtype=np.float64)
        for q, i in enumerate(CAV_INDICES):
            k = i - 1
            cav_upper[q] = min(upper, (predecessor[k] - state.speed_mps[k] + gain * h[k]) / tau)
        prediction = self._predicted_hdv_acceleration(state)
        soft_rows: list[tuple[np.ndarray, float, int]] = []
        for i in (3, 5, 6, 7):
            preceding_cavs = [j for j in CAV_INDICES if j < i]
            h_sufficient = h[i - 1] - coupling * sum(h[j - 1] for j in preceding_cavs)
            base = (
                predecessor[i - 1]
                - state.speed_mps[i - 1]
                - tau * prediction[i]
                - coupling * sum(predecessor[j - 1] - state.speed_mps[j - 1] for j in preceding_cavs)
                + gain * h_sufficient
            )
            coefficient = np.asarray(
                [tau * coupling if j in preceding_cavs else 0.0 for j in CAV_INDICES],
                dtype=np.float64,
            )
            robust_margin = self.bound * tau * (1.0 + coupling * len(preceding_cavs))
            soft_rows.append((coefficient, robust_margin - base, i))
        for _ in range(32):
            before = action.copy()
            action = np.minimum(np.clip(action, lower, upper), cav_upper)
            for coefficient, required, _i in soft_rows:
                value = float(coefficient @ action)
                norm_squared = float(coefficient @ coefficient)
                if value < required and norm_squared > 0.0:
                    action += (required - value) * coefficient / norm_squared
                    action = np.minimum(np.clip(action, lower, upper), cav_upper)
            if float(np.max(np.abs(action - before))) < 1e-10:
                break
        cav_residuals = []
        for q, i in enumerate(CAV_INDICES):
            k = i - 1
            cav_residuals.append(
                predecessor[k] - state.speed_mps[k] - tau * action[q] + gain * h[k]
            )
        relaxations = [max(0.0, required - float(coefficient @ action)) for coefficient, required, _i in soft_rows]
        return action, {
            "nominal_action": list(map(float, nominal_action)),
            "safe_correction": list(map(float, action - nominal_action)),
            "cav_min_residual": float(min(cav_residuals)),
            "relaxation_max": float(max(relaxations, default=0.0)),
            "relaxation_sum": float(sum(relaxations)),
        }


def orthogonal_linear(layer: nn.Linear, gain: float = 1.0) -> None:
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.zeros_(layer.bias)


class SharedActor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(16, 64)
        self.fc2 = nn.Linear(64, 64)
        self.mean = nn.Linear(64, 1)
        self.log_std = nn.Parameter(torch.zeros(1, dtype=DTYPE))
        orthogonal_linear(self.fc1)
        orthogonal_linear(self.fc2)
        orthogonal_linear(self.mean, gain=0.01)
        self.to(dtype=DTYPE)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        hidden = torch.tanh(self.fc1(observation))
        hidden = torch.tanh(self.fc2(hidden))
        return 5.0 * torch.tanh(self.mean(hidden)).squeeze(-1)

    def distribution(self, observation: torch.Tensor) -> torch.distributions.Normal:
        mean = self(observation)
        std = torch.exp(self.log_std.clamp(-5.0, 2.0)).expand_as(mean)
        return torch.distributions.Normal(mean, std)


class CentralCritic(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(14, 64)
        self.fc2 = nn.Linear(64, 64)
        self.value = nn.Linear(64, 1)
        orthogonal_linear(self.fc1)
        orthogonal_linear(self.fc2)
        orthogonal_linear(self.value)
        self.to(dtype=DTYPE)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        hidden = torch.tanh(self.fc1(state))
        hidden = torch.tanh(self.fc2(hidden))
        return self.value(hidden).squeeze(-1)


@dataclass
class TrainingArtifacts:
    actor: SharedActor
    critic: CentralCritic
    episode_returns: list[float]
    optimizer_updates: int
    total_environment_steps: int


def _ppo_update(
    actor: SharedActor,
    critic: CentralCritic,
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    segment: list[dict[str, Any]],
    next_state: NativeState,
    spec: dict[str, Any],
) -> dict[str, float]:
    rewards = np.asarray([item["reward"] for item in segment], dtype=np.float64)
    values = np.asarray([item["value"] for item in segment], dtype=np.float64)
    dones = np.asarray([item["done"] for item in segment], dtype=np.float64)
    with torch.no_grad():
        bootstrap = 0.0 if dones[-1] else float(
            critic(torch.as_tensor(normalized_global_state(next_state)[None, :], dtype=DTYPE))[0]
        )
    advantages = np.zeros_like(rewards)
    gae = 0.0
    gamma = float(spec["discount_factor"])
    lam = float(spec["gae_lambda"])
    for index in range(len(segment) - 1, -1, -1):
        next_value = bootstrap if index == len(segment) - 1 else values[index + 1]
        nonterminal = 1.0 - dones[index]
        delta = rewards[index] + gamma * next_value * nonterminal - values[index]
        gae = delta + gamma * lam * nonterminal * gae
        advantages[index] = gae
    returns = advantages + values
    observation = torch.as_tensor(np.concatenate([item["observation"] for item in segment]), dtype=DTYPE)
    action = torch.as_tensor(np.concatenate([item["nominal_action"] for item in segment]), dtype=DTYPE)
    old_log_prob = torch.as_tensor(np.concatenate([item["log_prob"] for item in segment]), dtype=DTYPE)
    advantage = torch.as_tensor(np.repeat(advantages, 2), dtype=DTYPE)
    advantage = (advantage - advantage.mean()) / (advantage.std(unbiased=False) + 1e-8)
    critic_state = torch.as_tensor(
        np.repeat(np.stack([item["central_state"] for item in segment]), 2, axis=0), dtype=DTYPE
    )
    return_target = torch.as_tensor(np.repeat(returns, 2), dtype=DTYPE)
    losses = []
    for _ in range(int(spec["ppo_epochs"])):
        distribution = actor.distribution(observation)
        new_log_prob = distribution.log_prob(action)
        ratio = torch.exp((new_log_prob - old_log_prob).clamp(-60.0, 60.0))
        surrogate_1 = ratio * advantage
        surrogate_2 = ratio.clamp(1.0 - float(spec["ppo_clip"]), 1.0 + float(spec["ppo_clip"])) * advantage
        actor_loss = -torch.minimum(surrogate_1, surrogate_2).mean() - float(spec["entropy_coefficient"]) * distribution.entropy().mean()
        actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(actor.parameters(), float(spec["gradient_clip_norm"]))
        actor_optimizer.step()
        value = critic(critic_state)
        critic_loss = torch.mean((value - return_target) ** 2)
        critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(critic.parameters(), float(spec["gradient_clip_norm"]))
        critic_optimizer.step()
        losses.append((float(actor_loss.detach()), float(critic_loss.detach())))
    means = np.mean(np.asarray(losses), axis=0)
    return {"actor_loss": float(means[0]), "critic_loss": float(means[1])}


def train_native_mappo(
    protocol: dict[str, Any], safety_layer: CooperativeSafetyLayer, output_dir: Path
) -> TrainingArtifacts:
    spec = protocol["mappo"]
    seed = int(spec["training_seed"])
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    actor = SharedActor()
    critic = CentralCritic()
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=float(spec["actor_learning_rate"]), eps=1e-5)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=float(spec["critic_learning_rate"]), eps=1e-5)
    env = NativePlatoonEnv(protocol)
    pending: list[dict[str, Any]] = []
    returns: list[float] = []
    update_metrics: list[dict[str, float]] = []
    batch_time_steps = int(spec["batch_size_agent_transitions"]) // 2
    total_steps = int(spec["episodes"]) * int(spec["steps_per_episode"])
    completed_steps = 0
    for episode in range(int(spec["episodes"])):
        state = env.reset()
        episode_return = 0.0
        for step in range(int(spec["steps_per_episode"])):
            observation_np = actor_observations(state)
            central_np = normalized_global_state(state)
            observation = torch.as_tensor(observation_np, dtype=DTYPE)
            with torch.no_grad():
                distribution = actor.distribution(observation)
                nominal = distribution.sample().clamp(-5.0, 5.0)
                log_probability = distribution.log_prob(nominal)
                value = critic(torch.as_tensor(central_np[None, :], dtype=DTYPE))[0]
            nominal_np = nominal.cpu().numpy()
            safe_action, _diagnostic = safety_layer.apply(nominal_np, state)
            next_state = env.step(safe_action, "training", step * env.dt, rng)
            reward, _terms = zhou_reward(next_state, protocol["reward"])
            done = step == int(spec["steps_per_episode"]) - 1
            pending.append(
                {
                    "observation": observation_np,
                    "central_state": central_np,
                    "nominal_action": nominal_np,
                    "log_prob": log_probability.cpu().numpy(),
                    "reward": reward,
                    "value": float(value),
                    "done": done,
                }
            )
            state = next_state
            episode_return += reward
            completed_steps += 1
            if len(pending) == batch_time_steps:
                progress = completed_steps / total_steps
                lr_factor = max(0.0, 1.0 - progress)
                actor_optimizer.param_groups[0]["lr"] = float(spec["actor_learning_rate"]) * lr_factor
                critic_optimizer.param_groups[0]["lr"] = float(spec["critic_learning_rate"]) * lr_factor
                update_metrics.append(
                    _ppo_update(actor, critic, actor_optimizer, critic_optimizer, pending, state, spec)
                )
                pending = []
        returns.append(float(episode_return))
        if (episode + 1) % 50 == 0:
            print(
                f"NATIVE_TRAIN episode={episode + 1}/{spec['episodes']} "
                f"mean_return_50={np.mean(returns[-50:]):.6f} updates={len(update_metrics)}",
                flush=True,
            )
    checkpoint = {
        "actor": actor.state_dict(),
        "critic": critic.state_dict(),
        "actor_optimizer": actor_optimizer.state_dict(),
        "critic_optimizer": critic_optimizer.state_dict(),
        "seed": seed,
        "episodes": int(spec["episodes"]),
        "steps_per_episode": int(spec["steps_per_episode"]),
    }
    torch.save(checkpoint, output_dir / "native_mappo_checkpoint.pt")
    training_summary = {
        "episodes": len(returns),
        "total_environment_steps": completed_steps,
        "optimizer_updates": len(update_metrics),
        "first_25_mean_return": float(np.mean(returns[:25])),
        "last_25_mean_return": float(np.mean(returns[-25:])),
        "final_actor_loss": update_metrics[-1]["actor_loss"] if update_metrics else None,
        "final_critic_loss": update_metrics[-1]["critic_loss"] if update_metrics else None,
        "episode_returns": returns,
    }
    dump_json(output_dir / "native_training_summary.json", training_summary)
    return TrainingArtifacts(actor, critic, returns, len(update_metrics), completed_steps)


def _evaluation_steps(protocol: dict[str, Any], scenario: str) -> int:
    duration = float(protocol["native_evaluation"][scenario]["duration_s"])
    return int(round(duration / float(protocol["native_environment"]["dt_s"])))


def evaluate_native_scenario(
    protocol: dict[str, Any],
    actor: SharedActor,
    safety_layer: CooperativeSafetyLayer,
    scenario: str,
    output_dir: Path,
) -> dict[str, Any]:
    rng = np.random.default_rng(int(protocol["mappo"]["training_seed"]) + 100 + len(scenario))
    env = NativePlatoonEnv(protocol)
    state = env.reset()
    tau = float(protocol["cooperative_cbf"]["tau_s"])
    trace = []
    runtime_ns = []
    import time
    for step in range(_evaluation_steps(protocol, scenario)):
        observation = torch.as_tensor(actor_observations(state), dtype=DTYPE)
        start = time.perf_counter_ns()
        with torch.no_grad():
            nominal = actor(observation).cpu().numpy()
        safe_action, diagnostic = safety_layer.apply(nominal, state)
        runtime_ns.append(time.perf_counter_ns() - start)
        state = env.step(safe_action, scenario, step * env.dt, rng)
        h = cbf_values(state, tau)
        trace.append(
            {
                "time_s": (step + 1) * env.dt,
                "leader_speed_mps": state.leader_speed_mps,
                "leader_acceleration_mps2": state.leader_acceleration_mps2,
                "spacing_m": list(map(float, state.spacing_m)),
                "speed_mps": list(map(float, state.speed_mps)),
                "acceleration_mps2": list(map(float, state.acceleration_mps2)),
                "nominal_cav_acceleration_mps2": list(map(float, nominal)),
                "safe_cav_acceleration_mps2": list(map(float, safe_action)),
                "cbf": list(map(float, h)),
                "qp_relaxation_max": diagnostic["relaxation_max"],
            }
        )
    spacing = np.asarray([row["spacing_m"] for row in trace])
    speed = np.asarray([row["speed_mps"] for row in trace])
    leader_speed = np.asarray([row["leader_speed_mps"] for row in trace])
    h_values = np.asarray([row["cbf"] for row in trace])
    cav_headway = spacing[:, [1, 3]] / np.maximum(speed[:, [1, 3]], 1e-6)
    aave = np.mean(np.abs(speed - leader_speed[:, None]))
    metrics = {
        "scenario": scenario,
        "average_cav_time_headway_s": float(np.mean(cav_headway)),
        "aave_mps": float(aave),
        "minimum_spacing_m": float(np.min(spacing)),
        "minimum_cbf": float(np.min(h_values)),
        "collision_count": int(np.sum(np.min(spacing, axis=1) <= 0.0)),
        "maximum_qp_relaxation": float(max(row["qp_relaxation_max"] for row in trace)),
        "controller_runtime_mean_ms": float(np.mean(runtime_ns) / 1e6),
        "controller_runtime_p99_ms": float(np.quantile(runtime_ns, 0.99) / 1e6),
        "step_count": len(trace),
    }
    trace_path = output_dir / f"native_{scenario}_trace.jsonl"
    with trace_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in trace:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    run_payload = {
        "campaign_id": protocol["campaign_id"],
        "paper_doi": protocol["paper"]["doi"],
        "arxiv_id": protocol["paper"]["arxiv_id"],
        "implementation_version": protocol["implementation_version"],
        "git_commit": git_commit(),
        "protocol_hash": file_hash(PROTOCOL_PATH),
        "scenario_hash": canonical_hash(protocol["native_evaluation"][scenario]),
        "seed": int(protocol["mappo"]["training_seed"]),
        "paper_eligible": False,
        "mode": "native",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "metrics": metrics,
        "trace_sha256": file_hash(trace_path),
    }
    run_payload["result_hash"] = canonical_hash(run_payload)
    dump_json(output_dir / f"native_{scenario}_result.json", run_payload)
    return run_payload


def native_gate(protocol: dict[str, Any], runs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    gate = protocol["native_evaluation"]["pass_gate"]
    target = protocol["native_evaluation"]["published_sanity_target"]
    checks = {
        "scenario1_collision_free": runs["scenario1"]["metrics"]["collision_count"] == int(gate["scenario1_and_scenario2_collision_count"]),
        "scenario2_collision_free": runs["scenario2"]["metrics"]["collision_count"] == int(gate["scenario1_and_scenario2_collision_count"]),
        "scenario1_cbf": runs["scenario1"]["metrics"]["minimum_cbf"] >= float(gate["scenario1_and_scenario2_minimum_cbf_lower_bound"]),
        "scenario2_cbf": runs["scenario2"]["metrics"]["minimum_cbf"] >= float(gate["scenario1_and_scenario2_minimum_cbf_lower_bound"]),
        "sine_headway_comparable": abs(runs["sine"]["metrics"]["average_cav_time_headway_s"] - float(target["average_cav_time_headway_s"])) <= float(gate["sine_headway_absolute_difference_max_s"]),
        "sine_aave_comparable": abs(runs["sine"]["metrics"]["aave_mps"] - float(target["aave_mps"])) <= float(gate["sine_aave_absolute_difference_max_mps"]),
    }
    finite_values = [
        value
        for run in runs.values()
        for key, value in run["metrics"].items()
        if isinstance(value, (float, int)) and key != "collision_count"
    ]
    checks["all_metrics_finite"] = bool(np.all(np.isfinite(finite_values)))
    passed = all(checks.values())
    return {
        "status": "PASS" if passed else "FAIL_MATERIAL_NATIVE_REPRODUCTION_MISMATCH",
        "checks": checks,
        "adapted_transfer_authorized": passed,
        "result_driven_retuning_performed": False,
    }


def run_native_campaign() -> dict[str, Any]:
    protocol = load_protocol()
    native_dir = RESULT_ROOT / "native_v1"
    if (native_dir / "native_result_manifest.json").exists():
        raise FileExistsError("native result is frozen; create a versioned correction instead of overwriting")
    native_dir.mkdir(parents=True, exist_ok=True)
    predictor, conformal = train_conformal_predictor(protocol, native_dir)
    safety_layer = CooperativeSafetyLayer(protocol, predictor, conformal["conformal_bound_mps2"])
    training = train_native_mappo(protocol, safety_layer, native_dir)
    runs = {
        name: evaluate_native_scenario(protocol, training.actor, safety_layer, name, native_dir)
        for name in ("scenario1", "scenario2", "sine")
    }
    gate = native_gate(protocol, runs)
    dump_json(native_dir / "native_gate.json", gate)
    files = sorted(path for path in native_dir.iterdir() if path.is_file())
    manifest = {
        "campaign_id": protocol["campaign_id"],
        "mode": "native",
        "paper_doi": protocol["paper"]["doi"],
        "arxiv_id": protocol["paper"]["arxiv_id"],
        "implementation_version": protocol["implementation_version"],
        "git_commit": git_commit(),
        "protocol_hash": file_hash(PROTOCOL_PATH),
        "seed": int(protocol["mappo"]["training_seed"]),
        "paper_eligible": False,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "training": {
            "episodes": len(training.episode_returns),
            "environment_steps": training.total_environment_steps,
            "optimizer_updates": training.optimizer_updates,
        },
        "native_gate": gate,
        "file_hashes": {path.name: file_hash(path) for path in files},
        "existing_frozen_evidence_modified": False,
    }
    manifest["result_hash"] = canonical_hash(manifest)
    manifest_path = native_dir / "native_result_manifest.json"
    dump_json(manifest_path, manifest)
    for path in native_dir.iterdir():
        if path.is_file():
            os.chmod(path, 0o444)
    print(json.dumps({"native_gate": gate, "manifest_sha256": file_hash(manifest_path)}, indent=2))
    return manifest

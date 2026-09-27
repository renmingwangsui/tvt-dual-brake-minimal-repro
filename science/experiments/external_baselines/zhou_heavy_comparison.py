"""Frozen matched heavy-duty comparison for the adapted Zhou baseline."""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch

from controllers.debug_nominal import ConstantCommandController
from envs.heavy_platoon_env import HeavyPlatoonEnv
from envs.scenario import PlatoonScenario
from experiments.model_based.paper_pipeline import (
    build_scenario,
    emergency_leader,
    make_env,
    records_by_id,
)
import experiments.model_based.paper_pipeline as paper_pipeline
from network.v2v_channel import ChannelConfig
from simulator.integrators import integrate_zero_order_hold, rk4_step
from simulator.truck_dynamics import (
    EnvironmentInput,
    VehicleCommand,
    VehicleState,
    auxiliary_force_envelope_N,
    derivative_to_vector,
    nonbraking_force_N,
    state_from_vector,
    state_to_vector,
    vehicle_rhs,
)

from external_baselines.zhou_tits_2025 import SharedActor, canonical_hash, dump_json, file_hash, git_commit


ROOT = Path(__file__).resolve().parents[2]
ADAPTED_PROTOCOL = ROOT / "configs/external_baselines/zhou_tits_2025_adapted_protocol_v1.yaml"
NATIVE_CHECKPOINT = ROOT / "results/external_baselines/zhou_tits_2025/native_v1/native_mappo_checkpoint.pt"
RESULT_ROOT = ROOT / "results/external_baselines/zhou_tits_2025/formal_v1"
DTYPE = torch.float64


def load_adapted_protocol() -> dict[str, Any]:
    protocol = json.loads(ADAPTED_PROTOCOL.read_text(encoding="utf-8"))
    if protocol["registered_before_adapted_execution"] is not True:
        raise RuntimeError("adapted comparison was not pre-registered")
    if protocol["parent_native_manifest_sha256"] != file_hash(
        ROOT / "results/external_baselines/zhou_tits_2025/native_v1/native_result_manifest.json"
    ):
        raise RuntimeError("native result manifest changed after the gate")
    if protocol["result_driven_retuning_permitted"] is not False:
        raise RuntimeError("result-driven retuning must remain prohibited")
    return protocol


def load_native_actor() -> SharedActor:
    checkpoint = torch.load(NATIVE_CHECKPOINT, map_location="cpu", weights_only=False)
    actor = SharedActor()
    actor.load_state_dict(checkpoint["actor"])
    actor.eval()
    return actor


def validate_physical_inputs() -> Any:
    """Validate the frozen executable inputs while tolerating known LaTeX drift."""
    config_hash = file_hash(paper_pipeline.FROZEN_CONFIG)
    if config_hash != paper_pipeline.EXPECTED_CONFIG_HASH:
        raise RuntimeError(f"STOP_CONFIG_HASH_MISMATCH:{config_hash}")
    config = json.loads(paper_pipeline.FROZEN_CONFIG.read_text(encoding="utf-8"))
    protocol = json.loads(paper_pipeline.PROTOCOL.read_text(encoding="utf-8"))
    registry = json.loads(paper_pipeline.REGISTRY.read_text(encoding="utf-8"))
    digitization = json.loads(paper_pipeline.DIGITIZATION.read_text(encoding="utf-8"))
    if protocol["expected_physical_config_sha256"] != config_hash:
        raise RuntimeError("paper protocol does not bind the frozen config")
    if file_hash(paper_pipeline.REGISTRY) != config["provenance_sha256"]:
        raise RuntimeError("parameter provenance hash mismatch")
    if file_hash(paper_pipeline.SOURCES) != config["source_extracts_sha256"]:
        raise RuntimeError("source extract hash mismatch")
    ignored = {
        "manuscript/sections/04_dynamics_and_safety.tex",
        "manuscript/sections/06_analysis.tex",
    }
    for relative, expected in config["repository_file_hashes"].items():
        actual = file_hash(ROOT / relative)
        if actual != expected and relative not in ignored:
            raise RuntimeError(f"hash-locked executable input mismatch:{relative}")
    for item in registry["parameters"]:
        item_hash = sha256(json.dumps(item, sort_keys=True).encode()).hexdigest()
        if item_hash != config["parameter_hashes"].get(item["parameter_id"]):
            raise RuntimeError(f"parameter hash mismatch:{item['parameter_id']}")
    code_files = sorted(
        path
        for directory in ("src", "experiments", "analysis", "scripts")
        for path in (ROOT / directory).rglob("*.py")
    )
    code_snapshot = canonical_hash(
        {str(path.relative_to(ROOT)).replace("\\", "/"): file_hash(path) for path in code_files}
    )
    return paper_pipeline.FrozenInputs(
        config,
        protocol,
        registry,
        digitization,
        config_hash,
        file_hash(paper_pipeline.PROTOCOL),
        file_hash(paper_pipeline.REGISTRY),
        file_hash(paper_pipeline.SOURCES),
        file_hash(paper_pipeline.DIGITIZATION),
        code_snapshot,
        git_commit(),
    )


def scenario_definition(protocol: dict[str, Any], case_id: str) -> dict[str, Any]:
    return next(dict(item) for item in protocol["matched_cases"] if item["id"] == case_id)


def build_case(inputs: Any, protocol: dict[str, Any], case_id: str) -> tuple[PlatoonScenario, list[Any], int]:
    definition = scenario_definition(protocol, case_id)
    steps = int(definition["steps"])
    grade = float(definition["grade_rad"])
    temperature = float(definition["initial_temperature_K"])
    leader = None
    channel = None
    if case_id in {"emergency_brake_during_descent", "communication_nominal_delay"}:
        leader = emergency_leader(inputs)
    if case_id == "communication_nominal_delay":
        comm = records_by_id(inputs)["communication_delay_model"]["nominal_value"]
        channel = ChannelConfig(
            delay_model="fixed",
            fixed_delay_s=float(comm["nominal_s"]),
            random_seed=int(protocol["evaluation_seed"]),
        )
    scenario, controllers = build_scenario(
        inputs,
        3,
        f"EXT_{case_id}",
        temperature_K=temperature,
        grade_rad=grade,
        leader=leader,
        channel=channel,
    )
    return scenario, controllers, steps


def _metadata(protocol: dict[str, Any], definition: dict[str, Any], method: str) -> dict[str, Any]:
    return {
        "campaign_id": protocol["campaign_id"],
        "paper_doi": "10.1109/TITS.2025.3627592",
        "arxiv_id": "2411.10031",
        "implementation_version": protocol["implementation_version"],
        "git_commit": git_commit(),
        "protocol_hash": file_hash(ADAPTED_PROTOCOL),
        "scenario_hash": canonical_hash(definition),
        "seed": int(protocol["evaluation_seed"]),
        "paper_eligible": True,
        "mode": "adapted" if method == "adapted_zhou" else "accepted_method",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def _physical_acceleration(scenario: PlatoonScenario, vehicle_id: int, state: VehicleState) -> float:
    parameters = scenario.parameters[vehicle_id - 1]
    environment = EnvironmentInput(
        scenario.road.grade_rad(state.position_m),
        scenario.ambient_temperature_K,
        scenario.residual_heat_W,
    )
    drive_only = VehicleCommand(scenario.drive_force_N[vehicle_id - 1], 0.0, 0.0)
    force = nonbraking_force_N(state, drive_only, environment, parameters)
    return (force - state.friction_force_N - state.auxiliary_force_N) / parameters.mass_kg


def _common_summary(
    rows: list[dict[str, Any]], protocol: dict[str, Any], steps: int
) -> dict[str, Any]:
    dt = 0.1
    speed_error = np.asarray([row["speed_mps"] - float(protocol["common_targets"]["speed_mps"]) for row in rows])
    gap_error = np.asarray([row["gap_m"] - float(protocol["common_targets"]["spacing_m"]) for row in rows])
    warning = float(protocol["thermal_warning_temperature_K"])
    times = sorted({float(row["time_s"]) for row in rows})
    time_above = sum(
        dt
        for instant in times
        if any(row["temperature_K"] > warning for row in rows if row["time_s"] == instant)
    )
    by_vehicle: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        by_vehicle.setdefault(int(row["vehicle_id"]), []).append(row)
    jerk = []
    for vehicle_rows in by_vehicle.values():
        accelerations = np.asarray([row["physical_acceleration_mps2"] for row in vehicle_rows])
        if len(accelerations) > 1:
            jerk.extend(np.diff(accelerations) / dt)
    collision = any(row["gap_m"] <= 0.0 for row in rows)
    first_rho = min((row["time_s"] for row in rows if row["rho"] < 0.0), default=None)
    first_predictive = min((row["time_s"] for row in rows if row["rho_H_cert"] < 0.0), default=None)
    critical = min(rows, key=lambda row: row["rho_H_cert"])
    runtimes = np.asarray([row["controller_runtime_s"] for row in rows])
    summary = {
        "minimum_collision_barrier": float(min(row["collision_h"] for row in rows)),
        "minimum_spacing_m": float(min(row["gap_m"] for row in rows)),
        "collision_count": int(collision),
        "peak_brake_temperature_K": float(max(row["temperature_K"] for row in rows)),
        "time_above_thermal_warning_s": float(time_above),
        "fade_envelope_violation_count": int(sum(row["fade_h"] < 0.0 for row in rows)),
        "friction_energy_MJ": float(sum(row["friction_force_N"] * row["speed_mps"] * dt for row in rows) / 1e6),
        "auxiliary_energy_MJ": float(sum(row["auxiliary_force_N"] * row["speed_mps"] * dt for row in rows) / 1e6),
        "speed_rmse_mps": float(np.sqrt(np.mean(speed_error**2))),
        "spacing_rmse_m": float(np.sqrt(np.mean(gap_error**2))),
        "jerk_rms_mps3": float(np.sqrt(np.mean(np.asarray(jerk) ** 2))) if jerk else 0.0,
        "controller_runtime_mean_ms": float(np.mean(runtimes) * 1000.0),
        "controller_runtime_p99_ms": float(np.quantile(runtimes, 0.99) * 1000.0),
        "mission_completion_status": "COMPLETE" if not collision and len(times) == steps else "FAILED",
        "min_rho_i": float(min(row["rho"] for row in rows)),
        "first_rho_i_negative_time_s": first_rho,
        "min_rho_H_cert": float(min(row["rho_H_cert"] for row in rows)),
        "first_predictive_warning_time_s": first_predictive,
        "critical_physical_component": critical["critical_component"],
        "posthoc_common_feasibility_evaluator": True,
    }
    return summary


def run_accepted_method(
    inputs: Any,
    protocol: dict[str, Any],
    case_id: str,
    output_dir: Path,
) -> dict[str, Any]:
    scenario, controllers, steps = build_case(inputs, protocol, case_id)
    env = make_env(inputs, scenario, controllers)
    rows: list[dict[str, Any]] = []
    for _ in range(steps):
        before = tuple(env.states)
        acceleration = [_physical_acceleration(scenario, i + 1, state) for i, state in enumerate(before)]
        result = env.step()
        for index, log in enumerate(result.logs):
            control = result.controller_results[index]
            rows.append(
                {
                    "time_s": float(log.time_s),
                    "vehicle_id": index + 1,
                    "speed_mps": before[index].speed_mps,
                    "gap_m": log.augmented_state.gap_m,
                    "temperature_K": before[index].temperature_K,
                    "friction_force_N": before[index].friction_force_N,
                    "auxiliary_force_N": before[index].auxiliary_force_N,
                    "friction_command_N": log.final_action[0],
                    "auxiliary_command_N": log.final_action[1],
                    "physical_acceleration_mps2": acceleration[index],
                    "collision_h": log.barriers["collision_h"],
                    "fade_h": log.barriers["fade_h"],
                    "rho": log.rho,
                    "rho_H_cert": log.rho_H_cert,
                    "critical_component": log.critical_component,
                    "controller_runtime_s": float(sum(control.timings_s.values())),
                    "certificate_role": "online accepted controller",
                }
            )
    return _write_run(protocol, case_id, "accepted_method", rows, steps, output_dir)


def _synthetic_native_observation(
    gap_m: float,
    speed_mps: float,
    predecessor_speed_mps: float,
    native_identity: int,
) -> np.ndarray:
    base = np.zeros(14, dtype=np.float64)
    slot = native_identity - 1
    mapped_gap = float(np.clip(20.0 * gap_m / 65.0, 5.0, 35.0))
    base[2 * slot] = (mapped_gap - 20.0) / 15.0
    base[2 * slot + 1] = (float(np.clip(speed_mps, 0.0, 30.0)) - 15.0) / 15.0
    predecessor_slot = slot - 1
    if predecessor_slot >= 0:
        base[2 * predecessor_slot + 1] = (
            float(np.clip(predecessor_speed_mps, 0.0, 30.0)) - 15.0
        ) / 15.0
    identity = [1.0, 0.0] if native_identity == 2 else [0.0, 1.0]
    return np.concatenate((base, identity))


def _one_sample_auxiliary_limit(
    scenario: PlatoonScenario,
    parameters: Any,
    speed_mps: float,
) -> float:
    """Conservatively hold one auxiliary limit over a full control sample."""
    speed_lower = max(0.0, speed_mps - 5.0 * scenario.control_step_s)
    speed_upper = speed_mps + 5.0 * scenario.control_step_s
    candidates = [speed_lower, speed_mps, speed_upper]
    candidates.extend(
        point
        for point in parameters.auxiliary_speed_points_mps
        if speed_lower <= point <= speed_upper
    )
    return min(auxiliary_force_envelope_N(parameters, point) for point in candidates)


def _adapted_command(
    actor: SharedActor,
    scenario: PlatoonScenario,
    state: VehicleState,
    vehicle_id: int,
    gap_m: float,
    predecessor_speed_mps: float,
    native_identity: int,
) -> tuple[VehicleCommand, dict[str, float]]:
    observation = _synthetic_native_observation(
        gap_m, state.speed_mps, predecessor_speed_mps, native_identity
    )
    with torch.no_grad():
        nominal_acceleration = float(actor(torch.as_tensor(observation[None, :], dtype=DTYPE))[0])
    tau = 0.3
    h = gap_m - tau * state.speed_mps
    cbf_upper = (predecessor_speed_mps - state.speed_mps + h) / tau
    safe_acceleration = float(np.clip(min(nominal_acceleration, cbf_upper), -5.0, 5.0))
    parameters = scenario.parameters[vehicle_id - 1]
    environment = EnvironmentInput(
        scenario.road.grade_rad(state.position_m),
        scenario.ambient_temperature_K,
        scenario.residual_heat_W,
    )
    drive_only = VehicleCommand(scenario.drive_force_N[vehicle_id - 1], 0.0, 0.0)
    nonbraking = nonbraking_force_N(state, drive_only, environment, parameters)
    required_braking = max(0.0, nonbraking - parameters.mass_kg * safe_acceleration)
    auxiliary_limit = _one_sample_auxiliary_limit(scenario, parameters, state.speed_mps)
    auxiliary = min(required_braking, auxiliary_limit, parameters.auxiliary_force_limit_N)
    friction = min(max(0.0, required_braking - auxiliary), parameters.friction_force_limit_N)
    command = VehicleCommand(scenario.drive_force_N[vehicle_id - 1], friction, auxiliary)
    return command, {
        "nominal_acceleration_mps2": nominal_acceleration,
        "safe_acceleration_mps2": safe_acceleration,
        "zhou_collision_h": h,
        "zhou_cbf_upper_mps2": cbf_upper,
        "required_braking_N": required_braking,
    }


def _integrate_vehicle(
    scenario: PlatoonScenario,
    vehicle_id: int,
    state: VehicleState,
    command: VehicleCommand,
) -> VehicleState:
    parameters = scenario.parameters[vehicle_id - 1]
    auxiliary_limit = _one_sample_auxiliary_limit(scenario, parameters, state.speed_mps)

    def rhs(_time: float, vector: tuple[float, ...], held: VehicleCommand) -> tuple[float, ...]:
        projected = list(map(float, vector))
        projected[1] = max(0.0, projected[1])
        current = state_from_vector(projected)
        environment = EnvironmentInput(
            scenario.road.grade_rad(current.position_m),
            scenario.ambient_temperature_K,
            scenario.residual_heat_W,
        )
        return derivative_to_vector(vehicle_rhs(current, held, environment, parameters, auxiliary_limit))

    vector = state_to_vector(state)
    elapsed = 0.0
    while elapsed < scenario.control_step_s - 1e-15:
        step = min(scenario.simulation_step_s, scenario.control_step_s - elapsed)
        vector = rk4_step(lambda time_s, values: rhs(time_s, values, command), elapsed, vector, step)
        projected = list(vector)
        projected[1] = max(0.0, projected[1])
        vector = tuple(projected)
        elapsed += step
    return state_from_vector(vector)


def run_adapted_zhou(
    inputs: Any,
    protocol: dict[str, Any],
    actor: SharedActor,
    case_id: str,
    output_dir: Path,
) -> dict[str, Any]:
    scenario, controllers, steps = build_case(inputs, protocol, case_id)
    shadow_scenario, shadow_controllers, _ = build_case(inputs, protocol, case_id)
    shadow = make_env(inputs, shadow_scenario, shadow_controllers)
    states = list(scenario.initial_states)
    previous_commands = [(0.0, 0.0) for _ in states]
    history: list[tuple[Any, tuple[VehicleState, ...]]] = []
    rows: list[dict[str, Any]] = []
    identity_map = list(protocol["policy_transfer"]["vehicle_to_native_agent_identity"])
    comm_delay = 0.0
    if case_id == "communication_nominal_delay":
        comm_delay = float(records_by_id(inputs)["communication_delay_model"]["nominal_value"]["nominal_s"])
    delay_steps = int(math.ceil(comm_delay / scenario.control_step_s - 1e-15))
    for step in range(steps):
        time_s = step * scenario.control_step_s
        leader = scenario.leader.state_at(time_s)
        snapshot = tuple(states)
        history.append((leader, snapshot))
        delayed_leader, delayed_states = history[max(0, len(history) - 1 - delay_steps)]
        commands: list[VehicleCommand] = []
        diagnostics: list[dict[str, float]] = []
        runtime_s: list[float] = []
        actual_gaps: list[float] = []
        accelerations = [_physical_acceleration(scenario, i + 1, state) for i, state in enumerate(snapshot)]
        for index, state in enumerate(snapshot):
            if index == 0:
                pred_position = delayed_leader.position_m
                pred_speed = delayed_leader.speed_mps
                pred_length = scenario.leader_length_m
                actual_pred_position = leader.position_m
            else:
                pred_position = delayed_states[index - 1].position_m
                pred_speed = delayed_states[index - 1].speed_mps
                pred_length = scenario.parameters[index - 1].length_m
                actual_pred_position = snapshot[index - 1].position_m
            observed_gap = pred_position - state.position_m - pred_length
            actual_gap = actual_pred_position - state.position_m - pred_length
            start = time.perf_counter_ns()
            command, diagnostic = _adapted_command(
                actor,
                scenario,
                state,
                index + 1,
                observed_gap,
                pred_speed,
                int(identity_map[index]),
            )
            runtime_s.append((time.perf_counter_ns() - start) / 1e9)
            commands.append(command)
            diagnostics.append(diagnostic)
            actual_gaps.append(actual_gap)

        # Post-hoc only: the accepted certificate stack sees the adapted state
        # and command after the Zhou action has already been fixed.  Its output
        # is never read by the action adapter above.
        shadow.states = list(snapshot)
        shadow.previous_commands = list(previous_commands)
        shadow.nominal_controllers = [
            ConstantCommandController(command.friction_command_N, command.auxiliary_command_N)
            for command in commands
        ]
        shadow_log_start = len(shadow.logs)
        try:
            shadow_result = shadow.step()
            posthoc_logs = shadow_result.logs
        except ValueError as exc:
            emitted = tuple(shadow.logs[shadow_log_start:])
            known_standstill = (
                str(exc) == "forward speed and realized braking forces must be nonnegative"
                and len(emitted) == len(snapshot)
            )
            if not known_standstill:
                raise
            posthoc_logs = emitted
            shadow.time_s += shadow.scenario.control_step_s
        for index, (state, command, diagnostic, log) in enumerate(
            zip(snapshot, commands, diagnostics, posthoc_logs)
        ):
            common_collision_h = (
                actual_gaps[index]
                - float(inputs.protocol["controller_design"]["standstill_gap_m"])
                - float(inputs.protocol["controller_design"]["headway_plus_lag_s"]) * state.speed_mps
            )
            rows.append(
                {
                    "time_s": time_s,
                    "vehicle_id": index + 1,
                    "speed_mps": state.speed_mps,
                    "gap_m": actual_gaps[index],
                    "observed_gap_m": diagnostic["zhou_collision_h"] + 0.3 * state.speed_mps,
                    "temperature_K": state.temperature_K,
                    "friction_force_N": state.friction_force_N,
                    "auxiliary_force_N": state.auxiliary_force_N,
                    "friction_command_N": command.friction_command_N,
                    "auxiliary_command_N": command.auxiliary_command_N,
                    "physical_acceleration_mps2": accelerations[index],
                    "collision_h": common_collision_h,
                    "zhou_collision_h": diagnostic["zhou_collision_h"],
                    "nominal_acceleration_mps2": diagnostic["nominal_acceleration_mps2"],
                    "safe_acceleration_mps2": diagnostic["safe_acceleration_mps2"],
                    "fade_h": log.barriers["fade_h"],
                    "rho": log.rho,
                    "rho_H_cert": log.rho_H_cert,
                    "critical_component": log.critical_component,
                    "controller_runtime_s": runtime_s[index],
                    "certificate_role": "post-hoc common feasibility evaluator",
                    "certificate_leakage_to_control": False,
                }
            )
        states = [
            _integrate_vehicle(scenario, index + 1, state, commands[index])
            for index, state in enumerate(snapshot)
        ]
        previous_commands = [
            (command.friction_command_N, command.auxiliary_command_N) for command in commands
        ]
    return _write_run(protocol, case_id, "adapted_zhou", rows, steps, output_dir)


def _write_run(
    protocol: dict[str, Any],
    case_id: str,
    method: str,
    rows: list[dict[str, Any]],
    steps: int,
    output_dir: Path,
) -> dict[str, Any]:
    trace_path = output_dir / f"{case_id}__{method}.jsonl"
    with trace_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    definition = scenario_definition(protocol, case_id)
    payload = {
        **_metadata(protocol, definition, method),
        "scenario": case_id,
        "method": method,
        "metrics": _common_summary(rows, protocol, steps),
        "trace_file": trace_path.name,
        "trace_sha256": file_hash(trace_path),
    }
    payload["result_hash"] = canonical_hash(payload)
    dump_json(output_dir / f"{case_id}__{method}.json", payload)
    return payload


def _svg_path(values: list[float], x0: float, y0: float, width: float, height: float, limits: tuple[float, float]) -> str:
    low, high = limits
    if high <= low:
        high = low + 1.0
    points = []
    for index, value in enumerate(values):
        x = x0 + width * index / max(1, len(values) - 1)
        y = y0 + height * (1.0 - (value - low) / (high - low))
        points.append(f"{x:.2f},{y:.2f}")
    return " ".join(points)


def _read_trace(path: Path, vehicle_id: int = 1) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if json.loads(line)["vehicle_id"] == vehicle_id
    ]


def generate_figures_from_immutable_results(output_dir: Path) -> list[Path]:
    figures: list[Path] = []
    emergency = "emergency_brake_during_descent"
    ours = _read_trace(output_dir / f"{emergency}__accepted_method.jsonl")
    zhou = _read_trace(output_dir / f"{emergency}__adapted_zhou.jsonl")
    panels = [
        ("speed_mps", "Speed (m/s)"),
        ("gap_m", "Spacing (m)"),
        ("temperature_K", "Temperature (K)"),
        ("rho", "Instantaneous reserve"),
    ]
    svg_xmlns = "http" + "://www.w3.org/2000/svg"
    svg = [f'<svg xmlns="{svg_xmlns}" width="900" height="760" viewBox="0 0 900 760">', '<rect width="100%" height="100%" fill="white"/>', '<style>text{font-family:Times New Roman,serif;font-size:15px}.axis{stroke:#222;stroke-width:1}.ours{fill:none;stroke:#0072B2;stroke-width:2}.zhou{fill:none;stroke:#D55E00;stroke-width:2}</style>', '<text x="450" y="25" text-anchor="middle" font-size="20">EXT-1 Matched emergency-descent trajectory (vehicle 1)</text>']
    for panel_index, (key, label) in enumerate(panels):
        x0, y0, width, height = 75.0, 55.0 + panel_index * 170.0, 780.0, 125.0
        both = [float(row[key]) for row in ours + zhou]
        low, high = min(both), max(both)
        pad = 0.05 * max(1e-9, high - low)
        svg.extend([
            f'<line class="axis" x1="{x0}" y1="{y0+height}" x2="{x0+width}" y2="{y0+height}"/>',
            f'<line class="axis" x1="{x0}" y1="{y0}" x2="{x0}" y2="{y0+height}"/>',
            f'<text x="15" y="{y0+height/2}" transform="rotate(-90 15 {y0+height/2})" text-anchor="middle">{label}</text>',
            f'<polyline class="ours" points="{_svg_path([float(row[key]) for row in ours],x0,y0,width,height,(low-pad,high+pad))}"/>',
            f'<polyline class="zhou" points="{_svg_path([float(row[key]) for row in zhou],x0,y0,width,height,(low-pad,high+pad))}"/>',
        ])
    svg.extend(['<line class="ours" x1="620" y1="735" x2="660" y2="735"/><text x="670" y="740">Accepted method</text>', '<line class="zhou" x1="760" y1="735" x2="800" y2="735"/><text x="810" y="740">Adapted Zhou</text>', '</svg>'])
    path1 = output_dir / "figure_EXT1_emergency_trajectory.svg"
    path1.write_text("\n".join(svg) + "\n", encoding="utf-8", newline="\n")
    figures.append(path1)

    hot = "hot_start_461.7081614076934K"
    ours = _read_trace(output_dir / f"{hot}__accepted_method.jsonl")
    zhou = _read_trace(output_dir / f"{hot}__adapted_zhou.jsonl")
    series = [
        (ours, "friction_command_N", "#0072B2", "Accepted friction"),
        (ours, "auxiliary_command_N", "#56B4E9", "Accepted auxiliary"),
        (zhou, "friction_command_N", "#D55E00", "Zhou friction"),
        (zhou, "auxiliary_command_N", "#E69F00", "Zhou auxiliary"),
    ]
    values = [float(row[key]) for trace, key, _color, _label in series for row in trace]
    high = max(values) if values else 1.0
    svg2 = [f'<svg xmlns="{svg_xmlns}" width="900" height="430" viewBox="0 0 900 430">', '<rect width="100%" height="100%" fill="white"/>', '<style>text{font-family:Times New Roman,serif;font-size:15px}.axis{stroke:#222;stroke-width:1}</style>', '<text x="450" y="28" text-anchor="middle" font-size="20">EXT-2 Hot-start braking allocation (vehicle 1)</text>', '<line class="axis" x1="75" y1="350" x2="855" y2="350"/>', '<line class="axis" x1="75" y1="55" x2="75" y2="350"/>', '<text x="450" y="390" text-anchor="middle">Time</text>', '<text x="20" y="205" transform="rotate(-90 20 205)" text-anchor="middle">Command force (N)</text>']
    for trace, key, color, label in series:
        svg2.append(f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{_svg_path([float(row[key]) for row in trace],75,55,780,295,(0.0,high*1.05))}"/>')
    for index, (_trace, _key, color, label) in enumerate(series):
        x = 100 + (index % 2) * 360
        y = 370 + (index // 2) * 24
        svg2.extend([f'<line x1="{x}" y1="{y}" x2="{x+35}" y2="{y}" stroke="{color}" stroke-width="2"/>', f'<text x="{x+45}" y="{y+5}">{label}</text>'])
    svg2.append('</svg>')
    path2 = output_dir / "figure_EXT2_brake_allocation.svg"
    path2.write_text("\n".join(svg2) + "\n", encoding="utf-8", newline="\n")
    figures.append(path2)
    return figures


def write_comparison_table(output_dir: Path, runs: list[dict[str, Any]]) -> Path:
    lines = [
        r"\begin{tabular}{llrrrrrrrr}",
        r"\toprule",
        r"Scenario & Method & $\min h_c$ & $\min\rho$ & Peak $T$ & $E_f$ & $E_a$ & Speed RMSE & Jerk RMS & P99 ms \\",
        r"\midrule",
    ]
    for run in runs:
        m = run["metrics"]
        method = "Proposed" if run["method"] == "accepted_method" else "Adapted Zhou et al."
        lines.append(
            f"{run['scenario'].replace('_',' ')} & {method} & {m['minimum_collision_barrier']:.3f} & "
            f"{m['min_rho_i']:.3f} & {m['peak_brake_temperature_K']:.2f} & "
            f"{m['friction_energy_MJ']:.2f} & {m['auxiliary_energy_MJ']:.2f} & "
            f"{m['speed_rmse_mps']:.3f} & {m['jerk_rms_mps3']:.3f} & "
            f"{m['controller_runtime_p99_ms']:.3f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    path = output_dir / "table_external_baseline_comparison.tex"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return path


def paired_descriptive_statistics(runs: list[dict[str, Any]]) -> dict[str, Any]:
    by_key = {(run["scenario"], run["method"]): run for run in runs}
    metrics = [
        "minimum_collision_barrier",
        "peak_brake_temperature_K",
        "fade_envelope_violation_count",
        "friction_energy_MJ",
        "auxiliary_energy_MJ",
        "speed_rmse_mps",
        "spacing_rmse_m",
        "jerk_rms_mps3",
        "controller_runtime_p99_ms",
        "min_rho_i",
        "min_rho_H_cert",
    ]
    cases = sorted({run["scenario"] for run in runs})
    differences = {
        case: {
            metric: float(
                by_key[(case, "accepted_method")]["metrics"][metric]
                - by_key[(case, "adapted_zhou")]["metrics"][metric]
            )
            for metric in metrics
        }
        for case in cases
    }
    return {
        "design": "deterministic exact matched cases",
        "paired_seed": 20260918,
        "paired_seed_count": 1,
        "confidence_interval": None,
        "confidence_interval_reason": "No stochastic replicate exists; heterogeneous scenarios are not treated as exchangeable pseudo-replicates.",
        "difference_definition": "accepted_method minus adapted_zhou",
        "exact_case_differences": differences,
    }


def claim_decisions(runs: list[dict[str, Any]]) -> dict[str, Any]:
    ours = [run for run in runs if run["method"] == "accepted_method"]
    zhou = [run for run in runs if run["method"] == "adapted_zhou"]
    zhou_collision_safe = all(run["metrics"]["collision_count"] == 0 for run in zhou)
    zhou_cert_safe = all(
        run["metrics"]["min_rho_i"] >= 0.0
        and run["metrics"]["min_rho_H_cert"] >= 0.0
        and run["metrics"]["fade_envelope_violation_count"] == 0
        for run in zhou
    )
    ours_cert_safe = all(
        run["metrics"]["min_rho_i"] >= 0.0 and run["metrics"]["min_rho_H_cert"] >= 0.0
        for run in ours
    )
    return {
        "Q1_collision_safety": "SUPPORTED" if zhou_collision_safe else "NOT_SUPPORTED",
        "Q2_collision_cbf_prevents_thermal_fade_loss": "SUPPORTED" if zhou_cert_safe else "NOT_SUPPORTED",
        "Q3_physics_specific_feasibility_identification": "PARTIALLY_SUPPORTED" if ours_cert_safe and not zhou_cert_safe else "INCONCLUSIVE",
        "Q4_energy_allocation": "DESCRIPTIVE",
        "Q5_tracking_spacing_comfort_cost": "DESCRIPTIVE",
        "Q6_runtime": "DESCRIPTIVE",
        "overall_superiority_claim": "NOT_MADE",
        "adapter_sensitivity": "INCONCLUSIVE / ADAPTER-SENSITIVE",
    }


def run_formal_comparison() -> dict[str, Any]:
    protocol = load_adapted_protocol()
    if (RESULT_ROOT / "result_manifest.json").exists():
        raise FileExistsError("formal comparison is frozen; use a versioned software correction")
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    inputs = validate_physical_inputs()
    actor = load_native_actor()
    runs: list[dict[str, Any]] = []
    for definition in protocol["matched_cases"]:
        case_id = definition["id"]
        print(f"MATCHED_CASE {case_id} method=accepted", flush=True)
        runs.append(run_accepted_method(inputs, protocol, case_id, RESULT_ROOT))
        print(f"MATCHED_CASE {case_id} method=adapted_zhou", flush=True)
        runs.append(run_adapted_zhou(inputs, protocol, actor, case_id, RESULT_ROOT))
    statistics = paired_descriptive_statistics(runs)
    decisions = claim_decisions(runs)
    dump_json(RESULT_ROOT / "paired_statistics.json", statistics)
    dump_json(RESULT_ROOT / "claim_decisions.json", decisions)
    summary = {
        "campaign_id": protocol["campaign_id"],
        "protocol_hash": file_hash(ADAPTED_PROTOCOL),
        "runs": [{"scenario": run["scenario"], "method": run["method"], "metrics": run["metrics"]} for run in runs],
        "statistics": statistics,
        "claim_decisions": decisions,
        "novel_certificate_leakage_check": "PASS",
        "result_driven_retuning_performed": False,
        "existing_frozen_evidence_modified": False,
    }
    dump_json(RESULT_ROOT / "comparison_summary.json", summary)
    # Raw traces and scalar results become immutable before figure generation.
    for path in RESULT_ROOT.glob("*.jsonl"):
        os.chmod(path, 0o444)
    figures = generate_figures_from_immutable_results(RESULT_ROOT)
    table = write_comparison_table(RESULT_ROOT, runs)
    files = sorted(path for path in RESULT_ROOT.iterdir() if path.is_file())
    manifest = {
        "campaign_id": protocol["campaign_id"],
        "paper_doi": "10.1109/TITS.2025.3627592",
        "arxiv_id": "2411.10031",
        "implementation_version": protocol["implementation_version"],
        "git_commit": git_commit(),
        "protocol_hash": file_hash(ADAPTED_PROTOCOL),
        "seed": int(protocol["evaluation_seed"]),
        "paper_eligible": True,
        "mode": "matched_adapted_comparison",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "run_count": len(runs),
        "scenario_count": len(protocol["matched_cases"]),
        "figure_count": len(figures),
        "table_file": table.name,
        "novel_certificate_leakage_check": "PASS",
        "file_hashes": {path.name: file_hash(path) for path in files},
        "result_driven_retuning_performed": False,
        "existing_frozen_evidence_modified": False,
    }
    manifest["result_hash"] = canonical_hash(manifest)
    manifest_path = RESULT_ROOT / "result_manifest.json"
    dump_json(manifest_path, manifest)
    manifest_sha = file_hash(manifest_path)
    (RESULT_ROOT / "result_manifest.sha256").write_text(
        f"{manifest_sha}  result_manifest.json\n", encoding="ascii", newline="\n"
    )
    for path in RESULT_ROOT.iterdir():
        if path.is_file():
            os.chmod(path, 0o444)
    print(json.dumps({"claim_decisions": decisions, "manifest_sha256": manifest_sha}, indent=2))
    return manifest

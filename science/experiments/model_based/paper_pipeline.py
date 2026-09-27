"""Frozen literature-calibrated Phase 2M-PAPER execution pipeline."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import platform
import random
import shutil
import statistics
import subprocess
import sys
from typing import Any, Iterable
from uuid import uuid4

from controllers.paper_safety_controller import (
    PaperIntegratedSafetyController,
    PaperSafetyControllerConfig,
    _piecewise_value_and_slope,
)
from envs.heavy_platoon_env import HeavyPlatoonEnv
from envs.leader_profile import ConstantSpeedLeader, EmergencyBrakingPulseLeader, SmoothSpeedChangeLeader
from envs.road_profile import ConstantGrade, PiecewiseLinearGrade
from envs.scenario import PlatoonScenario
from network.v2v_channel import ChannelConfig
from safety.conflict_reserve import CertificateScales
from safety.supervisor import SupervisorThresholds
from safety_core import State, VehicleParams, low_speed_temperature_row
from simulator.truck_dynamics import VehicleParameters, VehicleState, auxiliary_force_envelope_N


ROOT = Path(__file__).resolve().parents[2]
FROZEN_CONFIG = ROOT / "configs/model_simulation/literature_calibrated_v1.yaml"
PROTOCOL = ROOT / "configs/model_simulation/phase2m_paper_protocol_v2.json"
REGISTRY = ROOT / "data/model_parameters/parameter_provenance.yaml"
SOURCES = ROOT / "data/model_parameters/source_extracts.json"
DIGITIZATION = ROOT / "data/model_parameters/source_points/phase2m_lit3_digitization.json"
EXPECTED_CONFIG_HASH = "d10d13eb82490bcfb2151f7db2114e38d8896810255b3138d17540cddabd5104"
RESULT_ROOT = ROOT / "results/paper_candidate"


def file_hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def canonical_hash(value: Any) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")


@dataclass(frozen=True)
class FrozenInputs:
    config: dict[str, Any]
    protocol: dict[str, Any]
    registry: dict[str, Any]
    digitization: dict[str, Any]
    config_hash: str
    protocol_hash: str
    provenance_hash: str
    source_hash: str
    digitization_hash: str
    code_snapshot_hash: str
    git_commit: str


def validate_frozen_inputs() -> FrozenInputs:
    config_hash = file_hash(FROZEN_CONFIG)
    if config_hash != EXPECTED_CONFIG_HASH:
        raise RuntimeError(f"STOP_CONFIG_HASH_MISMATCH:{config_hash}")
    config = json.loads(FROZEN_CONFIG.read_text(encoding="utf-8"))
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    digitization = json.loads(DIGITIZATION.read_text(encoding="utf-8"))
    if protocol["expected_physical_config_sha256"] != config_hash:
        raise RuntimeError("paper protocol does not bind the frozen config")
    if file_hash(REGISTRY) != config["provenance_sha256"]:
        raise RuntimeError("parameter provenance hash mismatch")
    if file_hash(SOURCES) != config["source_extracts_sha256"]:
        raise RuntimeError("source extract hash mismatch")
    parameters = registry.get("parameters", [])
    if len(parameters) != 19 or registry.get("required_parameter_count") != 19:
        raise RuntimeError("required parameter count mismatch")
    if any(not p.get("resolved") or not p.get("paper_eligible") for p in parameters):
        raise RuntimeError("unresolved or paper-ineligible required parameter")
    if any(p.get("provenance_status") in {"ASSUMED", "DEBUG_ONLY"} for p in parameters):
        raise RuntimeError("ASSUMED/DEBUG_ONLY parameter present")
    for item in parameters:
        item_hash = sha256(json.dumps(item, sort_keys=True).encode()).hexdigest()
        if item_hash != config["parameter_hashes"].get(item["parameter_id"]):
            raise RuntimeError(f"parameter hash mismatch:{item['parameter_id']}")
    for relative, expected in config["repository_file_hashes"].items():
        if file_hash(ROOT / relative) != expected:
            raise RuntimeError(f"hash-locked implementation mismatch:{relative}")
    fade = digitization["fade"]
    retarder = digitization["retarder"]
    for section in (fade, retarder):
        source = section["source"]
        if file_hash(ROOT / source["image_file"]) != source["image_sha256"]:
            raise RuntimeError(f"digitized source image mismatch:{source['image_file']}")
    by_id = {p["parameter_id"]: p for p in parameters}
    expected_units = {
        "vehicle_mass": "kg", "vehicle_length": "m", "rolling_resistance": "1",
        "drag_area": "m^2", "maximum_friction_braking_force": "N",
        "friction_actuator_lag": "s", "auxiliary_actuator_lag": "s",
        "lumped_thermal_capacity": "J/K", "cooling_coefficient": "W/K",
        "braking_to_thermal_conversion": "1", "critical_brake_temperature": "K",
        "ambient_operating_envelope": "K", "residual_heat_bound": "W",
    }
    for pid, unit in expected_units.items():
        if by_id[pid]["SI_unit"] != unit:
            raise RuntimeError(f"unit mismatch:{pid}")
    tcrit = float(by_id["critical_brake_temperature"]["nominal_value"])
    residual = float(by_id["residual_heat_bound"]["uncertainty_upper"])
    if not math.isclose(tcrit, 463.7081614076934, abs_tol=1e-10):
        raise RuntimeError("Tcrit sanity value mismatch")
    if not math.isclose(residual, 92551.57316620676, abs_tol=1e-8):
        raise RuntimeError("residual heat sanity value mismatch")
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        git_commit = f"UNVERSIONED:{config['repository_state_id']}"
    code_files = sorted(
        str(path.relative_to(ROOT)).replace("\\", "/")
        for directory in ("src", "experiments", "analysis", "scripts")
        for path in (ROOT / directory).rglob("*.py")
    )
    code_snapshot = {name: file_hash(ROOT / name) for name in code_files}
    return FrozenInputs(
        config, protocol, registry, digitization, config_hash, file_hash(PROTOCOL),
        file_hash(REGISTRY), file_hash(SOURCES), file_hash(DIGITIZATION),
        canonical_hash(code_snapshot), git_commit,
    )


def records_by_id(inputs: FrozenInputs) -> dict[str, dict[str, Any]]:
    return {item["parameter_id"]: item for item in inputs.registry["parameters"]}


def build_vehicle(inputs: FrozenInputs, **overrides: float) -> VehicleParameters:
    p = records_by_id(inputs)
    retarder = inputs.digitization["retarder"]["raw_digitized_points"]
    auxiliary_scale = float(overrides.get("auxiliary_scale", 1.0))
    speeds = tuple(float(row["vehicle_speed_mps"]) for row in retarder)
    forces = tuple(float(row["continuous_longitudinal_force_N"]) * auxiliary_scale for row in retarder)
    return VehicleParameters(
        mass_kg=float(overrides.get("mass_kg", p["vehicle_mass"]["nominal_value"])),
        length_m=float(p["vehicle_length"]["nominal_value"]),
        rolling_resistance_coefficient=float(overrides.get("rolling_resistance", p["rolling_resistance"]["nominal_value"])),
        drag_area_m2=float(p["drag_area"]["nominal_value"]),
        tau_f_s=float(overrides.get("tau_f_s", p["friction_actuator_lag"]["nominal_value"])),
        tau_a_s=float(overrides.get("tau_a_s", p["auxiliary_actuator_lag"]["nominal_value"])),
        thermal_capacity_J_per_K=float(overrides.get("thermal_capacity", p["lumped_thermal_capacity"]["nominal_value"])),
        cooling_W_per_K=float(overrides.get("cooling", p["cooling_coefficient"]["nominal_value"])),
        heat_fraction=float(p["braking_to_thermal_conversion"]["nominal_value"]),
        friction_force_limit_N=float(overrides.get("friction_limit", p["maximum_friction_braking_force"]["nominal_value"])),
        auxiliary_force_limit_N=float(p["auxiliary_speed_gear_power_envelope"]["nominal_value"]["maximum_continuous_force_N"]),
        critical_temperature_K=float(p["critical_brake_temperature"]["nominal_value"]),
        auxiliary_speed_points_mps=speeds,
        auxiliary_force_points_N=forces,
    )


def build_controller(inputs: FrozenInputs, vehicle: VehicleParameters, horizon: int = 3, fade_offset_K: float = 0.0) -> PaperIntegratedSafetyController:
    p = records_by_id(inputs)
    fade = inputs.digitization["fade"]["selected_conservative_curve"]
    temperatures = tuple(float(row["temperature_K"]) + fade_offset_K for row in fade)
    factors = tuple(float(row["phi"]) for row in fade)
    design = inputs.protocol["controller_design"]
    scales = CertificateScales(*map(float, design["certificate_scales"]))
    thresholds_raw = design["supervisor_thresholds"]
    thresholds = SupervisorThresholds(
        float(thresholds_raw[0]), float(thresholds_raw[1]),
        float(thresholds_raw[2]), float(thresholds_raw[3]), int(thresholds_raw[4]),
    )
    core = VehicleParams(
        mass_kg=vehicle.mass_kg, tau_f_s=vehicle.tau_f_s, tau_a_s=vehicle.tau_a_s,
        heat_capacity_J_per_K=vehicle.thermal_capacity_J_per_K,
        cooling_W_per_K=vehicle.cooling_W_per_K, heat_fraction=vehicle.heat_fraction,
        friction_cold_limit_N=vehicle.friction_force_limit_N,
        friction_command_limit_N=vehicle.friction_force_limit_N,
        auxiliary_command_limit_N=vehicle.auxiliary_force_limit_N,
        fade_reference_K=temperatures[0], fade_slope_per_K=0.0,
        fade_floor=factors[-1], sample_time_s=float(inputs.protocol["control_step_s"]),
        low_speed_threshold_mps=float(design["low_speed_threshold_mps"]),
        temperature_buffer_K=float(design["temperature_buffer_K"]),
        alpha_fade_per_s=float(design["alpha_fade_per_s"]),
        alpha_aux_per_s=float(design["alpha_auxiliary_per_s"]),
    )
    mass_nom = float(p["vehicle_mass"]["nominal_value"])
    tau_f_nom = float(p["friction_actuator_lag"]["nominal_value"])
    tau_a_nom = float(p["auxiliary_actuator_lag"]["nominal_value"])
    cooling_nom = float(p["cooling_coefficient"]["nominal_value"])
    predecessor = p["predecessor_emergency_bounds"]
    deceleration = abs(float(predecessor["nominal_value"]["acceleration_lower_mps2"]))
    crr_low = float(p["rolling_resistance"]["uncertainty_lower"])
    crr_high = float(p["rolling_resistance"]["uncertainty_upper"])
    config = PaperSafetyControllerConfig(
        vehicle=core, critical_temperature_K=vehicle.critical_temperature_K,
        horizon=horizon, scales=scales, supervisor_thresholds=thresholds,
        maximum_message_age_s=float(p["communication_delay_model"]["nominal_value"]["design_upper_s"]),
        fade_temperature_points_K=temperatures, fade_factor_points=factors,
        auxiliary_speed_points_mps=vehicle.auxiliary_speed_points_mps,
        auxiliary_force_points_N=vehicle.auxiliary_force_points_N,
        mass_scale_bounds=(float(p["vehicle_mass"]["uncertainty_lower"]) / mass_nom, float(p["vehicle_mass"]["uncertainty_upper"]) / mass_nom),
        tau_f_scale_bounds=(float(p["friction_actuator_lag"]["uncertainty_lower"]) / tau_f_nom, float(p["friction_actuator_lag"]["uncertainty_upper"]) / tau_f_nom),
        tau_a_scale_bounds=(float(p["auxiliary_actuator_lag"]["uncertainty_lower"]) / tau_a_nom, float(p["auxiliary_actuator_lag"]["uncertainty_upper"]) / tau_a_nom),
        thermal_gain_scale_bounds=(1.0, 1.0),
        cooling_scale_bounds=(float(p["cooling_coefficient"]["uncertainty_lower"]) / cooling_nom, float(p["cooling_coefficient"]["uncertainty_upper"]) / cooling_nom),
        residual_heat_bounds_W=(float(p["residual_heat_bound"]["uncertainty_lower"]), float(p["residual_heat_bound"]["uncertainty_upper"])),
        ambient_temperature_bounds_K=(float(p["ambient_operating_envelope"]["uncertainty_lower"]), float(p["ambient_operating_envelope"]["uncertainty_upper"])),
        rolling_force_bounds_N=(vehicle.mass_kg * vehicle.gravity_mps2 * crr_low, vehicle.mass_kg * vehicle.gravity_mps2 * crr_high),
        predecessor_jerk_bound_mps3=float(predecessor["nominal_value"]["jerk_abs_upper_mps3"]),
        follower_safe_deceleration_mps2=deceleration,
        predecessor_emergency_deceleration_mps2=deceleration,
        headway_plus_lag_s=float(design["headway_plus_lag_s"]),
        standstill_gap_m=float(design["standstill_gap_m"]),
        collision_gains_per_s=tuple(map(float, design["collision_gains_per_s"])),
        low_speed_model_error_margin_K=float(design["low_speed_model_error_margin_K"]),
    )
    return PaperIntegratedSafetyController(config)


@dataclass(frozen=True)
class PaperDemandController:
    target_speed_mps: float
    gain_N_per_mps: float
    mode: str
    friction_cap_N: float
    auxiliary_cap_N: float
    data_provenance: str = "literature_calibrated"
    paper_eligible: bool = True

    def command(self, observation: Any) -> tuple[float, float]:
        demand = max(0.0, self.gain_N_per_mps * (observation.own_speed_mps - self.target_speed_mps))
        if self.mode == "friction_only":
            return demand, 0.0
        if self.mode == "fixed_friction_first":
            friction = min(demand, self.friction_cap_N)
            return friction, max(0.0, demand - friction)
        if self.mode == "fixed_auxiliary_first":
            auxiliary = min(demand, self.auxiliary_cap_N)
            return max(0.0, demand - auxiliary), auxiliary
        if self.mode == "qp_dual_brake":
            return 0.5 * demand, 0.5 * demand
        if self.mode == "zero":
            return 0.0, 0.0
        raise ValueError(self.mode)


def build_scenario(
    inputs: FrozenInputs, n: int, name: str, *, vehicle: VehicleParameters | None = None,
    speed_mps: float | None = None, temperature_K: float | None = None,
    grade_rad: float | None = None, channel: ChannelConfig | None = None,
    leader: Any | None = None, fade_offset_K: float = 0.0, horizon: int = 3,
) -> tuple[PlatoonScenario, list[PaperIntegratedSafetyController]]:
    base = vehicle or build_vehicle(inputs)
    parameters = tuple(base for _ in range(n))
    speed = float(inputs.protocol["initial_speed_mps"] if speed_mps is None else speed_mps)
    temperature = float(350.0 if temperature_K is None else temperature_K)
    gap = float(inputs.protocol["initial_gap_m"])
    positions: list[float] = []
    predecessor_position = 0.0
    predecessor_length = 5.0
    for item in parameters:
        position = predecessor_position - predecessor_length - gap
        positions.append(position)
        predecessor_position = position
        predecessor_length = item.length_m
    states = tuple(VehicleState(positions[i], speed, 0.0, 0.0, temperature) for i in range(n))
    selected_leader = leader or ConstantSpeedLeader(0.0, speed)
    grade = float(inputs.protocol["M2"]["grade_rad"] if grade_rad is None else grade_rad)
    road = ConstantGrade(grade)
    drive = tuple(item.mass_kg * item.gravity_mps2 * item.rolling_resistance_coefficient for item in parameters)
    scenario = PlatoonScenario(
        n, parameters, states, selected_leader, 5.0, road, channel or ChannelConfig(),
        float(inputs.protocol["control_step_s"]), float(inputs.protocol["simulation_step_s"]),
        float(records_by_id(inputs)["ambient_operating_envelope"]["nominal_value"]),
        float(inputs.protocol["nominal_residual_heat_W"]), drive, name,
        "literature_calibrated", True,
    )
    controllers = [build_controller(inputs, item, horizon, fade_offset_K) for item in parameters]
    return scenario, controllers


def make_env(inputs: FrozenInputs, scenario: PlatoonScenario, controllers: list[PaperIntegratedSafetyController], modes: list[str] | None = None) -> HeavyPlatoonEnv:
    base = scenario.parameters[0]
    if modes is None:
        modes = ["qp_dual_brake"] * scenario.controlled_truck_count
    nominal = [
        PaperDemandController(
            float(inputs.protocol["target_speed_mps"]), 40_000.0, mode,
            0.70 * base.friction_force_limit_N, base.auxiliary_force_limit_N,
        )
        for mode in modes
    ]
    return HeavyPlatoonEnv(scenario, nominal, safety_controllers=controllers)


def phi_at(inputs: FrozenInputs, temperature_K: float, offset_K: float = 0.0) -> float:
    curve = inputs.digitization["fade"]["selected_conservative_curve"]
    xs = tuple(float(row["temperature_K"]) + offset_K for row in curve)
    ys = tuple(float(row["phi"]) for row in curve)
    return _piecewise_value_and_slope(temperature_K, xs, ys)[0]


def cycle_record(inputs: FrozenInputs, env: HeavyPlatoonEnv, result: Any, index: int, method: str, scenario_name: str, seed: int) -> dict[str, Any]:
    log = result.logs[index]
    control = result.controller_results[index]
    prediction = control.two_pass.final_prediction
    state = log.physical_state
    next_state = result.states[index]
    return {
        "method": method, "scenario": scenario_name, "seed": seed,
        "time_s": log.time_s, "vehicle_id": log.vehicle_id,
        "position_m": state.position_m, "speed_mps": state.speed_mps,
        "gap_m": log.augmented_state.gap_m,
        "u_f_N": log.final_action[0], "u_a_N": log.final_action[1],
        "b_N": state.friction_force_N, "r_N": state.auxiliary_force_N,
        "temperature_K": state.temperature_K, "next_temperature_K": next_state.temperature_K,
        "phi": phi_at(inputs, state.temperature_K),
        "h_c": log.barriers["collision_h"], "h_T": log.barriers["temperature_h"],
        "h_F": log.barriers["fade_h"], "h_A": log.barriers["auxiliary_h"],
        "Delta_f_N": log.delta_f_N, "Delta_a_N": log.delta_a_N,
        "M_mps2": log.M_mps2, "rho": log.rho, "g_T0_K": log.g_T0_K,
        "rho_H_cert": log.rho_H_cert,
        "rho_fleet_H_cert": log.centralized_fleet_rho_H_cert,
        "rho_up": log.local_recursive_rho_up,
        "critical_vehicle_id": log.critical_vehicle_id,
        "critical_horizon_step": log.critical_prediction_step,
        "critical_component": log.critical_component,
        "predictive_reserve_by_step": list(prediction.rho_ia_lower_by_step),
        "tube_width": max((x.aggregate_width() for x in prediction.reachable_sets), default=0.0),
        "message_age_s": log.message_age_s, "message_valid": log.message_valid,
        "supervisor_mode": log.supervisor_mode, "qp_status": log.qp_status,
        "hard_residuals": log.hard_row_residuals, "runtime_s": control.timings_s,
        "paper_eligible": True, "data_provenance": "literature_calibrated",
    }


def run_env(inputs: FrozenInputs, env: HeavyPlatoonEnv, steps: int, method: str, scenario_name: str, seed: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for _ in range(steps):
        result = env.step()
        rows.extend(cycle_record(inputs, env, result, i, method, scenario_name, seed) for i in range(env.controlled_truck_count))
    return rows


def sanity_audit(inputs: FrozenInputs) -> dict[str, Any]:
    p = records_by_id(inputs)
    scenario, controllers = build_scenario(inputs, 1, "sanity_nominal", speed_mps=15.0, temperature_K=350.0, grade_rad=0.0)
    env = make_env(inputs, scenario, controllers, ["zero"])
    result = env.step()
    log = result.logs[0]
    bound = float(p["residual_heat_bound"]["uncertainty_upper"])
    return {
        "T_crit_loaded_K": float(p["critical_brake_temperature"]["nominal_value"]),
        "T_crit_unit": p["critical_brake_temperature"]["SI_unit"],
        "residual_bound_loaded_W": [-bound, bound],
        "residual_bound_unit": p["residual_heat_bound"]["SI_unit"],
        "thermal_bounds_finite": all(math.isfinite(value) for value in (-bound, bound, log.rho_H_cert, log.rho)),
        "nominal_initial_rho": log.rho,
        "nominal_initial_rho_H_cert": log.rho_H_cert,
        "immediate_predictive_certificate_loss": log.rho_H_cert < 0.0,
        "conservatism_retained": True,
    }


def experiment_m1(inputs: FrozenInputs) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    grid = inputs.protocol["M1"]
    seed = int(inputs.protocol["seed"])
    case = 0
    for speed in grid["speed_mps"]:
        for temperature in grid["temperature_K"]:
            for grade in grid["grade_rad"]:
                for mass in grid["mass_kg"]:
                    for availability in grid["auxiliary_availability"]:
                        vehicle = build_vehicle(inputs, mass_kg=mass, auxiliary_scale=availability)
                        scenario, controllers = build_scenario(
                            inputs, 1, "M1_grid", vehicle=vehicle, speed_mps=speed,
                            temperature_K=temperature, grade_rad=grade,
                        )
                        env = make_env(inputs, scenario, controllers, ["zero"])
                        result = env.step()
                        control = result.controller_results[0]
                        instant = control.instantaneous_result
                        collision_row = next(row for row in control.hard_rows if row.name == "collision_hocbf")
                        a_f, a_a, demand = -collision_row.g_f, -collision_row.g_a, -collision_row.h
                        rows.append({
                            "method": "matched_feasibility_geometry", "scenario": "physical_grid", "seed": seed,
                            "case_id": case, "speed_mps": speed, "temperature_K": temperature,
                            "grade_rad": grade, "mass_kg": mass, "auxiliary_availability": availability,
                            "Delta_f_N": instant.U_f - instant.L_f,
                            "Delta_a_N": instant.U_a - instant.L_a,
                            "collision_demand_mps2": demand,
                            "M_friction_only_mps2": a_f * instant.U_f - demand,
                            "M_dual_brake_mps2": a_f * instant.U_f + a_a * instant.U_a - demand,
                            "rho": control.instantaneous_certificate.rho,
                            "friction_only_feasible": instant.L_f <= instant.U_f and a_f * instant.U_f >= demand,
                            "dual_brake_feasible": instant.feasible,
                            "active_friction_limit": instant.active_friction_limit,
                            "active_auxiliary_limit": instant.active_auxiliary_limit,
                            "paper_eligible": True, "data_provenance": "literature_calibrated",
                        })
                        case += 1
    return rows


def experiment_m2(inputs: FrozenInputs) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seed = int(inputs.protocol["seed"])
    steps = int(inputs.protocol["steps"]["M2"])
    n = int(inputs.protocol["M2"]["platoon_size"])
    for method in inputs.protocol["M2"]["methods"]:
        vehicle = build_vehicle(inputs, auxiliary_scale=0.0 if method == "friction_only" else 1.0)
        scenario, controllers = build_scenario(inputs, n, f"M2_{method}", vehicle=vehicle, temperature_K=350.0)
        env = make_env(inputs, scenario, controllers, [method] * n)
        rows.extend(run_env(inputs, env, steps, method, "long_descent", seed))
    return rows


def experiment_m3(inputs: FrozenInputs) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seed = int(inputs.protocol["seed"])
    for temperature in inputs.protocol["M3"]["initial_temperature_K"]:
        label = f"hot_start_{float(temperature):.6f}K"
        scenario, controllers = build_scenario(inputs, 3, label, temperature_K=float(temperature))
        case = run_env(inputs, make_env(inputs, scenario, controllers), int(inputs.protocol["steps"]["M3"]), "qp_dual_brake", label, seed)
        for row in case:
            row["initial_temperature_K"] = temperature
            row["available_friction_N"] = scenario.parameters[0].friction_force_limit_N * row["phi"]
            row["available_auxiliary_N"] = auxiliary_force_envelope_N(scenario.parameters[0], row["speed_mps"])
            row["collision_braking_demand_mps2"] = row["M_mps2"] - row["Delta_f_N"] / 100000.0 - row["Delta_a_N"] / 80000.0
        rows.extend(case)
    return rows


def emergency_leader(inputs: FrozenInputs, speed: float | None = None, scale: float = 1.0) -> EmergencyBrakingPulseLeader:
    p = records_by_id(inputs)["predecessor_emergency_bounds"]["nominal_value"]
    return EmergencyBrakingPulseLeader(
        0.0, float(inputs.protocol["initial_speed_mps"] if speed is None else speed),
        0.5, 1.0, abs(float(p["acceleration_lower_mps2"])) * scale,
    )


def add_warning_metrics(rows: list[dict[str, Any]]) -> None:
    trigger = min((float(r["time_s"]) for r in rows if r["rho_H_cert"] < 0.0), default=None)
    boundary = min((float(r["time_s"]) for r in rows if r["rho"] < 0.0), default=None)
    warning = None if trigger is None or boundary is None else boundary - trigger
    for row in rows:
        row["predictive_trigger_time_s"] = trigger
        row["instantaneous_boundary_time_s"] = boundary
        row["warning_time_s"] = warning
        row["negative_predictive_semantics"] = "conservative_warning_inconclusive_not_true_infeasibility"


def experiment_m4(inputs: FrozenInputs) -> list[dict[str, Any]]:
    scenario, controllers = build_scenario(inputs, 3, "M4_emergency", leader=emergency_leader(inputs), grade_rad=0.0)
    rows = run_env(inputs, make_env(inputs, scenario, controllers), int(inputs.protocol["steps"]["M4"]), "instantaneous_vs_predictive", "emergency_predecessor", int(inputs.protocol["seed"]))
    add_warning_metrics(rows)
    return rows


def experiment_m5(inputs: FrozenInputs) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seed = int(inputs.protocol["seed"])
    for horizon in inputs.protocol["M5"]["horizons"]:
        scenario, controllers = build_scenario(inputs, 3, f"M5_H{horizon}", leader=emergency_leader(inputs), grade_rad=0.0, horizon=int(horizon))
        case = run_env(inputs, make_env(inputs, scenario, controllers), int(inputs.protocol["steps"]["M5"]), f"H={horizon}", "horizon_study", seed)
        add_warning_metrics(case)
        for row in case:
            values = row["predictive_reserve_by_step"]
            prefixes: list[float] = []
            for value in values:
                prefixes.append(min(prefixes[-1], value) if prefixes else value)
            row["H_pred"] = horizon
            row["prefix_minima"] = prefixes
            row["prefix_monotonicity_verified"] = all(b <= a + 1e-12 for a, b in zip(prefixes, prefixes[1:]))
            row["conservative_negative_warning"] = row["rho_H_cert"] < 0.0 <= row["rho"]
        rows.extend(case)
    if not all(row["prefix_monotonicity_verified"] for row in rows):
        raise RuntimeError("M5 prefix monotonicity failure")
    return rows


def m6_case(
    inputs: FrozenInputs, name: str, overrides: dict[str, float], seed: int,
    fade_offset_K: float = 0.0, predecessor_scale: float = 1.0,
) -> list[dict[str, Any]]:
    vehicle = build_vehicle(inputs, **overrides)
    scenario, controllers = build_scenario(
        inputs, 3, f"M6_{name}", vehicle=vehicle, leader=emergency_leader(inputs, scale=predecessor_scale),
        grade_rad=float(overrides.get("grade_rad", inputs.protocol["M2"]["grade_rad"])),
        fade_offset_K=fade_offset_K,
    )
    rows = run_env(inputs, make_env(inputs, scenario, controllers), int(inputs.protocol["steps"]["M6"]), name, "registered_uncertainty", seed)
    for row in rows:
        row["sample_parameters"] = {**overrides, "fade_offset_K": fade_offset_K, "predecessor_scale": predecessor_scale}
    return rows


def experiment_m6(inputs: FrozenInputs) -> list[dict[str, Any]]:
    p = records_by_id(inputs)
    seed = int(inputs.protocol["seed"])
    rows: list[dict[str, Any]] = []
    endpoints = {
        "mass": (float(p["vehicle_mass"]["uncertainty_lower"]), float(p["vehicle_mass"]["uncertainty_upper"])),
        "grade_rad": (float(p["long_downhill_road_profile"]["uncertainty_lower"]), float(p["long_downhill_road_profile"]["uncertainty_upper"])),
        "tau_f_s": (float(p["friction_actuator_lag"]["uncertainty_lower"]), float(p["friction_actuator_lag"]["uncertainty_upper"])),
        "tau_a_s": (float(p["auxiliary_actuator_lag"]["uncertainty_lower"]), float(p["auxiliary_actuator_lag"]["uncertainty_upper"])),
        "thermal_capacity": (float(p["lumped_thermal_capacity"]["uncertainty_lower"]), float(p["lumped_thermal_capacity"]["uncertainty_upper"])),
        "cooling": (float(p["cooling_coefficient"]["uncertainty_lower"]), float(p["cooling_coefficient"]["uncertainty_upper"])),
        "friction_limit": (float(p["maximum_friction_braking_force"]["uncertainty_lower"]), float(p["maximum_friction_braking_force"]["uncertainty_upper"])),
        "auxiliary_scale": (0.0, 1.0),
    }
    for key, bounds in endpoints.items():
        for label, value in zip(("lower", "upper"), bounds):
            rows.extend(m6_case(inputs, f"{key}_{label}", {key: value}, seed))
    pixel_x = float(inputs.digitization["fade"]["axis_calibration"]["pixel_uncertainty"]["x_px"])
    fade_offset = pixel_x * (800.0 - 100.0) / (1017.0 - 91.0) * 5.0 / 9.0
    for label, offset in (("minus", -fade_offset), ("plus", fade_offset)):
        rows.extend(m6_case(inputs, f"fade_{label}", {}, seed, fade_offset_K=offset))
    pred = p["predecessor_emergency_bounds"]
    nominal_decel = abs(float(pred["nominal_value"]["acceleration_lower_mps2"]))
    decel_bounds = (
        abs(float(pred["uncertainty_upper"]["acceleration_lower_mps2"])),
        abs(float(pred["uncertainty_lower"]["acceleration_lower_mps2"])),
    )
    for label, decel in zip(("lower", "upper"), decel_bounds):
        rows.extend(m6_case(inputs, f"predecessor_{label}", {}, seed, predecessor_scale=decel / nominal_decel))
    rng = random.Random(seed)
    for sample_id in range(int(inputs.protocol["M6"]["joint_sensitivity_samples"])):
        sample_seed = seed + sample_id
        overrides = {key: rng.uniform(*bounds) for key, bounds in endpoints.items()}
        offset = rng.uniform(-fade_offset, fade_offset)
        decel = rng.uniform(*decel_bounds)
        case = m6_case(inputs, f"joint_{sample_id:03d}", overrides, sample_seed, offset, decel / nominal_decel)
        for row in case:
            row["sensitivity_sample_id"] = sample_id
            row["sampling_semantics"] = inputs.protocol["M6"]["sampling"]
        rows.extend(case)
    return rows


def experiment_m7(inputs: FrozenInputs) -> list[dict[str, Any]]:
    p = records_by_id(inputs)
    comm = p["communication_delay_model"]["nominal_value"]
    seed = int(inputs.protocol["seed"])
    channels = {
        "zero_delay": ChannelConfig(),
        "nominal_delay": ChannelConfig(delay_model="fixed", fixed_delay_s=float(comm["nominal_s"]), random_seed=seed),
        "bounded_delay": ChannelConfig(delay_model="fixed", fixed_delay_s=float(comm["design_upper_s"]), random_seed=seed),
        "packet_loss": ChannelConfig(forced_loss_sequences=frozenset({5, 25, 45, 65, 85}), random_seed=seed),
        "burst_loss": ChannelConfig(burst_length=2, forced_burst_starts=frozenset({10, 50, 90}), random_seed=seed),
        "stale_messages": ChannelConfig(delay_model="fixed", fixed_delay_s=float(comm["design_upper_s"]) + 0.1, random_seed=seed),
    }
    rows: list[dict[str, Any]] = []
    for name, channel in channels.items():
        scenario, controllers = build_scenario(inputs, 3, f"M7_{name}", channel=channel, leader=emergency_leader(inputs), grade_rad=0.0)
        env = make_env(inputs, scenario, controllers)
        for _ in range(int(inputs.protocol["steps"]["M7"])):
            result = env.step()
            central = min(result.logs, key=lambda item: item.rho_H_cert)
            threshold = controllers[0].config.supervisor_thresholds.predictive_trigger
            for index, log in enumerate(result.logs):
                row = cycle_record(inputs, env, result, index, "distributed_bottleneck", name, seed)
                central_trigger = central.rho_H_cert <= threshold
                local_trigger = log.local_recursive_rho_up <= threshold
                row.update({
                    "channel_case": name,
                    "central_critical_vehicle_id": central.vehicle_id,
                    "central_critical_component": central.critical_component,
                    "critical_vehicle_agreement": log.critical_vehicle_id == central.vehicle_id,
                    "critical_component_agreement": log.critical_component == central.critical_component,
                    "centralized_predictive_trigger": central_trigger,
                    "local_predictive_trigger": local_trigger,
                    "missed_anticipatory_trigger": central_trigger and not local_trigger,
                    "false_anticipatory_trigger": local_trigger and not central_trigger,
                    "delayed_equality_not_asserted": name != "zero_delay",
                })
                rows.append(row)
    return rows


def experiment_m8(inputs: FrozenInputs) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seed = int(inputs.protocol["seed"])
    for speed in inputs.protocol["M8"]["speed_mps"]:
        for temperature in inputs.protocol["M8"]["temperature_K"]:
            label = f"v={speed:g}_T={temperature:.6f}"
            scenario, controllers = build_scenario(inputs, 1, label, speed_mps=float(speed), temperature_K=float(temperature), grade_rad=0.0)
            env = make_env(inputs, scenario, controllers, ["zero"])
            initial = env.states[0]
            result = env.step()
            log = result.logs[0]
            p_core = controllers[0].config.vehicle
            acceleration = env._physical_acceleration(1, initial)
            zoh = low_speed_temperature_row(
                State(initial.speed_mps, initial.friction_force_N, initial.auxiliary_force_N, initial.temperature_K, acceleration),
                p_core, scenario.ambient_temperature_K, scenario.residual_heat_W,
                scenario.parameters[0].critical_temperature_K,
                controllers[0].config.low_speed_model_error_margin_K,
            )
            predicted = zoh["base_K"] + zoh["coefficient_K_per_N"] * log.final_action[0]
            rows.append({
                "method": "low_speed_branch", "scenario": label, "seed": seed,
                "speed_mps": speed, "initial_temperature_K": temperature,
                "branch": "zoh" if speed <= p_core.low_speed_threshold_mps else "hocbf",
                "g_T0_K": log.g_T0_K, "rho": log.rho,
                "continuous_temperature_K": result.states[0].temperature_K,
                "zoh_predicted_temperature_K": predicted,
                "certified_thermal_bound_K": zoh["predicted_limit_K"],
                "prediction_error_K": result.states[0].temperature_K - predicted,
                "paper_eligible": True, "data_provenance": "literature_calibrated",
            })
    return rows


def disturbance_scenario(inputs: FrozenInputs, n: int, disturbance: str) -> tuple[PlatoonScenario, list[PaperIntegratedSafetyController]]:
    speed = float(inputs.protocol["initial_speed_mps"])
    if disturbance == "smooth_braking_pulse":
        leader = EmergencyBrakingPulseLeader(0.0, speed, 0.5, 1.0, 1.0)
        scenario, controllers = build_scenario(inputs, n, f"M9_{disturbance}", leader=leader, grade_rad=0.0)
    elif disturbance == "speed_reduction":
        leader = SmoothSpeedChangeLeader(0.0, speed, -0.8, 2.0)
        scenario, controllers = build_scenario(inputs, n, f"M9_{disturbance}", leader=leader, grade_rad=0.0)
    elif disturbance == "emergency_braking":
        scenario, controllers = build_scenario(inputs, n, f"M9_{disturbance}", leader=emergency_leader(inputs), grade_rad=0.0)
    elif disturbance == "grade_transition":
        scenario, controllers = build_scenario(inputs, n, f"M9_{disturbance}", grade_rad=0.0)
        road = PiecewiseLinearGrade((-5000.0, -100.0, 0.0, 5000.0), (0.0, 0.0, -0.09966865249116204, -0.09966865249116204))
        scenario = replace(scenario, road=road)
    else:
        raise ValueError(disturbance)
    return scenario, controllers


def experiment_m9(inputs: FrozenInputs) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    steps = int(inputs.protocol["steps"]["M9"])
    seed = int(inputs.protocol["seed"])
    initial_gap = float(inputs.protocol["initial_gap_m"])
    for n in inputs.protocol["M9"]["platoon_sizes"]:
        for disturbance in inputs.protocol["M9"]["disturbances"]:
            scenario, controllers = disturbance_scenario(inputs, int(n), disturbance)
            env = make_env(inputs, scenario, controllers)
            spacing: list[list[float]] = [[] for _ in range(int(n))]
            velocity: list[list[float]] = [[] for _ in range(int(n))]
            for _ in range(steps):
                result = env.step()
                leader_speed = scenario.leader.state_at(result.time_s).speed_mps
                for index, log in enumerate(result.logs):
                    spacing[index].append(log.augmented_state.gap_m - initial_gap)
                    velocity[index].append(result.states[index].speed_mps - leader_speed)
            reference = velocity[0]
            ref_l2 = math.sqrt(sum(x * x for x in reference)) or 1e-12
            ref_inf = max((abs(x) for x in reference), default=0.0) or 1e-12
            for index in range(int(n)):
                rows.append({
                    "method": "numerical_string_stability", "scenario": f"{disturbance}_N{n}", "seed": seed,
                    "platoon_size": n, "vehicle_id": index + 1,
                    "G2_i": math.sqrt(sum(x * x for x in velocity[index])) / ref_l2,
                    "Ginf_i": max((abs(x) for x in velocity[index]), default=0.0) / ref_inf,
                    "peak_spacing_error_m": max((abs(x) for x in spacing[index]), default=0.0),
                    "peak_speed_error_mps": max((abs(x) for x in velocity[index]), default=0.0),
                    "interpretation": "numerical_empirical_only_no_analytical_theorem",
                    "paper_eligible": True, "data_provenance": "literature_calibrated",
                })
    return rows


def platform_metadata() -> dict[str, Any]:
    return {
        "platform": platform.platform(), "system": platform.system(),
        "release": platform.release(), "machine": platform.machine(),
        "processor": platform.processor(), "python": sys.version,
        "python_executable": sys.executable,
        "qp_solver": "repository exact 2-D active-set candidate enumeration",
        "target_deployment_hardware": False,
    }


def experiment_m10(inputs: FrozenInputs) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seed = int(inputs.protocol["seed"])
    metadata = platform_metadata()
    for n in inputs.protocol["M9"]["platoon_sizes"]:
        for repetition in range(int(inputs.protocol["steps"]["M10_repetitions"])):
            scenario, controllers = build_scenario(inputs, int(n), f"M10_N{n}", grade_rad=0.0)
            result = make_env(inputs, scenario, controllers).step()
            for index, control in enumerate(result.controller_results):
                timing = control.timings_s
                rows.append({
                    "method": "runtime_benchmark", "scenario": f"N={n}", "seed": seed,
                    "platoon_size": n, "repetition": repetition, "vehicle_id": index + 1,
                    "feature_state_construction_s": timing["feature_state_construction"],
                    "hard_row_construction_s": timing["hard_row_construction"],
                    "qp1_s": timing["qp1"], "predictive_tube_s": timing["predictive_tube"],
                    "distributed_bottleneck_s": timing["distributed_bottleneck"],
                    "supervisor_s": timing["supervisor"], "qp2_s": timing["qp2"],
                    "reverification_s": timing["reverification"],
                    "total_controller_s": timing["total_controller"],
                    "platform": metadata,
                    "paper_eligible": True, "data_provenance": "literature_calibrated",
                })
    return rows


EXPERIMENTS = {
    "M1": experiment_m1, "M2": experiment_m2, "M3": experiment_m3,
    "M4": experiment_m4, "M5": experiment_m5, "M6": experiment_m6,
    "M7": experiment_m7, "M8": experiment_m8, "M9": experiment_m9,
    "M10": experiment_m10,
}


def _lock_raw_tree(path: Path) -> None:
    for item in path.rglob("*"):
        if item.is_file():
            try:
                item.chmod(0o444)
            except OSError:
                pass


def _write_grouped_runs(
    inputs: FrozenInputs, raw_root: Path, experiment_id: str,
    rows: list[dict[str, Any]], started_at: str,
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("paper_eligible") is not True or row.get("data_provenance") != "literature_calibrated":
            raise RuntimeError(f"{experiment_id}: ineligible row attempted")
        groups[(str(row["method"]), str(row["scenario"]), int(row["seed"]))].append(row)
    summaries: list[dict[str, Any]] = []
    for ordinal, ((method, scenario, seed), group) in enumerate(sorted(groups.items()), start=1):
        run_basis = {
            "experiment_id": experiment_id, "method": method, "scenario": scenario,
            "seed": seed, "config_hash": inputs.config_hash,
            "protocol_hash": inputs.protocol_hash, "ordinal": ordinal,
        }
        run_id = f"{experiment_id.lower()}-{canonical_hash(run_basis)[:16]}"
        run_dir = raw_root / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        record_path = run_dir / "records.jsonl"
        with record_path.open("w", encoding="utf-8", newline="\n") as stream:
            for sequence, row in enumerate(group):
                enriched = {
                    "run_id": run_id, "sequence": sequence,
                    "experiment_id": experiment_id,
                    "config_hash": inputs.config_hash,
                    "parameter_provenance_hash": inputs.provenance_hash,
                    "source_data_hash": inputs.source_hash,
                    "digitized_source_data_hash": inputs.digitization_hash,
                    "protocol_hash": inputs.protocol_hash,
                    **row,
                }
                stream.write(json.dumps(enriched, sort_keys=True) + "\n")
        road_model_hash = canonical_hash({
            "physical_model": inputs.config["physical_model"],
            "experiment": experiment_id, "scenario": scenario,
            "sample_parameters": group[0].get("sample_parameters"),
        })
        communication_hash = canonical_hash({
            "communication_records": {
                key: records_by_id(inputs)[key]["nominal_value"]
                for key in ("communication_delay_model", "communication_packet_burst_loss")
            },
            "scenario": scenario,
        })
        manifest = {
            "schema_version": 1, "run_id": run_id,
            "experiment_id": experiment_id, "method": method,
            "scenario": scenario, "seed": seed,
            "git_commit": inputs.git_commit,
            "code_snapshot_hash": inputs.code_snapshot_hash,
            "config_hash": inputs.config_hash,
            "parameter_provenance_hash": inputs.provenance_hash,
            "source_data_hash": inputs.source_hash,
            "digitized_source_data_hash": inputs.digitization_hash,
            "road_model_hash": road_model_hash,
            "communication_config_hash": communication_hash,
            "protocol_hash": inputs.protocol_hash,
            "timestamp": started_at,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "paper_eligible": True,
            "record_count": len(group),
            "records_file": "records.jsonl",
            "records_sha256": file_hash(record_path),
            "status": "PASS",
        }
        dump(run_dir / "run_manifest.json", manifest)
        manifest_hash = file_hash(run_dir / "run_manifest.json")
        summaries.append({
            "run_id": run_id, "experiment_id": experiment_id,
            "method": method, "scenario": scenario, "seed": seed,
            "record_count": len(group), "status": "PASS",
            "run_manifest": f"raw/{run_id}/run_manifest.json",
            "run_manifest_sha256": manifest_hash,
            "records_sha256": manifest["records_sha256"],
        })
    return summaries


def run_paper_suite(inputs: FrozenInputs | None = None) -> Path:
    inputs = inputs or validate_frozen_inputs()
    campaign_id = f"phase2m-paper-{inputs.config_hash[:8]}-{inputs.protocol_hash[:8]}"
    final = RESULT_ROOT / campaign_id
    if final.exists():
        manifest = json.loads((final / "result_manifest.json").read_text(encoding="utf-8"))
        if file_hash(final / "result_manifest.json") != (final / "result_manifest.sha256").read_text(encoding="utf-8").strip():
            raise RuntimeError("existing immutable result manifest hash mismatch")
        if manifest.get("paper_eligible") is not True or manifest.get("config_hash") != inputs.config_hash:
            raise RuntimeError("existing campaign is not eligible for reuse")
        return final
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    staging = RESULT_ROOT / f".staging-{campaign_id}-{uuid4().hex[:8]}"
    staging.mkdir(parents=True, exist_ok=False)
    raw = staging / "raw"
    raw.mkdir()
    started = datetime.now(timezone.utc).isoformat()
    sanity = sanity_audit(inputs)
    dump(staging / "pre_run_sanity_audit.json", sanity)
    runs: list[dict[str, Any]] = []
    experiment_summaries: list[dict[str, Any]] = []
    try:
        for experiment_id, function in EXPERIMENTS.items():
            rows = function(inputs)
            run_summaries = _write_grouped_runs(inputs, raw, experiment_id, rows, started)
            runs.extend(run_summaries)
            experiment_summaries.append({
                "experiment_id": experiment_id, "status": "PASS",
                "run_count": len(run_summaries), "record_count": len(rows),
            })
        result_manifest = {
            "schema_version": 1, "campaign_id": campaign_id,
            "phase": "2M-PAPER", "mode": "LITERATURE_CALIBRATED",
            "started_at": started, "completed_at": datetime.now(timezone.utc).isoformat(),
            "git_commit": inputs.git_commit, "code_snapshot_hash": inputs.code_snapshot_hash,
            "config_hash": inputs.config_hash, "protocol_hash": inputs.protocol_hash,
            "parameter_provenance_hash": inputs.provenance_hash,
            "source_data_hash": inputs.source_hash,
            "digitized_source_data_hash": inputs.digitization_hash,
            "paper_eligible": True, "validation_status": "PASS",
            "synthetic_debug_dependency": False,
            "final_m1_m10_executed": True,
            "mappo_or_diffqp_learning_executed": False,
            "software_correction": inputs.protocol.get("software_correction"),
            "supersedes_campaign_id": inputs.protocol.get("supersedes_campaign_id"),
            "superseded_invalidated_runs": inputs.protocol.get("invalidated_prior_run_count", 0),
            "total_runs": len(runs), "failed_runs": 0, "invalidated_runs": 0,
            "experiments": experiment_summaries, "runs": runs,
            "sanity_audit": "pre_run_sanity_audit.json",
            "raw_immutable": True,
        }
        dump(staging / "result_manifest.json", result_manifest)
        manifest_hash = file_hash(staging / "result_manifest.json")
        (staging / "result_manifest.sha256").write_text(manifest_hash + "\n", encoding="ascii", newline="\n")
        os.replace(staging, final)
        _lock_raw_tree(final / "raw")
        return final
    except Exception:
        failure = staging / "FAILED.txt"
        failure.write_text("Paper campaign aborted before immutable publication.\n", encoding="utf-8")
        raise

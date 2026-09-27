"""Phase 2M non-learning model-simulation experiments M1--M10.

All values produced here are synthetic DEBUG software-validation artifacts.
No function in this module can mark a run paper eligible.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
import json
from math import sqrt
from pathlib import Path
from typing import Iterable
import random

from controllers.debug_nominal import ConstantCommandController
from controllers.integrated_safety_controller import IntegratedSafetyController
from envs.heavy_platoon_env import HeavyPlatoonEnv
from envs.leader_profile import ConstantSpeedLeader, EmergencyBrakingPulseLeader, SmoothSpeedChangeLeader
from envs.road_profile import ConstantGrade, PiecewiseLinearGrade
from envs.scenario import PlatoonScenario, build_synthetic_debug_scenario
from network.v2v_channel import ChannelConfig
from safety_core import State, fade_factor, low_speed_temperature_row


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "model_simulation" / "debug_suite.json"
DEFAULT_PARAMETERS = ROOT / "data" / "model_parameters" / "synthetic_debug.json"
DEFAULT_OUTPUT = ROOT / "artifacts" / "model_based" / "debug"


def _canonical_hash(document: object) -> str:
    return sha256(json.dumps(document, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _file_hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class DemandSplitController:
    target_speed_mps: float
    gain_N_per_mps: float
    mode: str
    friction_cap_N: float = 60_000.0
    auxiliary_cap_N: float = 45_000.0
    data_provenance: str = "synthetic_debug"
    paper_eligible: bool = False

    def command(self, observation: object) -> tuple[float, float]:
        demand = max(0.0, self.gain_N_per_mps * (observation.own_speed_mps - self.target_speed_mps))
        if self.mode == "friction_only":
            return demand, 0.0
        if self.mode == "friction_first":
            friction = min(demand, self.friction_cap_N)
            return friction, max(0.0, demand - friction)
        if self.mode == "auxiliary_first":
            auxiliary = min(demand, self.auxiliary_cap_N)
            return max(0.0, demand - auxiliary), auxiliary
        if self.mode == "hard_qp_dual":
            return 0.5 * demand, 0.5 * demand
        raise ValueError(f"unknown DEBUG demand split {self.mode}")


class ModelRunLogger:
    def __init__(self, output: Path, config: dict[str, object], parameter_path: Path) -> None:
        if config.get("data_provenance") != "synthetic_debug" or config.get("paper_eligible") is not False:
            raise ValueError("Phase 2M DEBUG logger rejects paper-eligible configuration")
        self.output = output
        self.output.mkdir(parents=True, exist_ok=True)
        self.config_hash = _canonical_hash(config)
        self.provenance_hash = _file_hash(parameter_path)
        self.seed = int(config["seed"])
        self.run_id = f"phase2m-debug-{self.config_hash[:12]}"
        self.counts: dict[str, int] = {}

    def write(self, experiment_id: str, records: Iterable[dict[str, object]]) -> int:
        path = self.output / f"{experiment_id.lower()}_records.jsonl"
        count = 0
        with path.open("w", encoding="utf-8", newline="\n") as stream:
            for raw in records:
                record = {
                    "run_id": self.run_id,
                    "experiment_id": experiment_id,
                    "config_hash": self.config_hash,
                    "source_provenance_hash": self.provenance_hash,
                    "seed": self.seed,
                    "data_provenance": "synthetic_debug",
                    "paper_eligible": False,
                    **raw,
                }
                if record["paper_eligible"] is not False:
                    raise ValueError("DEBUG record cannot become paper eligible")
                stream.write(json.dumps(record, sort_keys=True) + "\n")
                count += 1
        self.counts[experiment_id] = count
        return count


def _parameterized_scenario(
    n: int,
    variant: str,
    *,
    speed_mps: float | None = None,
    temperature_K: float | None = None,
    grade_rad: float | None = None,
    mass_scale: float = 1.0,
    auxiliary_scale: float = 1.0,
    tau_f_scale: float = 1.0,
    tau_a_scale: float = 1.0,
    thermal_capacity_scale: float = 1.0,
    cooling_scale: float = 1.0,
    friction_limit_scale: float = 1.0,
    channel: ChannelConfig | None = None,
) -> PlatoonScenario:
    scenario = build_synthetic_debug_scenario(n, variant, channel)
    parameters = tuple(replace(
        item,
        mass_kg=item.mass_kg * mass_scale,
        tau_f_s=item.tau_f_s * tau_f_scale,
        tau_a_s=item.tau_a_s * tau_a_scale,
        thermal_capacity_J_per_K=item.thermal_capacity_J_per_K * thermal_capacity_scale,
        cooling_W_per_K=item.cooling_W_per_K * cooling_scale,
        friction_force_limit_N=item.friction_force_limit_N * friction_limit_scale,
        auxiliary_force_limit_N=item.auxiliary_force_limit_N * auxiliary_scale,
    ) for item in scenario.parameters)
    states = tuple(replace(
        state,
        speed_mps=state.speed_mps if speed_mps is None else speed_mps,
        temperature_K=state.temperature_K if temperature_K is None else temperature_K,
    ) for state in scenario.initial_states)
    leader = scenario.leader if speed_mps is None else ConstantSpeedLeader(0.0, speed_mps)
    road = scenario.road if grade_rad is None else ConstantGrade(grade_rad)
    drive = tuple(
        item.mass_kg * item.gravity_mps2 * item.rolling_resistance_coefficient
        for item in parameters
    )
    return replace(scenario, parameters=parameters, initial_states=states, leader=leader, road=road, drive_force_N=drive)


def _cycle_record(env: HeavyPlatoonEnv, step_result: object, index: int, method: str) -> dict[str, object]:
    log = step_result.logs[index]
    controller = step_result.controller_results[index]
    prediction = controller.two_pass.final_prediction
    instantaneous = controller.instantaneous_result
    next_state = step_result.states[index]
    max_tube_width = max((state.aggregate_width() for state in prediction.reachable_sets), default=0.0)
    return {
        "method": method,
        "time_s": log.time_s,
        "vehicle_id": log.vehicle_id,
        "position_m": log.physical_state.position_m,
        "speed_mps": log.physical_state.speed_mps,
        "gap_m": log.augmented_state.gap_m,
        "u_f_nom_N": log.nominal_action[0],
        "u_a_nom_N": log.nominal_action[1],
        "u_f_provisional_N": log.provisional_action[0],
        "u_a_provisional_N": log.provisional_action[1],
        "u_f_final_N": log.final_action[0],
        "u_a_final_N": log.final_action[1],
        "friction_force_N": log.physical_state.friction_force_N,
        "auxiliary_force_N": log.physical_state.auxiliary_force_N,
        "temperature_K": log.physical_state.temperature_K,
        "next_temperature_K": next_state.temperature_K,
        "fade_factor": fade_factor(log.physical_state.temperature_K, env.controllers[index].config.vehicle),
        "h_c": log.barriers["collision_h"],
        "h_T": log.barriers["temperature_h"],
        "h_F": log.barriers["fade_h"],
        "h_A": log.barriers["auxiliary_h"],
        "Delta_f_N": log.delta_f_N,
        "Delta_a_N": log.delta_a_N,
        "friction_lower_bound_N": instantaneous.L_f,
        "friction_upper_bound_N": instantaneous.U_f,
        "auxiliary_lower_bound_N": instantaneous.L_a,
        "auxiliary_upper_bound_N": instantaneous.U_a,
        "collision_demand_mps2": instantaneous.D_c,
        "M_mps2": log.M_mps2,
        "rho": log.rho,
        "g_T0_K": log.g_T0_K,
        "rho_H_cert": log.rho_H_cert,
        "rho_fleet_centralized": log.centralized_fleet_rho_H_cert,
        "rho_up_local": log.local_recursive_rho_up,
        "critical_vehicle_id": log.critical_vehicle_id,
        "critical_prediction_step": log.critical_prediction_step,
        "critical_component": log.critical_component,
        "predictive_reserve_by_step": list(prediction.rho_ia_lower_by_step),
        "backup_control_widths_N": [list(item) for item in prediction.backup_control_widths_N],
        "tube_aggregate_width_max": max_tube_width,
        "message_age_s": log.message_age_s,
        "message_valid": log.message_valid,
        "supervisor_mode": log.supervisor_mode,
        "qp_status": log.qp_status,
        "hard_residuals": log.hard_row_residuals,
        "runtime_s": controller.timings_s,
    }


def _run_env(env: HeavyPlatoonEnv, steps: int, method: str) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for _ in range(steps):
        result = env.step()
        records.extend(_cycle_record(env, result, index, method) for index in range(env.controlled_truck_count))
    return records


def experiment_m1(config: dict[str, object]) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    grid = config["M1"]
    case = 0
    for speed in grid["speed_mps"]:
        for temperature in grid["temperature_K"]:
            for grade in grid["grade_rad"]:
                for mass_scale in grid["mass_scale"]:
                    for auxiliary_scale in grid["auxiliary_availability"]:
                        scenario = _parameterized_scenario(
                            1, "constant_descent", speed_mps=speed, temperature_K=temperature,
                            grade_rad=grade, mass_scale=mass_scale, auxiliary_scale=auxiliary_scale,
                        )
                        env = HeavyPlatoonEnv(scenario, [ConstantCommandController(0.0, 0.0)])
                        result = env.step()
                        controller = result.controller_results[0]
                        instant = controller.instantaneous_result
                        collision_row = next(row for row in controller.hard_rows if row.name == "collision_hocbf")
                        A_f, A_a, D_c = -collision_row.g_f, -collision_row.g_a, -collision_row.h
                        records.append({
                            "case_id": case, "speed_mps": speed, "temperature_K": temperature,
                            "grade_rad": grade, "mass_kg": scenario.parameters[0].mass_kg,
                            "auxiliary_availability": auxiliary_scale,
                            "Delta_f_N": instant.U_f - instant.L_f,
                            "Delta_a_N": instant.U_a - instant.L_a,
                            "D_c_mps2": D_c,
                            "M_friction_only_mps2": A_f * instant.U_f - D_c,
                            "M_dual_brake_mps2": instant.M,
                            "rho": controller.instantaneous_certificate.rho,
                            "friction_only_feasible": instant.L_f <= instant.U_f and A_f * instant.U_f >= D_c,
                            "dual_brake_feasible": instant.feasible,
                            "active_friction_limit": instant.active_friction_limit,
                            "active_auxiliary_limit": instant.active_auxiliary_limit,
                        })
                        case += 1
    return records


def experiment_m2(config: dict[str, object]) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for mode in ("friction_only", "friction_first", "auxiliary_first", "hard_qp_dual"):
        scenario = _parameterized_scenario(3, "constant_descent")
        controllers = [DemandSplitController(17.0, 8_000.0, mode) for _ in range(3)]
        records.extend(_run_env(HeavyPlatoonEnv(scenario, controllers), int(config["steps"]["M2"]), mode))
    return records


def experiment_m3(config: dict[str, object]) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for temperature in config["M3"]["initial_temperature_K"]:
        scenario = _parameterized_scenario(3, "constant_descent", temperature_K=temperature)
        case_records = _run_env(HeavyPlatoonEnv(scenario), int(config["steps"]["M3"]), f"T0={temperature:g}K")
        for record in case_records:
            record["initial_temperature_K"] = temperature
        records.extend(case_records)
    return records


def experiment_m4(config: dict[str, object]) -> list[dict[str, object]]:
    records = _run_env(
        HeavyPlatoonEnv(_parameterized_scenario(3, "leader_braking")),
        int(config["steps"]["M4"]),
        "instantaneous_and_predictive",
    )
    trigger = min((r["time_s"] for r in records if r["rho_H_cert"] < 0.0), default=None)
    boundary = min((r["time_s"] for r in records if r["rho"] < 0.0), default=None)
    warning = None if trigger is None or boundary is None else boundary - trigger
    for record in records:
        record["predictive_trigger_time_s"] = trigger
        record["instantaneous_boundary_time_s"] = boundary
        record["warning_time_s"] = warning
        record["negative_predictive_semantics"] = "conservative_warning_not_true_infeasibility"
    return records


def experiment_m5(config: dict[str, object]) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for horizon in config["M5"]["horizons"]:
        env = HeavyPlatoonEnv(_parameterized_scenario(3, "leader_braking"))
        env.controllers = [
            IntegratedSafetyController(replace(controller.config, horizon=int(horizon)))
            for controller in env.controllers
        ]
        case_records = _run_env(env, int(config["steps"]["M5"]), f"H={horizon}")
        for record in case_records:
            record["H_pred"] = horizon
            record["conservative_negative_warning"] = record["rho_H_cert"] < 0.0 <= record["rho"]
            running_min: list[float] = []
            for value in record["predictive_reserve_by_step"]:
                running_min.append(min(running_min[-1], value) if running_min else value)
            record["stored_prefix_reserve"] = running_min
            record["stored_prefix_monotonicity_verified"] = all(
                later <= earlier for earlier, later in zip(running_min, running_min[1:])
            )
        records.extend(case_records)
    return records


def _adjust_fade(env: HeavyPlatoonEnv, scale: float) -> None:
    env.controllers = [
        IntegratedSafetyController(replace(
            controller.config,
            vehicle=replace(controller.config.vehicle, fade_slope_per_K=controller.config.vehicle.fade_slope_per_K * scale),
        ))
        for controller in env.controllers
    ]


def experiment_m6(config: dict[str, object]) -> list[dict[str, object]]:
    perturbation = float(config["M6"]["deterministic_relative_perturbation"])
    steps = int(config["steps"]["M6"])
    records: list[dict[str, object]] = []
    independent: list[tuple[str, dict[str, object]]] = []
    for sign, label in ((-1.0, "minus"), (1.0, "plus")):
        factor = 1.0 + sign * perturbation
        independent.extend((
            (f"mass_{label}", {"mass_scale": factor}),
            (f"road_grade_{label}", {"grade_rad": -0.03 * factor}),
            (f"friction_lag_{label}", {"tau_f_scale": factor}),
            (f"auxiliary_lag_{label}", {"tau_a_scale": factor}),
            (f"thermal_capacity_{label}", {"thermal_capacity_scale": factor}),
            (f"cooling_{label}", {"cooling_scale": factor}),
            (f"friction_force_limit_{label}", {"friction_limit_scale": factor}),
            (f"auxiliary_availability_{label}", {"auxiliary_scale": factor}),
        ))
    independent.extend((
        ("communication_delay_low", {"channel": ChannelConfig(delay_model="fixed", fixed_delay_s=0.1)}),
        ("communication_delay_high", {"channel": ChannelConfig(delay_model="fixed", fixed_delay_s=0.2)}),
    ))
    for name, kwargs in independent:
        env = HeavyPlatoonEnv(_parameterized_scenario(3, "constant_descent", **kwargs))
        case_records = _run_env(env, steps, name)
        for record in case_records:
            record.update({"perturbation_mode": "deterministic_independent", "perturbed_parameter": name})
        records.extend(case_records)
    for factor, label in ((1.0 - perturbation, "minus"), (1.0 + perturbation, "plus")):
        fade_env = HeavyPlatoonEnv(_parameterized_scenario(3, "constant_descent"))
        _adjust_fade(fade_env, factor)
        fade_records = _run_env(fade_env, steps, f"fade_parameters_{label}")
        for record in fade_records:
            record.update({"perturbation_mode": "deterministic_independent", "perturbed_parameter": f"fade_parameters_{label}"})
        records.extend(fade_records)
        predecessor = _parameterized_scenario(3, "leader_braking")
        predecessor = replace(predecessor, leader=EmergencyBrakingPulseLeader(0.0, 18.0, 0.2, 0.5, 3.0 * factor))
        pred_records = _run_env(HeavyPlatoonEnv(predecessor), steps, f"predecessor_motion_{label}")
        for record in pred_records:
            record.update({"perturbation_mode": "deterministic_independent", "perturbed_parameter": f"predecessor_motion_{label}"})
        records.extend(pred_records)

    rng = random.Random(int(config["seed"]))
    lower, upper = config["M6"]["debug_distribution_bounds"]
    for sample_id in range(int(config["M6"]["joint_monte_carlo_samples"])):
        factor = lambda: 1.0 + rng.uniform(lower, upper)
        scenario = _parameterized_scenario(
            3, "leader_braking", mass_scale=factor(), grade_rad=-0.03 * factor(),
            tau_f_scale=factor(), tau_a_scale=factor(), thermal_capacity_scale=factor(),
            cooling_scale=factor(), friction_limit_scale=factor(), auxiliary_scale=factor(),
            channel=ChannelConfig(delay_model="random_bounded", maximum_random_delay_s=0.2, random_seed=int(config["seed"]) + sample_id),
        )
        env = HeavyPlatoonEnv(scenario)
        _adjust_fade(env, factor())
        case_records = _run_env(env, steps, f"joint_mc_{sample_id}")
        for record in case_records:
            record.update({
                "perturbation_mode": "joint_monte_carlo",
                "debug_distribution": config["M6"]["debug_distribution"],
                "sample_id": sample_id,
            })
        records.extend(case_records)
    return records


def experiment_m7(config: dict[str, object]) -> list[dict[str, object]]:
    channels = {
        "zero_delay": ChannelConfig(),
        "bounded_delay": ChannelConfig(delay_model="random_bounded", maximum_random_delay_s=0.2, random_seed=int(config["seed"])),
        "packet_loss": ChannelConfig(packet_loss_probability=0.2, random_seed=int(config["seed"])),
        "burst_loss": ChannelConfig(burst_length=2, forced_burst_starts=frozenset({1})),
        "stale_packets": ChannelConfig(delay_model="fixed", fixed_delay_s=0.5),
    }
    records: list[dict[str, object]] = []
    for name, channel in channels.items():
        env = HeavyPlatoonEnv(_parameterized_scenario(3, "communication_delay_loss", channel=channel))
        for _ in range(int(config["steps"]["M7"])):
            result = env.step()
            central_record = min(result.logs, key=lambda item: item.rho_H_cert)
            for index, log in enumerate(result.logs):
                row = _cycle_record(env, result, index, name)
                trigger = env.controllers[index].config.supervisor_thresholds.predictive_trigger
                centralized_trigger = central_record.rho_H_cert <= trigger
                local_trigger = log.local_recursive_rho_up <= trigger
                row.update({
                    "channel_case": name,
                    "central_critical_vehicle_id": central_record.vehicle_id,
                    "central_critical_component": central_record.critical_component,
                    "critical_vehicle_agreement": log.critical_vehicle_id == central_record.vehicle_id,
                    "critical_component_agreement": log.critical_component == central_record.critical_component,
                    "anticipatory_trigger": log.supervisor_mode == "anticipatory",
                    "centralized_predictive_trigger": centralized_trigger,
                    "local_recursive_predictive_trigger": local_trigger,
                    "missed_anticipatory_trigger": centralized_trigger and not local_trigger,
                    "false_anticipatory_trigger": local_trigger and not centralized_trigger,
                })
                records.append(row)
    return records


def experiment_m8(config: dict[str, object]) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for speed in config["M8"]["speed_mps"]:
        for temperature in config["M8"]["temperature_K"]:
            scenario = _parameterized_scenario(1, "low_speed", speed_mps=speed, temperature_K=temperature)
            env = HeavyPlatoonEnv(scenario, [ConstantCommandController(0.0, 0.0)])
            initial = env.states[0]
            result = env.step()
            log = result.logs[0]
            controller = result.controller_results[0]
            vehicle = controller.instantaneous_result
            p = env.controllers[0].config.vehicle
            acceleration = env._physical_acceleration(1, initial)
            row = low_speed_temperature_row(
                State(initial.speed_mps, initial.friction_force_N, initial.auxiliary_force_N, initial.temperature_K, acceleration),
                p, scenario.ambient_temperature_K, scenario.residual_heat_W,
                scenario.parameters[0].critical_temperature_K, 0.5,
            )
            predicted = row["base_K"] + row["coefficient_K_per_N"] * log.final_action[0]
            records.append({
                "speed_mps": speed, "initial_temperature_K": temperature,
                "speed_relation": "below_or_at_epsilon" if speed <= p.low_speed_threshold_mps else "above_epsilon",
                "g_T0_K": log.g_T0_K, "rho": log.rho,
                "zoh_predicted_temperature_K": predicted,
                "continuous_plant_temperature_K": result.states[0].temperature_K,
                "temperature_prediction_error_K": result.states[0].temperature_K - predicted,
                "low_speed_thermal_feasible": vehicle.low_speed_thermal_feasible,
            })
    return records


def _disturbance_scenario(n: int, disturbance: str) -> PlatoonScenario:
    scenario = _parameterized_scenario(n, "flat_steady")
    if disturbance == "smooth_braking_pulse":
        return replace(scenario, leader=EmergencyBrakingPulseLeader(0.0, 18.0, 0.1, 0.8, 1.0))
    if disturbance == "speed_reduction":
        return replace(scenario, leader=SmoothSpeedChangeLeader(0.0, 18.0, -0.8, 1.0))
    if disturbance == "emergency_braking":
        return replace(scenario, leader=EmergencyBrakingPulseLeader(0.0, 18.0, 0.1, 0.5, 3.0))
    if disturbance == "grade_transition":
        return replace(scenario, road=PiecewiseLinearGrade((-5000.0, -100.0, 0.0, 5000.0), (0.0, 0.0, -0.03, -0.03)))
    raise ValueError(disturbance)


def experiment_m9(config: dict[str, object]) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    steps = int(config["steps"]["M9"])
    for n in config["M9"]["platoon_sizes"]:
        for disturbance in config["M9"]["disturbances"]:
            env = HeavyPlatoonEnv(_disturbance_scenario(int(n), disturbance))
            spacing: list[list[float]] = [[] for _ in range(int(n))]
            velocity: list[list[float]] = [[] for _ in range(int(n))]
            for _ in range(steps):
                result = env.step()
                leader_speed = env.scenario.leader.state_at(result.time_s).speed_mps
                for index, log in enumerate(result.logs):
                    spacing[index].append(log.augmented_state.gap_m - 65.0)
                    velocity[index].append(result.states[index].speed_mps - leader_speed)
            reference = velocity[0]
            ref_l2 = sqrt(sum(value * value for value in reference)) or 1e-12
            ref_inf = max((abs(value) for value in reference), default=0.0) or 1e-12
            for index in range(int(n)):
                records.append({
                    "platoon_size": n, "disturbance": disturbance, "vehicle_id": index + 1,
                    "G2_i": sqrt(sum(value * value for value in velocity[index])) / ref_l2,
                    "Ginf_i": max((abs(value) for value in velocity[index]), default=0.0) / ref_inf,
                    "peak_spacing_error_m": max((abs(value) for value in spacing[index]), default=0.0),
                    "peak_velocity_error_mps": max((abs(value) for value in velocity[index]), default=0.0),
                    "study_type": "numerical_empirical_no_theorem",
                })
    return records


def experiment_m10(config: dict[str, object]) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for n in config["M9"]["platoon_sizes"]:
        for repetition in range(int(config["steps"]["M10_repetitions"])):
            env = HeavyPlatoonEnv(_parameterized_scenario(int(n), "flat_steady"))
            result = env.step()
            for index, controller in enumerate(result.controller_results):
                records.append({
                    "platoon_size": n,
                    "repetition": repetition,
                    "vehicle_id": index + 1,
                    **{f"{name}_s": value for name, value in controller.timings_s.items()},
                    "hardware_claim_allowed": False,
                })
    return records


EXPERIMENTS = {
    "M1": experiment_m1,
    "M2": experiment_m2,
    "M3": experiment_m3,
    "M4": experiment_m4,
    "M5": experiment_m5,
    "M6": experiment_m6,
    "M7": experiment_m7,
    "M8": experiment_m8,
    "M9": experiment_m9,
    "M10": experiment_m10,
}


def run_debug_suite(
    config_path: Path = DEFAULT_CONFIG,
    parameter_path: Path = DEFAULT_PARAMETERS,
    output: Path = DEFAULT_OUTPUT,
) -> dict[str, object]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    parameters = json.loads(parameter_path.read_text(encoding="utf-8"))
    if parameters.get("paper_eligible") is not False or any(
        item.get("paper_eligible") is not False for item in parameters["parameters"]
    ):
        raise ValueError("DEBUG parameter registry must be entirely paper ineligible")
    logger = ModelRunLogger(output, config, parameter_path)
    experiment_summaries: list[dict[str, object]] = []
    for experiment_id, function in EXPERIMENTS.items():
        records = function(config)
        count = logger.write(experiment_id, records)
        experiment_summaries.append({"experiment_id": experiment_id, "record_count": count, "status": "PASS"})
    manifest = {
        "schema_version": 1,
        "run_id": logger.run_id,
        "mode": "DEBUG_SYNTHETIC",
        "data_provenance": "synthetic_debug",
        "paper_eligible": False,
        "config_hash": logger.config_hash,
        "source_provenance_hash": logger.provenance_hash,
        "seed": logger.seed,
        "parameter_dependency_count": len(parameters["parameters"]),
        "all_parameter_sources_literature_calibrated": False,
        "experiments": experiment_summaries,
        "publication_gate": "FAIL_CLOSED_DEBUG_INPUT",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")
    return manifest

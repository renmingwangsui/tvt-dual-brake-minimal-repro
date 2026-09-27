"""Targeted Phase 2M follow-up without physical-model retuning.

The scenario grid is loaded from a file frozen before evaluation.  Only the
M8 branch diagnostic and the predictive-warning scenario family are run.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
from typing import Any
from uuid import uuid4

from envs.leader_profile import EmergencyBrakingPulseLeader
from experiments.model_based.paper_pipeline import (
    EXPECTED_CONFIG_HASH,
    RESULT_ROOT,
    build_scenario,
    build_vehicle,
    cycle_record,
    file_hash,
    make_env,
    records_by_id,
    validate_frozen_inputs,
)
from safety_core import State, low_speed_temperature_row, state_derivatives, thermal_affine


ROOT = Path(__file__).resolve().parents[2]
SPEC = ROOT / "configs" / "model_simulation" / "phase2m_followup_protocol_v1.json"


def _dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def validate_followup() -> tuple[Any, dict[str, Any], str]:
    inputs = validate_frozen_inputs()
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    spec_hash = file_hash(SPEC)
    if inputs.config_hash != EXPECTED_CONFIG_HASH or spec["expected_physical_config_sha256"] != EXPECTED_CONFIG_HASH:
        raise RuntimeError("frozen physical configuration mismatch")
    parent = ROOT / spec["parent_campaign"] / "result_manifest.json"
    if file_hash(parent) != spec["parent_result_manifest_sha256"]:
        raise RuntimeError("parent result manifest mismatch")
    if spec.get("no_physical_retuning") is not True or spec.get("frozen_before_followup_evaluation") is not True:
        raise RuntimeError("follow-up specification is not frozen/no-retuning")
    return inputs, spec, spec_hash


def _row_by_name(rows: tuple[Any, ...], name: str) -> Any:
    return next(row for row in rows if row.name == name)


def m8_dense_diagnostic(inputs: Any, spec: dict[str, Any]) -> list[dict[str, Any]]:
    design = spec["m8_diagnostic"]
    epsilon = float(design["epsilon_mps"])
    rows: list[dict[str, Any]] = []
    for temperature in design["temperatures_K"]:
        for offset in design["speed_offsets_mps"]:
            speed = epsilon + float(offset)
            scenario, controllers = build_scenario(
                inputs, 1, "followup_m8_dense", speed_mps=speed,
                temperature_K=float(temperature), grade_rad=0.0,
            )
            env = make_env(inputs, scenario, controllers, ["zero"])
            initial = env.states[0]
            result = env.step()
            control = result.controller_results[0]
            cert = control.instantaneous_certificate
            instant = control.instantaneous_result
            core = controllers[0].config.vehicle
            acceleration = env._physical_acceleration(1, initial)
            state = State(initial.speed_mps, initial.friction_force_N, initial.auxiliary_force_N, initial.temperature_K, acceleration)
            zoh = low_speed_temperature_row(
                state, core, scenario.ambient_temperature_K, scenario.residual_heat_W,
                scenario.parameters[0].critical_temperature_K,
                controllers[0].config.low_speed_model_error_margin_K,
            )
            rates = state_derivatives(
                state, (0.0, 0.0), core,
                acceleration * core.mass_kg,
                scenario.ambient_temperature_K, scenario.residual_heat_W,
            )
            hocbf = thermal_affine(
                state, core,
                scenario.parameters[0].critical_temperature_K - core.temperature_buffer_K,
                rates["T_dot"], 0.0, 0.0, 0.8, 1.1,
            )
            active = next(row for row in control.hard_rows if row.name in {"thermal_zoh", "thermal_zoh_state", "thermal_hocbf"})
            friction_base = min(
                _row_by_name(control.hard_rows, "friction_command_envelope").h,
                _row_by_name(control.hard_rows, "fade_cbf").h,
            )
            zoh_upper = zoh["g_T0_K"] / zoh["coefficient_K_per_N"] if zoh["coefficient_K_per_N"] > 0.0 else core.friction_command_limit_N
            hocbf_upper = hocbf["rhs"] / hocbf["B_f"] if hocbf["B_f"] > 0.0 else core.friction_command_limit_N
            auxiliary_upper = instant.U_a
            collision = _row_by_name(control.hard_rows, "collision_hocbf")
            A_f, A_a, demand = -collision.g_f, -collision.g_a, -collision.h
            zoh_U = min(friction_base, zoh_upper)
            hocbf_U = min(friction_base, hocbf_upper)
            zoh_M = A_f * zoh_U + A_a * auxiliary_upper - demand
            hocbf_M = A_f * hocbf_U + A_a * auxiliary_upper - demand
            scales = controllers[0].config.scales
            rows.append({
                "method": "m8_dense_branch_diagnostic", "scenario": f"T={float(temperature):.9f}",
                "seed": int(spec["seed"]), "speed_mps": speed, "speed_offset_mps": float(offset),
                "temperature_K": float(temperature), "epsilon_mps": epsilon,
                "active_mode": "zoh" if speed <= epsilon else "hocbf",
                "active_thermal_row": active.name,
                "active_thermal_coefficient": active.g_f,
                "active_thermal_rhs": active.h,
                "c_T_K_per_N": zoh["coefficient_K_per_N"], "g_T0_K": zoh["g_T0_K"],
                "zoh_thermal_upper_N": zoh_upper, "hocbf_B_f": hocbf["B_f"],
                "hocbf_rhs": hocbf["rhs"], "hocbf_thermal_upper_N": hocbf_upper,
                "friction_nonthermal_upper_N": friction_base,
                "U_f_N": instant.U_f, "U_f_collision_contribution_mps2": A_f * instant.U_f,
                "Delta_f_N": cert.delta_f_N, "M_mps2": cert.M_mps2,
                "normalized_friction": cert.normalized_friction,
                "normalized_auxiliary": cert.normalized_auxiliary,
                "normalized_collision": cert.normalized_collision,
                "normalized_low_speed_thermal": cert.normalized_low_speed_thermal,
                "rho": cert.rho, "feasible": cert.feasible,
                "critical_component": cert.critical_limit_type,
                "zoh_counterfactual": {
                    "U_f_N": zoh_U, "M_mps2": zoh_M,
                    "normalized_friction": zoh_U / scales.friction_force_N,
                    "normalized_collision": zoh_M / scales.collision_mps2,
                    "normalized_low_speed_thermal": zoh["g_T0_K"] / scales.low_speed_temperature_K,
                    "feasible": zoh_U >= 0.0 and zoh_M >= 0.0 and zoh["g_T0_K"] >= 0.0,
                },
                "hocbf_counterfactual": {
                    "U_f_N": hocbf_U, "M_mps2": hocbf_M,
                    "normalized_friction": hocbf_U / scales.friction_force_N,
                    "normalized_collision": hocbf_M / scales.collision_mps2,
                    "feasible": hocbf_U >= 0.0 and hocbf_M >= 0.0,
                },
                "paper_eligible": True, "data_provenance": "literature_calibrated",
            })
    return rows


def _crossings(rows: list[dict[str, Any]], key: str) -> list[float]:
    by_time: dict[float, float] = {}
    for row in rows:
        time = float(row["time_s"])
        by_time[time] = min(by_time.get(time, math.inf), float(row[key]))
    ordered = sorted(by_time.items())
    return [time for (prior_time, prior), (time, value) in zip(ordered, ordered[1:]) if prior >= 0.0 > value]


def _first_time(rows: list[dict[str, Any]], predicate: Any) -> float | None:
    return min((float(row["time_s"]) for row in rows if predicate(row)), default=None)


def _classify_warning(rows: list[dict[str, Any]]) -> dict[str, Any]:
    trigger = _first_time(rows, lambda row: float(row["rho_H_cert"]) < 0.0)
    crossings = _crossings(rows, "rho")
    boundary = crossings[0] if len(crossings) == 1 else None
    certificate_loss = trigger
    supervisor_transition = _first_time(rows, lambda row: row["supervisor_mode"] != "normal_actor")
    if boundary is not None:
        if trigger is None or trigger > boundary:
            classification = "MISSED_WARNING"
        elif trigger < boundary:
            classification = "TRUE_EARLY_WARNING"
        else:
            classification = "SIMULTANEOUS_WARNING"
    elif trigger is not None:
        classification = "CONSERVATIVE_WARNING_WITHOUT_BOUNDARY"
    else:
        classification = "NO_WARNING_NO_BOUNDARY"
    reference_time = trigger if trigger is not None else boundary
    event_rows = [row for row in rows if reference_time is not None and abs(float(row["time_s"]) - reference_time) <= 1e-12]
    critical = min(event_rows, key=lambda row: float(row["rho_H_cert"])) if event_rows else min(rows, key=lambda row: float(row["rho_H_cert"]))
    return {
        "classification": classification,
        "predictive_trigger_time_s": trigger,
        "instantaneous_boundary_time_s": boundary,
        "boundary_crossing_count": len(crossings),
        "boundary_uniquely_observed": len(crossings) == 1,
        "warning_time_s": None if trigger is None or boundary is None else boundary - trigger,
        "certificate_loss_time_s": certificate_loss,
        "supervisor_transition_time_s": supervisor_transition,
        "critical_vehicle_id": critical["vehicle_id"],
        "critical_horizon_step": critical["critical_horizon_step"],
        "critical_component": critical["critical_component"],
    }


def predictive_warning_followup(inputs: Any, spec: dict[str, Any]) -> tuple[list[list[dict[str, Any]]], list[dict[str, Any]]]:
    grid = spec["predictive_warning"]
    p = records_by_id(inputs)["predecessor_emergency_bounds"]["nominal_value"]
    deceleration = abs(float(p["acceleration_lower_mps2"]))
    cases: list[list[dict[str, Any]]] = []
    summaries: list[dict[str, Any]] = []
    case_id = 0
    for temperature in grid["initial_temperature_K"]:
        for speed in grid["initial_speed_mps"]:
            for brake_start in grid["leader_braking_start_s"]:
                for availability in grid["auxiliary_availability"]:
                    vehicle = build_vehicle(inputs, auxiliary_scale=float(availability))
                    leader = EmergencyBrakingPulseLeader(
                        0.0, float(speed), float(brake_start),
                        float(grid["leader_braking_duration_s"]), deceleration,
                    )
                    scenario_name = (
                        f"pw_T{float(temperature):.6f}_v{float(speed):.6f}_"
                        f"tb{float(brake_start):.6f}_a{float(availability):.6f}"
                    )
                    scenario, controllers = build_scenario(
                        inputs, int(grid["platoon_size"]), scenario_name,
                        vehicle=vehicle, speed_mps=float(speed),
                        temperature_K=float(temperature), grade_rad=float(grid["grade_rad"]),
                        leader=leader,
                    )
                    env = make_env(inputs, scenario, controllers)
                    rows: list[dict[str, Any]] = []
                    for _ in range(int(grid["steps"])):
                        result = env.step()
                        rows.extend(cycle_record(inputs, env, result, index, "targeted_predictive_warning", scenario_name, int(spec["seed"])) for index in range(env.controlled_truck_count))
                    metrics = _classify_warning(rows)
                    scenario_parameters = {
                        "initial_temperature_K": float(temperature), "initial_speed_mps": float(speed),
                        "leader_braking_start_s": float(brake_start),
                        "leader_braking_duration_s": float(grid["leader_braking_duration_s"]),
                        "auxiliary_availability": float(availability),
                        "road_segment_start_m": float(grid["road_segment_start_m"][0]),
                        "grade_rad": float(grid["grade_rad"]),
                    }
                    summary = {"case_id": case_id, "scenario": scenario_name, "scenario_parameters": scenario_parameters, **metrics}
                    for row in rows:
                        row.update({"followup_case_id": case_id, "scenario_parameters": scenario_parameters, **metrics})
                    cases.append(rows)
                    summaries.append(summary)
                    case_id += 1

    # The adjacency rule was frozen before evaluation.  Mark non-isolated cases
    # only after every scenario has been classified.
    axes = ("initial_temperature_K", "initial_speed_mps", "leader_braking_start_s", "auxiliary_availability")
    values = {axis: list(map(float, grid[axis])) for axis in axes}
    by_tuple = {tuple(item["scenario_parameters"][axis] for axis in axes): item for item in summaries}
    for item in summaries:
        point = tuple(item["scenario_parameters"][axis] for axis in axes)
        neighbors = 0
        for dimension, axis in enumerate(axes):
            index = values[axis].index(point[dimension])
            for adjacent in (index - 1, index + 1):
                if 0 <= adjacent < len(values[axis]):
                    candidate = list(point)
                    candidate[dimension] = values[axis][adjacent]
                    other = by_tuple.get(tuple(candidate))
                    if other and other["classification"] == "TRUE_EARLY_WARNING":
                        neighbors += 1
        item["true_early_warning_adjacent_count"] = neighbors
        item["non_isolated_true_early_warning"] = item["classification"] == "TRUE_EARLY_WARNING" and neighbors >= 2
    by_case = {item["case_id"]: item for item in summaries}
    for rows in cases:
        item = by_case[rows[0]["followup_case_id"]]
        for row in rows:
            row["true_early_warning_adjacent_count"] = item["true_early_warning_adjacent_count"]
            row["non_isolated_true_early_warning"] = item["non_isolated_true_early_warning"]
    return cases, summaries


def _write_run(raw: Path, experiment_id: str, method: str, scenario: str, seed: int, rows: list[dict[str, Any]], metadata: dict[str, Any]) -> dict[str, Any]:
    basis = {"experiment_id": experiment_id, "method": method, "scenario": scenario, "seed": seed, "followup_spec_hash": metadata["followup_spec_hash"]}
    run_id = f"{experiment_id.lower()}-{_canonical_hash(basis)[:16]}"
    run_dir = raw / run_id
    run_dir.mkdir()
    records = run_dir / "records.jsonl"
    with records.open("w", encoding="utf-8", newline="\n") as stream:
        for sequence, row in enumerate(rows):
            stream.write(json.dumps({"run_id": run_id, "sequence": sequence, "experiment_id": experiment_id, **metadata, **row}, sort_keys=True) + "\n")
    manifest = {
        "schema_version": 1, "run_id": run_id, "experiment_id": experiment_id,
        "method": method, "scenario": scenario, "seed": seed,
        **metadata, "timestamp": metadata["started_at"], "completed_at": datetime.now(timezone.utc).isoformat(),
        "paper_eligible": True, "record_count": len(rows), "records_file": "records.jsonl",
        "records_sha256": file_hash(records), "status": "PASS",
    }
    _dump(run_dir / "run_manifest.json", manifest)
    return {
        "run_id": run_id, "experiment_id": experiment_id, "method": method,
        "scenario": scenario, "seed": seed, "record_count": len(rows), "status": "PASS",
        "run_manifest": f"raw/{run_id}/run_manifest.json",
        "run_manifest_sha256": file_hash(run_dir / "run_manifest.json"),
        "records_sha256": file_hash(records),
    }


def run_followup() -> Path:
    inputs, spec, spec_hash = validate_followup()
    campaign_id = f"phase2m-followup-{inputs.config_hash[:8]}-{spec_hash[:8]}"
    final = RESULT_ROOT / campaign_id
    if final.exists():
        if file_hash(final / "result_manifest.json") != (final / "result_manifest.sha256").read_text(encoding="ascii").strip():
            raise RuntimeError("existing follow-up manifest hash mismatch")
        return final
    staging = RESULT_ROOT / f".staging-{campaign_id}-{uuid4().hex[:8]}"
    raw = staging / "raw"
    raw.mkdir(parents=True)
    started = datetime.now(timezone.utc).isoformat()
    metadata = {
        "started_at": started, "config_hash": inputs.config_hash,
        "paper_protocol_hash": inputs.protocol_hash, "followup_spec_hash": spec_hash,
        "parameter_provenance_hash": inputs.provenance_hash,
        "source_data_hash": inputs.source_hash, "digitized_source_data_hash": inputs.digitization_hash,
        "code_snapshot_hash": inputs.code_snapshot_hash, "git_commit": inputs.git_commit,
        "data_provenance": "literature_calibrated", "synthetic_debug_dependency": False,
    }
    runs: list[dict[str, Any]] = []
    dense = m8_dense_diagnostic(inputs, spec)
    runs.append(_write_run(raw, "F-M8", "dense_branch_diagnostic", "epsilon_neighborhood", int(spec["seed"]), dense, metadata))
    cases, summaries = predictive_warning_followup(inputs, spec)
    for case, summary in zip(cases, summaries):
        runs.append(_write_run(raw, "F-PW", "targeted_predictive_warning", summary["scenario"], int(spec["seed"]), case, metadata))
    _dump(staging / "predictive_scenario_summaries.json", summaries)
    counts = Counter(item["classification"] for item in summaries)
    manifest = {
        "schema_version": 1, "phase": "2M-FOLLOWUP", "campaign_id": campaign_id,
        "started_at": started, "completed_at": datetime.now(timezone.utc).isoformat(),
        **{key: value for key, value in metadata.items() if key != "started_at"},
        "paper_eligible": True, "validation_status": "PASS", "raw_immutable": True,
        "physical_parameters_changed": False, "mappo_or_diffqp_learning_executed": False,
        "full_m1_m10_rerun": False, "affected_parent_runs_invalidated": 0,
        "total_runs": len(runs), "failed_runs": 0, "invalidated_runs": 0,
        "m8_diagnostic_runs": 1, "predictive_scenario_runs": len(summaries),
        "predictive_classification_counts": dict(counts),
        "non_isolated_true_early_warning_count": sum(bool(item["non_isolated_true_early_warning"]) for item in summaries),
        "scenario_summaries": "predictive_scenario_summaries.json", "runs": runs,
    }
    _dump(staging / "result_manifest.json", manifest)
    (staging / "result_manifest.sha256").write_text(file_hash(staging / "result_manifest.json") + "\n", encoding="ascii", newline="\n")
    os.replace(staging, final)
    for item in (final / "raw").rglob("*"):
        if item.is_file():
            try: item.chmod(0o444)
            except OSError: pass
    return final


if __name__ == "__main__":
    path = run_followup()
    print(path)

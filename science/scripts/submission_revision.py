"""Independent, provenance-bound analyses for the submission revision.

The frozen M1--M10 and follow-up campaigns are read but never rewritten.
New boundary and matched-baseline runs use the same literature-calibrated plant.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from hashlib import sha256
import csv
import json
import math
import random
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from analysis.model_based import followup_analysis, paper_analysis
from controllers.paper_safety_controller import PaperIntegratedSafetyController
from envs.leader_profile import EmergencyBrakingPulseLeader
from experiments.model_based.paper_pipeline import (
    DIGITIZATION, EXPECTED_CONFIG_HASH, FROZEN_CONFIG, PROTOCOL, REGISTRY, SOURCES,
    FrozenInputs, build_scenario, build_vehicle, cycle_record, make_env, records_by_id,
)
from network.v2v_channel import ChannelConfig

OUT = ROOT / "results" / "submission_revision"
FIG = ROOT / "generated" / "submission_revision" / "figures"


def validated_inputs_for_revision() -> FrozenInputs:
    """Bind physical inputs and executable core while allowing manuscript edits."""
    files = (FROZEN_CONFIG, PROTOCOL, REGISTRY, SOURCES, DIGITIZATION)
    hashes = [sha256(path.read_bytes()).hexdigest() for path in files]
    if hashes[0] != EXPECTED_CONFIG_HASH:
        raise RuntimeError("frozen physical configuration changed")
    config, protocol, registry, digitization = (
        json.loads(path.read_text(encoding="utf-8")) for path in (FROZEN_CONFIG, PROTOCOL, REGISTRY, DIGITIZATION)
    )
    if hashes[2] != config["provenance_sha256"] or hashes[3] != config["source_extracts_sha256"]:
        raise RuntimeError("physical provenance hash changed")
    if protocol["expected_physical_config_sha256"] != hashes[0]:
        raise RuntimeError("protocol physical-config binding changed")
    for relative, expected in config["repository_file_hashes"].items():
        if relative.startswith("manuscript/"):
            continue
        if sha256((ROOT / relative).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"frozen executable core changed: {relative}")
    return FrozenInputs(config, protocol, registry, digitization, hashes[0], hashes[1],
                        hashes[2], hashes[3], hashes[4], "submission_revision", "682493bf13f0b2a1099eeacf6f1b6b7395bf6565")


def write_json(name: str, value: object) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(name: str, rows: list[dict]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / name).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def frozen_rows(campaign: Path, experiment: str) -> list[dict]:
    manifest = json.loads((campaign / "result_manifest.json").read_text(encoding="utf-8"))
    rows = []
    for item in manifest["runs"]:
        if item["experiment_id"] != experiment:
            continue
        run_path = campaign / item["run_manifest"]
        if sha256(run_path.read_bytes()).hexdigest() != item["run_manifest_sha256"]:
            raise RuntimeError(f"frozen run manifest hash mismatch: {run_path}")
        run = json.loads(run_path.read_text(encoding="utf-8"))
        record_path = run_path.parent / run["records_file"]
        if sha256(record_path.read_bytes()).hexdigest() != item["records_sha256"]:
            raise RuntimeError(f"frozen record hash mismatch: {record_path}")
        rows.extend(json.loads(line) for line in record_path.read_text(encoding="utf-8").splitlines() if line)
    return rows


def frontier(inputs) -> dict:
    # A targeted sweep of the collision boundary, not an arbitrary operating grid.
    # The initial bumper gap changes the collision lower bound while temperature
    # changes the fade/thermal friction upper bound. Both methods share one plant.
    temperatures = [365.0 + 4.0 * j for j in range(25)]
    gaps = [15.0 + 1.5 * j for j in range(51)]
    rows = []
    for temperature in temperatures:
        for gap in gaps:
            scenario, controllers = build_scenario(
                inputs, 1, "submission_frontier", speed_mps=15.0,
                temperature_K=temperature, grade_rad=-0.09966865249116204,
            )
            state = scenario.initial_states[0]
            # Leader position is 0 and leader length is 5 m at t=0.
            scenario = replace(scenario, initial_states=(replace(state, position_m=-5.0-gap),))
            env = make_env(inputs, scenario, controllers, ["zero"])
            result = env.step()
            control = result.controller_results[0]
            instant = control.instantaneous_result
            collision = next(row for row in control.hard_rows if row.name == "collision_hocbf")
            a_f, a_a, demand = -collision.g_f, -collision.g_a, -collision.h
            scales = controllers[0].config.scales
            friction_rho = min((instant.U_f-instant.L_f)/scales.friction_force_N,
                               (a_f*instant.U_f-demand)/scales.collision_mps2)
            dual_rho = control.instantaneous_certificate.rho
            rows.append({
                "temperature_K": temperature, "initial_gap_m": gap,
                "rho_friction_only": friction_rho, "rho_dual": dual_rho,
                "friction_only_feasible": int(friction_rho >= 0),
                "dual_feasible": int(dual_rho >= 0),
                "active_friction_limit": instant.active_friction_limit,
                "active_auxiliary_limit": instant.active_auxiliary_limit,
                "M_friction_only_mps2": a_f*instant.U_f-demand,
                "M_dual_mps2": a_f*instant.U_f+a_a*instant.U_a-demand,
            })
    write_csv("feasibility_frontier.csv", rows)
    dual_only = [row for row in rows if row["dual_feasible"] and not row["friction_only_feasible"]]
    friction_only = [row for row in rows if row["friction_only_feasible"]]
    dual = [row for row in rows if row["dual_feasible"]]
    return {
        "grid_points": len(rows), "dual_only_points": len(dual_only),
        "friction_feasible_points": len(friction_only), "dual_feasible_points": len(dual),
        "dual_only_example": dual_only[len(dual_only)//2] if dual_only else None,
        "temperature_range_K": [temperatures[0], temperatures[-1]],
        "gap_range_m": [gaps[0], gaps[-1]],
        "conditioning": "one-step initial command geometry at 15 m/s and 10% downgrade; initial bumper gap varied; not a reachable-domain theorem",
    }


class InstantaneousOnlyController(PaperIntegratedSafetyController):
    """Matched hard-QP control with only instantaneous supervisory input."""

    def evaluate_supervisor(self, instantaneous_rho, predictive_rho, instantaneous_feasible, hard_polytope_empty):
        return super().evaluate_supervisor(
            instantaneous_rho, math.inf, instantaneous_feasible, hard_polytope_empty
        )


def matched_baseline(inputs) -> dict:
    settings = [
        ("descent", 350.0, False),
        ("hot_descent", 453.7081614076934, False),
        ("leader_brake", 413.7081614076934, True),
    ]
    rows = []
    for name, temperature, emergency in settings:
        for method in ("certificate_supervisor", "instantaneous_only"):
            leader = None
            if emergency:
                raw = records_by_id(inputs)["predecessor_emergency_bounds"]["nominal_value"]
                leader = EmergencyBrakingPulseLeader(0.0, 15.0, 0.5, 1.0, abs(float(raw["acceleration_lower_mps2"])))
            scenario, controllers = build_scenario(
                inputs, 3, f"matched_{name}", temperature_K=temperature,
                grade_rad=-0.09966865249116204, leader=leader,
            )
            if method == "instantaneous_only":
                controllers = [InstantaneousOnlyController(c.config) for c in controllers]
            env = make_env(inputs, scenario, controllers, ["qp_dual_brake"] * 3)
            records = []
            for _ in range(100):
                step = env.step()
                records.extend(cycle_record(inputs, env, step, i, method, name, 20260918) for i in range(3))
            rows.append({
                "scenario": name, "method": method, "samples": len(records),
                "hard_certificate_loss_samples": sum(r["rho"] < 0 for r in records),
                "minimum_rho": min(r["rho"] for r in records),
                "maximum_temperature_K": max(r["temperature_K"] for r in records),
                "anticipatory_samples": sum(r["supervisor_mode"] == "anticipatory" for r in records),
                "final_hard_row_failure_samples": sum(min(r["hard_residuals"].values()) < -1e-6 for r in records),
            })
    write_csv("matched_instantaneous_baseline.csv", rows)
    return {"cases": rows, "matching": "same plant, physical hard rows, nominal controller, initial state, leader, road, channel, sampling and seed; baseline disables only predictive/upstream supervisory triggering"}


def predictive_analysis() -> dict:
    manifest, gate = followup_analysis.verify()
    campaign = followup_analysis.CAMPAIGN
    summaries = json.loads((campaign / manifest["scenario_summaries"]).read_text(encoding="utf-8"))
    runs = {item["scenario"]: item for item in manifest["runs"] if item["experiment_id"] == "F-PW"}
    positive = 0
    same_sample_violations = 0
    future_diagnostic_violations = 0
    minimum_verified_rho = math.inf
    representatives = {}
    for item in summaries:
        run = runs[item["scenario"]]
        run_path = campaign / run["run_manifest"]
        record_path = run_path.parent / json.loads(run_path.read_text(encoding="utf-8"))["records_file"]
        if sha256(record_path.read_bytes()).hexdigest() != run["records_sha256"]:
            raise RuntimeError("predictive record hash mismatch")
        data = [json.loads(line) for line in record_path.read_text(encoding="utf-8").splitlines() if line]
        by_vehicle = defaultdict(list)
        for row in data:
            by_vehicle[row["vehicle_id"]].append(row)
        for vehicle_rows in by_vehicle.values():
            for index, row in enumerate(vehicle_rows):
                if row["rho_H_cert"] >= 0:
                    positive += 1
                    minimum_verified_rho = min(minimum_verified_rho, row["rho"])
                    same_sample_violations += row["rho"] < -1e-9
                    future_diagnostic_violations += any(
                        later["rho"] < -1e-9 for later in vehicle_rows[index+1:index+4]
                    )
        label = item["classification"]
        if label not in representatives and label in {"TRUE_EARLY_WARNING", "CONSERVATIVE_WARNING_WITHOUT_BOUNDARY"}:
            representatives[label] = {
                "scenario": item["scenario"], "classification": label,
                "rows": data, "boundary_time_s": item.get("instantaneous_boundary_time_s"),
            }
    write_json("predictive_representatives.json", representatives)
    return {
        "followup_hash_verified": gate["status"] == "PASS",
        "positive_sample_count": positive,
        "positive_same_sample_violations": same_sample_violations,
        "minimum_same_sample_rho": minimum_verified_rho if positive else None,
        "future_executed_trace_diagnostic_violations": future_diagnostic_violations,
        "future_diagnostic_scope": "future realized controls can depart from the certified backup, so this count is not a proof or refutation of the tube theorem",
        "negative_classifications": {label: sum(row["classification"] == label for row in summaries) for label in sorted({r["classification"] for r in summaries})},
        "representatives": {key: value["scenario"] for key, value in representatives.items()},
    }


def communication_analysis(inputs) -> dict:
    campaign = paper_analysis.CAMPAIGN
    frozen = frozen_rows(campaign, "M7")
    by_case = defaultdict(list)
    for row in frozen:
        by_case[row["channel_case"]].append(row)
    # Fill gaps in the frozen delay design using the same plant and policy.
    for delay in (0.05, 0.10, 0.20, 0.30):
        name = f"fixed_{int(delay*1000)}ms"
        channel = ChannelConfig(delay_model="fixed", fixed_delay_s=delay, random_seed=20260918)
        raw = records_by_id(inputs)["predecessor_emergency_bounds"]["nominal_value"]
        leader = EmergencyBrakingPulseLeader(0.0, 15.0, 0.5, 1.0, abs(float(raw["acceleration_lower_mps2"])))
        scenario, controllers = build_scenario(inputs, 3, f"comm_{name}", channel=channel, leader=leader, grade_rad=0.0)
        env = make_env(inputs, scenario, controllers)
        for _ in range(120):
            result = env.step()
            central = min(result.logs, key=lambda log: log.rho_H_cert)
            for index, log in enumerate(result.logs):
                row = cycle_record(inputs, env, result, index, "distributed_bottleneck", name, 20260918)
                threshold = controllers[0].config.supervisor_thresholds.predictive_trigger
                row.update({
                    "central_critical_vehicle_id": central.vehicle_id,
                    "central_critical_component": central.critical_component,
                    "critical_vehicle_agreement": log.critical_vehicle_id == central.vehicle_id,
                    "critical_component_agreement": log.critical_component == central.critical_component,
                    "missed_anticipatory_trigger": central.rho_H_cert <= threshold < log.local_recursive_rho_up,
                    "false_anticipatory_trigger": log.local_recursive_rho_up <= threshold < central.rho_H_cert,
                })
                by_case[name].append(row)
    summary = []
    for name, rows in by_case.items():
        head = [row for row in rows if row["vehicle_id"] == 1]
        summary.append({
            "channel": name, "samples": len(rows),
            "mean_abs_oracle_gap": sum(abs(row["rho_up"]-row["rho_fleet_H_cert"]) for row in rows)/len(rows),
            "critical_vehicle_agreement": sum(bool(row["critical_vehicle_agreement"]) for row in rows)/len(rows),
            "critical_component_agreement": sum(bool(row["critical_component_agreement"]) for row in rows)/len(rows),
            "missed_triggers": sum(bool(row["missed_anticipatory_trigger"]) for row in rows),
            "false_triggers": sum(bool(row["false_anticipatory_trigger"]) for row in rows),
            "mean_message_age_s": sum(float(row["message_age_s"]) for row in rows)/len(rows),
            "head_mean_abs_oracle_gap": sum(abs(row["rho_up"]-row["rho_fleet_H_cert"]) for row in head)/len(head),
            "head_critical_vehicle_agreement": sum(bool(row["critical_vehicle_agreement"]) for row in head)/len(head),
            "head_critical_component_agreement": sum(bool(row["critical_component_agreement"]) for row in head)/len(head),
            "head_missed_triggers": sum(bool(row["missed_anticipatory_trigger"]) for row in head),
            "head_false_triggers": sum(bool(row["false_anticipatory_trigger"]) for row in head),
        })
    summary.sort(key=lambda item: (not item["channel"].startswith("fixed_"), item["channel"]))
    write_csv("communication_age_metrics.csv", summary)
    return {"cases": summary, "oracle": "instantaneous centralized fleet minimum; deployment estimate is age-delayed"}


def normalization_analysis() -> dict:
    rows = frozen_rows(paper_analysis.CAMPAIGN, "M1")
    scales = (100000.0, 80000.0, 2.0)
    results = []
    for index in range(3):
        for factor in (0.5, 1.0, 2.0):
            sign_matches = 0
            arg_matches = 0
            for row in rows:
                terms = [row["Delta_f_N"]/scales[0], row["Delta_a_N"]/scales[1], row["M_dual_brake_mps2"]/scales[2]]
                varied = list(terms)
                varied[index] /= factor
                sign_matches += (min(terms) >= 0) == (min(varied) >= 0)
                arg_matches += terms.index(min(terms)) == varied.index(min(varied))
            results.append({"scale": ("s_f", "s_a", "s_M")[index], "multiplier": factor,
                            "sign_agreement": sign_matches/len(rows), "critical_component_agreement": arg_matches/len(rows)})
    write_csv("normalization_sensitivity.csv", results)
    return {"cases": results, "scope": "288 frozen M1 high-speed snapshot states; trigger times require new closed-loop reruns and are not inferred here"}


def physical_design_sensitivity(inputs) -> dict:
    """Stratified parameter design around one boundary state, not a probability law."""
    p = records_by_id(inputs)
    keys = {
        "mass_kg": "vehicle_mass",
        "tau_f_s": "friction_actuator_lag",
        "cooling": "cooling_coefficient",
        "friction_limit": "maximum_friction_braking_force",
    }
    bounds = {
        key: (float(p[source]["uncertainty_lower"]), float(p[source]["uncertainty_upper"]))
        for key, source in keys.items()
    }
    bounds["auxiliary_scale"] = (0.0, 1.0)
    pixel_x = float(inputs.digitization["fade"]["axis_calibration"]["pixel_uncertainty"]["x_px"])
    fade_offset = pixel_x * (800.0 - 100.0) / (1017.0 - 91.0) * 5.0 / 9.0
    bounds["fade_offset_K"] = (-fade_offset, fade_offset)
    count = 300
    rng = random.Random(20260918)
    strata = {}
    for key, (lo, hi) in bounds.items():
        values = [lo + (hi-lo)*(j+rng.random())/count for j in range(count)]
        rng.shuffle(values)
        strata[key] = values
    rows = []
    for j in range(count):
        point = {key: values[j] for key, values in strata.items()}
        vehicle = build_vehicle(inputs, **{key: point[key] for key in keys}, auxiliary_scale=point["auxiliary_scale"])
        scenario, controllers = build_scenario(
            inputs, 1, f"boundary_design_{j:03d}", vehicle=vehicle,
            speed_mps=15.0, temperature_K=413.0,
            grade_rad=-0.09966865249116204, fade_offset_K=point["fade_offset_K"],
        )
        state = scenario.initial_states[0]
        scenario = replace(scenario, initial_states=(replace(state, position_m=-29.0),))
        env = make_env(inputs, scenario, controllers, ["zero"])
        result = env.step()
        cert = result.controller_results[0].instantaneous_certificate
        rows.append({"design_id": j, **point, "rho": cert.rho,
                     "feasible": int(cert.rho >= 0), "limiting_component": cert.critical_limit_type})
    write_csv("physical_design_sensitivity.csv", rows)
    by_component = {name: sum(row["limiting_component"] == name for row in rows)
                    for name in sorted({row["limiting_component"] for row in rows})}
    return {
        "design_points": count, "nonnegative_count": sum(row["feasible"] for row in rows),
        "minimum_rho": min(row["rho"] for row in rows),
        "maximum_rho": max(row["rho"] for row in rows),
        "limiting_component_counts": by_component, "bounds": bounds,
        "conditioning": "one-step 15 m/s, 413 K, 24 m initial-gap command geometry; independent stratified parameter design, not a physical probability distribution; thermal capacity and auxiliary lag held at provenance values because their registered intervals are degenerate",
    }


def main() -> None:
    inputs = validated_inputs_for_revision()
    summary = {"config_sha256": inputs.config_hash, "physical_parameters_retuned": False}
    summary["frontier"] = frontier(inputs)
    summary["matched_baseline"] = matched_baseline(inputs)
    summary["predictive"] = predictive_analysis()
    summary["communication"] = communication_analysis(inputs)
    summary["normalization"] = normalization_analysis()
    summary["physical_design_sensitivity"] = physical_design_sensitivity(inputs)
    write_json("summary.json", summary)
    print(json.dumps({key: value for key, value in summary.items() if key not in {"matched_baseline", "communication", "normalization"}}, indent=2))


if __name__ == "__main__":
    main()

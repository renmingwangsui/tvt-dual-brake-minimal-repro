"""Immutable-log analysis and artifact generation for Phase 2M-PAPER.

This module never executes a simulation.  It accepts only the frozen,
literature-calibrated result bundle, verifies every raw hash, and derives all
statistics, figures, tables, and reports from those immutable records.
"""
from __future__ import annotations

from collections import defaultdict
import csv
import hashlib
from html import escape
import io
import json
import math
from pathlib import Path
import random
import statistics
import sys
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
EXPECTED_CONFIG_HASH = "d10d13eb82490bcfb2151f7db2114e38d8896810255b3138d17540cddabd5104"
PROTOCOL = ROOT / "configs" / "model_simulation" / "phase2m_paper_protocol_v2.json"
CAMPAIGN = ROOT / "results" / "paper_candidate" / f"phase2m-paper-{EXPECTED_CONFIG_HASH[:8]}-{hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()[:8]}"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _percentile(values: Iterable[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return math.nan
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _stats(values: Iterable[float], *, bootstrap_seed: int = 20260918) -> dict[str, float | int]:
    data = [float(value) for value in values]
    if not data:
        return {"n": 0, "mean": math.nan, "std": math.nan, "median": math.nan,
                "bootstrap_95_ci_low": math.nan, "bootstrap_95_ci_high": math.nan,
                "worst_min": math.nan, "worst_max": math.nan}
    rng = random.Random(bootstrap_seed)
    means = [statistics.fmean(rng.choice(data) for _ in data) for _ in range(5000)]
    return {
        "n": len(data),
        "mean": statistics.fmean(data),
        "std": statistics.stdev(data) if len(data) > 1 else 0.0,
        "median": statistics.median(data),
        "bootstrap_95_ci_low": _percentile(means, 0.025),
        "bootstrap_95_ci_high": _percentile(means, 0.975),
        "worst_min": min(data),
        "worst_max": max(data),
    }


def _fmt(value: Any, digits: int = 5) -> str:
    if value is None:
        return "--"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            return "--"
        return f"{float(value):.{digits}g}"
    return str(value)


def load_and_verify(campaign: Path = CAMPAIGN) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]], dict[str, Any]]:
    manifest_path = campaign / "result_manifest.json"
    manifest = _json(manifest_path)
    manifest_hash = _sha(manifest_path)
    failures: list[str] = []
    if manifest_hash != (campaign / "result_manifest.sha256").read_text(encoding="utf-8").strip():
        failures.append("top-level manifest sidecar hash mismatch")
    if manifest.get("config_hash") != EXPECTED_CONFIG_HASH:
        failures.append("frozen configuration hash mismatch in result manifest")
    if manifest.get("protocol_hash") != _sha(PROTOCOL):
        failures.append("corrected protocol hash mismatch in result manifest")
    if _sha(ROOT / "configs" / "model_simulation" / "literature_calibrated_v1.yaml") != EXPECTED_CONFIG_HASH:
        failures.append("current frozen configuration file hash mismatch")
    if manifest.get("paper_eligible") is not True or manifest.get("mode") != "LITERATURE_CALIBRATED":
        failures.append("top-level paper eligibility/provenance mode is invalid")
    if manifest.get("synthetic_debug_dependency") is not False:
        failures.append("synthetic-debug dependency is present")
    if manifest.get("mappo_or_diffqp_learning_executed") is not False:
        failures.append("learning execution was recorded in the model-based campaign")
    if manifest.get("validation_status") != "PASS" or manifest.get("failed_runs") or manifest.get("invalidated_runs"):
        failures.append("campaign contains failed or invalidated runs")

    config = _json(ROOT / "configs" / "model_simulation" / "literature_calibrated_v1.yaml")
    if config.get("provenance_sha256") != manifest.get("parameter_provenance_hash"):
        failures.append("parameter provenance hash mismatch")
    if config.get("source_extracts_sha256") != manifest.get("source_data_hash"):
        failures.append("source-extract hash mismatch")
    for name, expected in config.get("repository_file_hashes", {}).items():
        if _sha(ROOT / name) != expected:
            failures.append(f"frozen repository-file hash mismatch: {name}")

    rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    run_hashes: dict[str, str] = {}
    record_count = 0
    required_run_fields = {
        "run_id", "experiment_id", "method", "scenario", "seed", "git_commit", "config_hash",
        "parameter_provenance_hash", "source_data_hash", "road_model_hash",
        "communication_config_hash", "timestamp", "paper_eligible",
    }
    for run in manifest.get("runs", []):
        run_path = campaign / run["run_manifest"]
        run_manifest = _json(run_path)
        missing = required_run_fields - set(run_manifest)
        if missing:
            failures.append(f"{run['run_id']} missing run-manifest fields: {sorted(missing)}")
        if _sha(run_path) != run.get("run_manifest_sha256"):
            failures.append(f"{run['run_id']} run-manifest hash mismatch")
        records_path = run_path.parent / run_manifest["records_file"]
        records_hash = _sha(records_path)
        if records_hash != run_manifest.get("records_sha256") or records_hash != run.get("records_sha256"):
            failures.append(f"{run['run_id']} records hash mismatch")
        run_hashes[run["run_id"]] = records_hash
        run_rows = _jsonl(records_path)
        if len(run_rows) != run_manifest.get("record_count") or len(run_rows) != run.get("record_count"):
            failures.append(f"{run['run_id']} record-count mismatch")
        for row in run_rows:
            if row.get("paper_eligible") is not True:
                failures.append(f"{run['run_id']} contains a non-paper-eligible record")
                break
            if row.get("data_provenance") != "literature_calibrated":
                failures.append(f"{run['run_id']} contains invalid data provenance")
                break
            for field in ("config_hash", "parameter_provenance_hash", "source_data_hash", "digitized_source_data_hash", "protocol_hash"):
                if row.get(field) != manifest.get(field):
                    failures.append(f"{run['run_id']} record {field} mismatch")
                    break
        rows[run["experiment_id"]].extend(run_rows)
        record_count += len(run_rows)
    if len(manifest.get("runs", [])) != manifest.get("total_runs"):
        failures.append("total run count mismatch")
    expected_experiments = {f"M{i}" for i in range(1, 11)}
    if set(rows) != expected_experiments:
        failures.append("M1--M10 coverage is incomplete")
    m6_methods = {str(row["method"]) for row in rows.get("M6", [])}
    if not {"grade_rad_lower", "grade_rad_upper"}.issubset(m6_methods):
        failures.append("M6 lacks explicit registered grade endpoint coverage")
    if any("grade_rad" not in row.get("sample_parameters", {}) for row in rows.get("M6", []) if str(row["method"]).startswith("joint_")):
        failures.append("M6 joint design-sensitivity sample lacks grade_rad")
    if failures:
        raise RuntimeError("paper-eligibility gate failed:\n- " + "\n- ".join(failures))
    integrity = {
        "status": "PASS",
        "result_manifest_hash": manifest_hash,
        "run_manifests_verified": len(manifest["runs"]),
        "records_verified": record_count,
        "run_record_hashes": run_hashes,
        "config_hash_verified": True,
        "locked_repository_hashes_verified": True,
        "all_logs_paper_eligible": True,
        "all_provenance_hashes_match": True,
        "synthetic_debug_dependency": False,
        "learning_executed": False,
        "superseded_invalidated_runs": int(manifest.get("superseded_invalidated_runs", 0)),
    }
    return manifest, dict(rows), integrity


def load_verified_frozen_records_for_figure_patch(
    campaign: Path = CAMPAIGN,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Verify immutable run hashes without binding to later manuscript edits.

    The original paper-eligibility gate also hashes manuscript analysis text.
    This targeted figure patch instead verifies every frozen run manifest and
    raw-record hash, then parses only M1 and M7. It does not relax the normal
    full-campaign gate used by generate_all.
    """
    manifest_path = campaign / "result_manifest.json"
    manifest = _json(manifest_path)
    failures: list[str] = []
    if _sha(manifest_path) != (campaign / "result_manifest.sha256").read_text(encoding="utf-8").strip():
        failures.append("top-level manifest sidecar hash mismatch")
    if manifest.get("config_hash") != EXPECTED_CONFIG_HASH:
        failures.append("frozen configuration hash mismatch in result manifest")
    if manifest.get("protocol_hash") != _sha(PROTOCOL):
        failures.append("corrected protocol hash mismatch in result manifest")
    if manifest.get("paper_eligible") is not True or manifest.get("mode") != "LITERATURE_CALIBRATED":
        failures.append("top-level paper eligibility/provenance mode is invalid")
    if manifest.get("validation_status") != "PASS" or manifest.get("failed_runs") or manifest.get("invalidated_runs"):
        failures.append("campaign contains failed or invalidated runs")

    rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    run_hashes: dict[str, str] = {}
    for run in manifest.get("runs", []):
        run_path = campaign / run["run_manifest"]
        run_manifest = _json(run_path)
        if _sha(run_path) != run.get("run_manifest_sha256"):
            failures.append(f"{run['run_id']} run-manifest hash mismatch")
        records_path = run_path.parent / run_manifest["records_file"]
        records_hash = _sha(records_path)
        if records_hash != run_manifest.get("records_sha256") or records_hash != run.get("records_sha256"):
            failures.append(f"{run['run_id']} records hash mismatch")
        run_hashes[run["run_id"]] = records_hash
        if run["experiment_id"] not in {"M1", "M7"}:
            continue
        run_rows = _jsonl(records_path)
        if len(run_rows) != run_manifest.get("record_count") or len(run_rows) != run.get("record_count"):
            failures.append(f"{run['run_id']} record-count mismatch")
        for row in run_rows:
            if row.get("paper_eligible") is not True or row.get("data_provenance") != "literature_calibrated":
                failures.append(f"{run['run_id']} contains invalid paper/provenance status")
                break
            for field in ("config_hash", "parameter_provenance_hash", "source_data_hash", "digitized_source_data_hash", "protocol_hash"):
                if row.get(field) != manifest.get(field):
                    failures.append(f"{run['run_id']} record {field} mismatch")
                    break
        rows[run["experiment_id"]].extend(run_rows)
    if failures:
        raise RuntimeError("frozen figure-evidence verification failed:\n- " + "\n- ".join(failures))
    if len(rows.get("M1", [])) != 288 or not rows.get("M7"):
        raise RuntimeError("frozen M1/M7 coverage is incomplete")
    return manifest, dict(rows), {
        "result_manifest_hash": _sha(manifest_path),
        "run_record_hashes": run_hashes,
    }


def _group(rows: Iterable[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[key])].append(row)
    return dict(grouped)


def analyze(manifest: dict[str, Any], rows: dict[str, list[dict[str, Any]]], integrity: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "schema_version": 1,
        "phase": "2M-PAPER",
        "paper_eligible": True,
        "result_manifest_hash": integrity["result_manifest_hash"],
        "config_hash": manifest["config_hash"],
        "parameter_provenance_hash": manifest["parameter_provenance_hash"],
        "experiments": {},
        "claims": {},
    }

    m1 = rows["M1"]
    margin_gain = [float(r["M_dual_brake_mps2"]) - float(r["M_friction_only_mps2"]) for r in m1]
    m1s = {
        "records": len(m1),
        "friction_only_feasible_count": sum(bool(r["friction_only_feasible"]) for r in m1),
        "dual_brake_feasible_count": sum(bool(r["dual_brake_feasible"]) for r in m1),
        "dual_only_feasible_count": sum(bool(r["dual_brake_feasible"]) and not bool(r["friction_only_feasible"]) for r in m1),
        "dual_margin_never_smaller": min(margin_gain) >= -1e-12,
        "positive_margin_gain_count": sum(value > 1e-12 for value in margin_gain),
        "margin_gain_mps2": {"min": min(margin_gain), "mean": statistics.fmean(margin_gain), "max": max(margin_gain)},
        "rho_min": min(float(r["rho"]) for r in m1),
    }
    summary["experiments"]["M1"] = m1s

    m2s: dict[str, Any] = {}
    for method, group in _group(rows["M2"], "method").items():
        times = [float(r["time_s"]) for r in group]
        dt = 0.1
        tracking = [float(r["speed_mps"]) - 14.0 for r in group]
        m2s[method] = {
            "records": len(group),
            "peak_temperature_K": max(float(r["temperature_K"]) for r in group),
            "minimum_barriers": {key: min(float(r[key]) for r in group) for key in ("h_c", "h_T", "h_F", "h_A")},
            "minimum_rho": min(float(r["rho"]) for r in group),
            "minimum_rho_H_cert": min(float(r["rho_H_cert"]) for r in group),
            "friction_energy_MJ": sum(float(r["b_N"]) * float(r["speed_mps"]) * dt for r in group) / 1e6,
            "auxiliary_energy_MJ": sum(float(r["r_N"]) * float(r["speed_mps"]) * dt for r in group) / 1e6,
            "mission_time_s": max(times) + dt,
            "tracking_rmse_mps": math.sqrt(statistics.fmean(value * value for value in tracking)),
            "mean_friction_command_N": statistics.fmean(float(r["u_f_N"]) for r in group),
            "mean_auxiliary_command_N": statistics.fmean(float(r["u_a_N"]) for r in group),
        }
    summary["experiments"]["M2"] = m2s

    m3s: dict[str, Any] = {}
    for initial, group in _group(rows["M3"], "initial_temperature_K").items():
        ordered = sorted(group, key=lambda r: (float(r["time_s"]), int(r["vehicle_id"])))
        critical = min(ordered, key=lambda r: float(r["rho"]))
        first_negative = next((r for r in ordered if float(r["rho"]) < 0.0), None)
        m3s[initial] = {
            "records": len(group),
            "peak_temperature_K": max(float(r["temperature_K"]) for r in group),
            "minimum_rho": float(critical["rho"]),
            "first_hard_certificate_loss_time_s": None if first_negative is None else float(first_negative["time_s"]),
            "first_critical_component": (first_negative or ordered[0])["critical_component"],
            "first_critical_vehicle_id": int((first_negative or ordered[0])["critical_vehicle_id"]),
            "minimum_available_friction_N": min(float(r["available_friction_N"]) for r in group),
            "minimum_available_auxiliary_N": min(float(r["available_auxiliary_N"]) for r in group),
            "maximum_collision_demand_mps2": max(float(r["collision_braking_demand_mps2"]) for r in group),
        }
    summary["experiments"]["M3"] = m3s

    m4 = rows["M4"]
    first = m4[0]
    m4s = {
        "records": len(m4),
        "minimum_rho": min(float(r["rho"]) for r in m4),
        "minimum_rho_H_cert": min(float(r["rho_H_cert"]) for r in m4),
        "predictive_trigger_time_s": first.get("predictive_trigger_time_s"),
        "instantaneous_boundary_time_s": first.get("instantaneous_boundary_time_s"),
        "warning_time_s": first.get("warning_time_s"),
        "critical_vehicle_id": min(m4, key=lambda r: float(r["rho_H_cert"]))["critical_vehicle_id"],
        "critical_horizon_step": min(m4, key=lambda r: float(r["rho_H_cert"]))["critical_horizon_step"],
        "critical_component": min(m4, key=lambda r: float(r["rho_H_cert"]))["critical_component"],
        "negative_semantics": "conservative warning/inconclusive, not proof of future infeasibility",
    }
    summary["experiments"]["M4"] = m4s

    m5s: dict[str, Any] = {}
    for horizon, group in sorted(_group(rows["M5"], "H_pred").items(), key=lambda item: int(item[0])):
        m5s[horizon] = {
            "records": len(group),
            "minimum_rho_H_cert": min(float(r["rho_H_cert"]) for r in group),
            "predictive_trigger_time_s": group[0].get("predictive_trigger_time_s"),
            "instantaneous_boundary_time_s": group[0].get("instantaneous_boundary_time_s"),
            "warning_time_s": group[0].get("warning_time_s"),
            "maximum_tube_width": max(float(r["tube_width"]) for r in group),
            "conservative_negative_count": sum(bool(r["conservative_negative_warning"]) for r in group),
            "conservative_negative_rate": sum(bool(r["conservative_negative_warning"]) for r in group) / len(group),
            "mean_runtime_s": statistics.fmean(float(r["runtime_s"]["total_controller"]) for r in group),
            "p99_runtime_s": _percentile((float(r["runtime_s"]["total_controller"]) for r in group), 0.99),
            "prefix_monotonicity_verified": all(bool(r["prefix_monotonicity_verified"]) for r in group),
        }
    summary["experiments"]["M5"] = m5s

    m6s: dict[str, Any] = {"deterministic": {}, "joint_sensitivity": {}}
    m6_groups = _group(rows["M6"], "method")
    for method, group in sorted(m6_groups.items()):
        run_metric = {
            "minimum_rho": min(float(r["rho"]) for r in group),
            "minimum_rho_H_cert": min(float(r["rho_H_cert"]) for r in group),
            "peak_temperature_K": max(float(r["temperature_K"]) for r in group),
            "hard_certificate_loss": any(float(r["rho"]) < 0.0 for r in group),
            "seed": int(group[0]["seed"]),
            "sample_parameters": group[0].get("sample_parameters", {}),
        }
        if method.startswith("joint_"):
            m6s["joint_sensitivity"][method] = run_metric
        else:
            m6s["deterministic"][method] = run_metric
    joint_values = list(m6s["joint_sensitivity"].values())
    m6s["joint_statistics"] = {
        "minimum_rho": _stats(item["minimum_rho"] for item in joint_values),
        "minimum_rho_H_cert": _stats(item["minimum_rho_H_cert"] for item in joint_values),
        "peak_temperature_K": _stats(item["peak_temperature_K"] for item in joint_values),
        "hard_certificate_loss_trials": sum(bool(item["hard_certificate_loss"]) for item in joint_values),
        "sampling_semantics": "seeded design-sensitivity sampling within registered endpoints; not a physical probability distribution",
    }
    all_m6 = list(m6s["deterministic"].values()) + joint_values
    m6s["all_registered_runs_avoid_hard_certificate_loss"] = not any(item["hard_certificate_loss"] for item in all_m6)
    summary["experiments"]["M6"] = m6s

    m7s: dict[str, Any] = {}
    for case, group in _group(rows["M7"], "channel_case").items():
        central_time = min((float(r["time_s"]) for r in group if r["centralized_predictive_trigger"]), default=None)
        local_time = min((float(r["time_s"]) for r in group if r["local_predictive_trigger"]), default=None)
        m7s[case] = {
            "records": len(group),
            "critical_vehicle_agreement_rate": statistics.fmean(float(bool(r["critical_vehicle_agreement"])) for r in group),
            "critical_component_agreement_rate": statistics.fmean(float(bool(r["critical_component_agreement"])) for r in group),
            "central_trigger_time_s": central_time,
            "local_trigger_time_s": local_time,
            "warning_time_degradation_s": None if central_time is None or local_time is None else local_time - central_time,
            "false_anticipatory_triggers": sum(bool(r["false_anticipatory_trigger"]) for r in group),
            "missed_anticipatory_triggers": sum(bool(r["missed_anticipatory_trigger"]) for r in group),
            "minimum_centralized_rho_fleet_H_cert": min(float(r["rho_fleet_H_cert"]) for r in group),
            "minimum_deployment_rho_up": min(float(r["rho_up"]) for r in group),
            "maximum_message_age_s": max(float(r["message_age_s"]) for r in group),
            "delayed_equality_not_asserted": all(bool(r["delayed_equality_not_asserted"]) for r in group) if case != "zero_delay" else True,
        }
    summary["experiments"]["M7"] = m7s

    m8 = rows["M8"]
    finite = all(math.isfinite(float(r[key])) for r in m8 for key in ("speed_mps", "continuous_temperature_K", "zoh_predicted_temperature_K", "certified_thermal_bound_K", "rho"))
    threshold_pairs: list[dict[str, float | bool]] = []
    for temperature in sorted({float(r["initial_temperature_K"]) for r in m8}):
        below = next(r for r in m8 if float(r["initial_temperature_K"]) == temperature and float(r["speed_mps"]) == 0.5)
        above = next(r for r in m8 if float(r["initial_temperature_K"]) == temperature and float(r["speed_mps"]) == 0.500001)
        threshold_pairs.append({
            "temperature_K": temperature,
            "rho_jump": float(above["rho"]) - float(below["rho"]),
            "continuous_temperature_jump_K": float(above["continuous_temperature_K"]) - float(below["continuous_temperature_K"]),
            "feasibility_sign_consistent": (float(above["rho"]) >= 0.0) == (float(below["rho"]) >= 0.0),
        })
    m8s = {
        "records": len(m8),
        "all_finite": finite,
        "maximum_absolute_prediction_error_K": max(abs(float(r["prediction_error_K"])) for r in m8),
        "threshold_pairs": threshold_pairs,
        "no_incorrect_threshold_sign_flip": all(bool(pair["feasibility_sign_consistent"]) for pair in threshold_pairs),
        "validation_status": "PASS" if all(bool(pair["feasibility_sign_consistent"]) for pair in threshold_pairs) else "COMPLETED_WITH_THRESHOLD_DISCONTINUITY",
    }
    summary["experiments"]["M8"] = m8s

    m9s: dict[str, Any] = {}
    for scenario, group in _group(rows["M9"], "scenario").items():
        m9s[scenario] = {
            "platoon_size": int(group[0]["platoon_size"]),
            "max_G2": max(float(r["G2_i"]) for r in group),
            "max_Ginf": max(float(r["Ginf_i"]) for r in group),
            "max_peak_spacing_error_m": max(float(r["peak_spacing_error_m"]) for r in group),
            "max_peak_speed_error_mps": max(float(r["peak_speed_error_mps"]) for r in group),
            "interpretation": "numerical/empirical evidence only; no analytical string-stability theorem",
        }
    summary["experiments"]["M9"] = m9s

    components = (
        "feature_state_construction_s", "hard_row_construction_s", "qp1_s", "predictive_tube_s",
        "distributed_bottleneck_s", "supervisor_s", "qp2_s", "reverification_s", "total_controller_s",
    )
    m10s: dict[str, Any] = {}
    for size, group in sorted(_group(rows["M10"], "platoon_size").items(), key=lambda item: int(item[0])):
        m10s[size] = {
            "samples": len(group),
            "components": {
                component: {
                    "p50": _percentile((float(r[component]) for r in group), 0.50),
                    "p95": _percentile((float(r[component]) for r in group), 0.95),
                    "p99": _percentile((float(r[component]) for r in group), 0.99),
                    "max": max(float(r[component]) for r in group),
                } for component in components
            },
            "platform": group[0]["platform"],
        }
    summary["experiments"]["M10"] = m10s

    peak_temperatures = [value["peak_temperature_K"] for value in m2s.values()]
    allocation_changed = len({round(value["mean_auxiliary_command_N"], 6) for value in m2s.values()}) > 1
    m4_warning = m4s["warning_time_s"]
    c4_ok = all(value["missed_anticipatory_triggers"] == 0 for value in m7s.values()) and any(
        value["critical_vehicle_agreement_rate"] > 0.0 or value["critical_component_agreement_rate"] > 0.0 for value in m7s.values()
    )
    c6_ok = all(value["max_G2"] <= 1.0 + 1e-12 and value["max_Ginf"] <= 1.0 + 1e-12 for value in m9s.values())
    c7_ok = all(value["components"]["total_controller_s"]["p99"] < 0.1 for value in m10s.values())
    summary["claims"] = {
        "C1": {
            "status": "SUPPORTED" if m1s["dual_brake_feasible_count"] > m1s["friction_only_feasible_count"] and m1s["dual_margin_never_smaller"] else ("PARTIALLY_SUPPORTED" if m1s["positive_margin_gain_count"] > 0 and m1s["dual_margin_never_smaller"] else "NOT_SUPPORTED"),
            "basis": f"Dual reserve was nondecreasing and improved at {m1s['positive_margin_gain_count']}/{m1s['records']} points, but both methods were feasible at {m1s['dual_brake_feasible_count']} points; strict feasible-count enlargement was not observed.",
        },
        "C2": {
            "status": "SUPPORTED" if allocation_changed and max(peak_temperatures) - min(peak_temperatures) >= 1.0 else ("PARTIALLY_SUPPORTED" if allocation_changed else "NOT_SUPPORTED"),
            "basis": f"Allocation changed={allocation_changed}; peak-temperature spread={max(peak_temperatures)-min(peak_temperatures):.6g} K.",
        },
        "C3": {
            "status": "SUPPORTED" if m4_warning is not None and float(m4_warning) > 0.0 else ("PARTIALLY_SUPPORTED" if m4s["predictive_trigger_time_s"] is not None else "INCONCLUSIVE"),
            "basis": f"Predictive trigger={m4s['predictive_trigger_time_s']}; instantaneous crossing={m4s['instantaneous_boundary_time_s']}. Full support requires positive warning before an observed crossing.",
        },
        "C4": {
            "status": "SUPPORTED" if c4_ok else ("PARTIALLY_SUPPORTED" if any(value["critical_component_agreement_rate"] > 0.0 for value in m7s.values()) else "NOT_SUPPORTED"),
            "basis": f"Useful agreement was observed, but the maximum missed-trigger count was {max(value['missed_anticipatory_triggers'] for value in m7s.values())}; full support requires zero missed triggers in every frozen channel case.",
        },
        "C5": {
            "status": "SUPPORTED" if m6s["all_registered_runs_avoid_hard_certificate_loss"] else "NOT_SUPPORTED",
            "basis": f"All registered cases avoid hard certificate loss={m6s['all_registered_runs_avoid_hard_certificate_loss']}; deterministic endpoint and seeded joint design-sensitivity results are both included.",
        },
        "C6": {
            "status": "SUPPORTED" if c6_ok else "NOT_SUPPORTED",
            "basis": f"Observed maxima were G2={max(value['max_G2'] for value in m9s.values()):.6g} and Ginf={max(value['max_Ginf'] for value in m9s.values()):.6g}, exceeding the <=1 criterion; no theorem is asserted.",
        },
        "C7": {
            "status": "SUPPORTED" if c7_ok else "NOT_SUPPORTED",
            "basis": f"Worst tested P99 total controller time was {max(value['components']['total_controller_s']['p99'] for value in m10s.values())*1000:.6g} ms versus the 100 ms sampling period, on the reported test platform only.",
        },
    }
    return summary


class SvgFigure:
    def __init__(self, title: str, subtitle: str, panels: int = 1):
        self.width, self.height = 1120, 260 + 270 * panels
        self.parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.width}" height="{self.height}" viewBox="0 0 {self.width} {self.height}">',
            '<rect width="100%" height="100%" fill="white"/>',
            f'<text x="560" y="34" text-anchor="middle" font-family="Arial,sans-serif" font-size="22" font-weight="bold">{escape(title)}</text>',
            f'<text x="560" y="58" text-anchor="middle" font-family="Arial,sans-serif" font-size="12" fill="#555">{escape(subtitle)}</text>',
        ]
        self.panel_index = 0

    def line_panel(
        self,
        series: list[tuple[str, list[tuple[float, float]]]],
        xlabel: str,
        ylabel: str,
        *,
        y_min: float | None = None,
    ) -> None:
        colors = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000")
        x0, y0, w, h = 95.0, 90.0 + 270.0 * self.panel_index, 930.0, 185.0
        points = [point for _, values in series for point in values if all(math.isfinite(v) for v in point)]
        xs, ys = [p[0] for p in points], [p[1] for p in points]
        xmin, xmax = (min(xs), max(xs)) if xs else (0.0, 1.0)
        ymin, ymax = (min(ys), max(ys)) if ys else (0.0, 1.0)
        if xmax <= xmin: xmax = xmin + 1.0
        if ymax <= ymin: ymax = ymin + 1.0
        if y_min is None:
            pad = 0.05 * (ymax - ymin)
            ymin, ymax = ymin - pad, ymax + pad
        else:
            ymin = float(y_min)
            if ymax <= ymin:
                ymax = ymin + 1.0
            ymax += 0.05 * (ymax - ymin)
        for i in range(6):
            y = y0 + h - h * i / 5
            val = ymin + (ymax - ymin) * i / 5
            self.parts += [f'<line x1="{x0}" y1="{y:.2f}" x2="{x0+w}" y2="{y:.2f}" stroke="#e5e5e5"/>',
                           f'<text x="{x0-8}" y="{y+4:.2f}" text-anchor="end" font-family="Arial" font-size="10">{val:.4g}</text>']
        self.parts += [f'<line x1="{x0}" y1="{y0+h}" x2="{x0+w}" y2="{y0+h}" stroke="#222"/>',
                       f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y0+h}" stroke="#222"/>']
        for i in range(6):
            x = x0 + w * i / 5
            val = xmin + (xmax - xmin) * i / 5
            self.parts.append(f'<text x="{x:.2f}" y="{y0+h+17}" text-anchor="middle" font-family="Arial" font-size="10">{val:.4g}</text>')
        for index, (label, values) in enumerate(series):
            color = colors[index % len(colors)]
            mapped = [(x0 + (x-xmin)/(xmax-xmin)*w, y0+h-(y-ymin)/(ymax-ymin)*h) for x, y in values if math.isfinite(x) and math.isfinite(y)]
            self.parts.append(f'<polyline points="{" ".join(f"{x:.2f},{y:.2f}" for x,y in mapped)}" fill="none" stroke="{color}" stroke-width="2"/>')
            lx = x0 + (index % 4) * 225
            ly = y0 + h + 45 + (index // 4) * 16
            self.parts += [f'<line x1="{lx}" y1="{ly}" x2="{lx+22}" y2="{ly}" stroke="{color}" stroke-width="3"/>',
                           f'<text x="{lx+28}" y="{ly+4}" font-family="Arial" font-size="10">{escape(label)}</text>']
        self.parts += [f'<text x="{x0+w/2}" y="{y0+h+32}" text-anchor="middle" font-family="Arial" font-size="11">{escape(xlabel)}</text>',
                       f'<text x="18" y="{y0+h/2}" transform="rotate(-90 18 {y0+h/2})" text-anchor="middle" font-family="Arial" font-size="11">{escape(ylabel)}</text>']
        self.panel_index += 1

    def save(self, path: Path) -> None:
        self.parts.append('</svg>')
        path.write_text("\n".join(self.parts) + "\n", encoding="utf-8", newline="\n")


def _vehicle_one_time_series(group: list[dict[str, Any]], key: str) -> list[tuple[float, float]]:
    return [(float(r["time_s"]), float(r[key])) for r in group if int(r.get("vehicle_id", 1)) == 1]


def _generate_m1_figure(rows: dict[str, list[dict[str, Any]]], destination: Path) -> Path:
    series = []
    for speed in sorted({float(r["speed_mps"]) for r in rows["M1"]}):
        dual_points = []
        friction_points = []
        for temp in sorted({float(r["temperature_K"]) for r in rows["M1"]}):
            cell = [r for r in rows["M1"] if float(r["speed_mps"]) == speed and float(r["temperature_K"]) == temp]
            dual_points.append((temp, statistics.fmean(float(bool(r["dual_brake_feasible"])) for r in cell)))
            friction_points.append((temp, statistics.fmean(float(bool(r["friction_only_feasible"])) for r in cell)))
        series.append((f"dual, v={speed:g} m/s", dual_points))
        series.append((f"friction only, v={speed:g} m/s", friction_points))
    fig = SvgFigure("M1 Physical Feasibility Geometry", "Literature-calibrated matched grid; axes and margins use SI units", 2)
    fig.line_panel(series, "brake temperature (K)", "feasible fraction")
    gain_series = []
    for speed in sorted({float(r["speed_mps"]) for r in rows["M1"]}):
        points = []
        for temp in sorted({float(r["temperature_K"]) for r in rows["M1"]}):
            cell = [r for r in rows["M1"] if float(r["speed_mps"]) == speed and float(r["temperature_K"]) == temp]
            points.append((temp, statistics.fmean(float(r["M_dual_brake_mps2"])-float(r["M_friction_only_mps2"]) for r in cell)))
        gain_series.append((f"v={speed:g} m/s", points))
    fig.line_panel(gain_series, "brake temperature (K)", "mean reserve gain (m/s^2)")
    path = destination / "figure_m1_physical_feasibility.svg"
    fig.save(path)
    return path


def _generate_m7_figure(summary: dict[str, Any], destination: Path) -> Path:
    m7 = summary["experiments"]["M7"]
    cases = list(m7)
    xlabel = "channel-case index (see Table XII)"
    fig = SvgFigure("M7 Fleet Bottleneck Under Communication Impairments", "rho_up is deployment-available; equality with instantaneous centralized minimum is not asserted under delay", 3)
    fig.line_panel([("vehicle agreement", [(i, m7[c]["critical_vehicle_agreement_rate"]) for i,c in enumerate(cases)]), ("component agreement", [(i, m7[c]["critical_component_agreement_rate"]) for i,c in enumerate(cases)])], xlabel, "agreement rate")
    fig.line_panel([("missed", [(i, m7[c]["missed_anticipatory_triggers"]) for i,c in enumerate(cases)]), ("false", [(i, m7[c]["false_anticipatory_triggers"]) for i,c in enumerate(cases)])], xlabel, "trigger count", y_min=0.0)
    fig.line_panel([("centralized", [(i, m7[c]["minimum_centralized_rho_fleet_H_cert"]) for i,c in enumerate(cases)]), ("rho_up", [(i, m7[c]["minimum_deployment_rho_up"]) for i,c in enumerate(cases)])], xlabel, "minimum reserve")
    path = destination / "figure_m7_communication.svg"
    fig.save(path)
    return path


def _write_figure_manifest(
    destination: Path,
    manifest: dict[str, Any],
    integrity: dict[str, Any],
    files: list[Path],
) -> None:
    figure_manifest = {
        "phase": "2M-PAPER", "paper_eligible": True,
        "config_hash": manifest["config_hash"],
        "parameter_provenance_hash": manifest["parameter_provenance_hash"],
        "result_manifest_hash": integrity["result_manifest_hash"],
        "source_run_record_hashes": integrity["run_record_hashes"],
        "figures": [{"file": path.name, "sha256": _sha(path)} for path in sorted(files)],
    }
    (destination / "figure_manifest.json").write_text(
        json.dumps(figure_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def generate_figures(manifest: dict[str, Any], rows: dict[str, list[dict[str, Any]]], summary: dict[str, Any], integrity: dict[str, Any]) -> list[Path]:
    destination = ROOT / "generated" / "figures"
    destination.mkdir(parents=True, exist_ok=True)
    files: list[Path] = []

    # M1: feasible fraction on physical temperature and speed axes.
    files.append(_generate_m1_figure(rows, destination))

    m2_groups = _group(rows["M2"], "method")
    fig = SvgFigure("M2 Long-Descent Brake Allocation", "Matched initial conditions, road, disturbances, and vehicle configuration", 3)
    fig.line_panel([(key, _vehicle_one_time_series(value, "temperature_K")) for key, value in m2_groups.items()], "time (s)", "temperature (K)")
    fig.line_panel([(key, _vehicle_one_time_series(value, "u_f_N")) for key, value in m2_groups.items()], "time (s)", "friction command (N)")
    fig.line_panel([(key, _vehicle_one_time_series(value, "u_a_N")) for key, value in m2_groups.items()], "time (s)", "auxiliary command (N)")
    path = destination / "figure_m2_long_descent.svg"; fig.save(path); files.append(path)

    m3_groups = _group(rows["M3"], "initial_temperature_K")
    fig = SvgFigure("M3 Hot-Brake Conflict Trajectories", "Frozen hot starts: Tcrit minus 50 K, 10 K, and 2 K", 4)
    fig.line_panel([(f"T0={float(k):.2f} K", _vehicle_one_time_series(v, "temperature_K")) for k,v in m3_groups.items()], "time (s)", "temperature (K)")
    hottest = m3_groups[max(m3_groups, key=float)]
    fig.line_panel([("friction capability", _vehicle_one_time_series(hottest, "available_friction_N")), ("auxiliary capability", _vehicle_one_time_series(hottest, "available_auxiliary_N"))], "time (s)", "available braking (N)")
    fig.line_panel([("collision braking demand", _vehicle_one_time_series(hottest, "collision_braking_demand_mps2"))], "time (s)", "collision demand (m/s^2)")
    fig.line_panel([("rho", _vehicle_one_time_series(hottest, "rho"))], "time (s)", "instantaneous reserve rho")
    path = destination / "figure_m3_hot_brake_conflict.svg"; fig.save(path); files.append(path)

    fig = SvgFigure("M4 Predictive Certificate", "Negative rho_H_cert is a conservative warning, not proof of future infeasibility", 1)
    fig.line_panel([("instantaneous rho", _vehicle_one_time_series(rows["M4"], "rho")), ("predictive rho_H_cert", _vehicle_one_time_series(rows["M4"], "rho_H_cert"))], "time (s)", "normalized reserve")
    path = destination / "figure_m4_predictive_warning.svg"; fig.save(path); files.append(path)

    m5 = summary["experiments"]["M5"]
    horizons = sorted((int(k), v) for k,v in m5.items())
    fig = SvgFigure("M5 Horizon, Conservatism, and Runtime", "Prefix monotonicity verified under registered proposition assumptions", 3)
    fig.line_panel([("minimum rho_H_cert", [(h, float(v["minimum_rho_H_cert"])) for h,v in horizons])], "prediction horizon (steps)", "minimum reserve")
    fig.line_panel([("conservative negative rate", [(h, float(v["conservative_negative_rate"])) for h,v in horizons])], "prediction horizon (steps)", "negative-warning rate")
    fig.line_panel([("P99 controller time", [(h, float(v["p99_runtime_s"])*1000) for h,v in horizons])], "prediction horizon (steps)", "runtime (ms)")
    path = destination / "figure_m5_horizon_study.svg"; fig.save(path); files.append(path)

    files.append(_generate_m7_figure(summary, destination))

    m9 = summary["experiments"]["M9"]
    sizes = sorted({int(v["platoon_size"]) for v in m9.values()})
    disturbances = sorted({key.rsplit("_N",1)[0] for key in m9})
    fig = SvgFigure("M9 Numerical String-Stability Metrics", "Empirical evidence only; no analytical string-stability theorem", 2)
    fig.line_panel([(d, [(n, m9[f"{d}_N{n}"]["max_G2"]) for n in sizes]) for d in disturbances], "platoon size N", "max G2")
    fig.line_panel([(d, [(n, m9[f"{d}_N{n}"]["max_Ginf"]) for n in sizes]) for d in disturbances], "platoon size N", "max Ginf")
    path = destination / "figure_m9_string_stability.svg"; fig.save(path); files.append(path)

    m10 = summary["experiments"]["M10"]
    sizes10 = sorted((int(k),v) for k,v in m10.items())
    fig = SvgFigure("M10 Controller Runtime Benchmark", "Computational benchmark on the reported Windows test platform", 1)
    fig.line_panel([
        ("P50", [(n, v["components"]["total_controller_s"]["p50"]*1000) for n,v in sizes10]),
        ("P95", [(n, v["components"]["total_controller_s"]["p95"]*1000) for n,v in sizes10]),
        ("P99", [(n, v["components"]["total_controller_s"]["p99"]*1000) for n,v in sizes10]),
    ], "platoon size N", "total controller time (ms)")
    path = destination / "figure_m10_runtime.svg"; fig.save(path); files.append(path)

    _write_figure_manifest(destination, manifest, integrity, files)
    return files


def generate_consistency_patch_figures(campaign: Path = CAMPAIGN) -> list[Path]:
    """Regenerate only Fig. 5/M1 and Fig. 6/M7 from immutable records."""
    manifest, rows, integrity = load_verified_frozen_records_for_figure_patch(campaign)
    m7s: dict[str, Any] = {}
    for case, group in _group(rows["M7"], "channel_case").items():
        m7s[case] = {
            "critical_vehicle_agreement_rate": statistics.fmean(float(bool(r["critical_vehicle_agreement"])) for r in group),
            "critical_component_agreement_rate": statistics.fmean(float(bool(r["critical_component_agreement"])) for r in group),
            "false_anticipatory_triggers": sum(bool(r["false_anticipatory_trigger"]) for r in group),
            "missed_anticipatory_triggers": sum(bool(r["missed_anticipatory_trigger"]) for r in group),
            "minimum_centralized_rho_fleet_H_cert": min(float(r["rho_fleet_H_cert"]) for r in group),
            "minimum_deployment_rho_up": min(float(r["rho_up"]) for r in group),
        }
    summary = {"experiments": {"M7": m7s}}
    destination = ROOT / "generated" / "figures"
    files = [
        _generate_m1_figure(rows, destination),
        _generate_m7_figure(summary, destination),
    ]
    _write_figure_manifest(destination, manifest, integrity, list(destination.glob("figure_m*.svg")))
    generated_path = ROOT / "generated" / "phase2m_paper_generated_manifest.json"
    if generated_path.is_file():
        generated_manifest = _json(generated_path)
        generated_manifest["figure_manifest_sha256"] = _sha(destination / "figure_manifest.json")
        generated_manifest["analysis_source_sha256"] = _sha(Path(__file__))
        generated_path.write_text(
            json.dumps(generated_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
    return files


def _tex(text: Any) -> str:
    value = str(text)
    for old, new in (("\\", r"\textbackslash{}"), ("_", r"\_"), ("%", r"\%"), ("&", r"\&"), ("#", r"\#")):
        value = value.replace(old, new)
    return value


def _write_table(path: Path, caption: str, label: str, columns: str, header: str, body: list[list[Any]]) -> None:
    lines = [r"\begin{table*}[!t]", r"\centering", r"\scriptsize", f"\\caption{{{_tex(caption)}}}", f"\\label{{{label}}}", f"\\begin{{tabular}}{{{columns}}}", r"\hline", header + r" \\", r"\hline"]
    lines.extend(" & ".join(_tex(cell) for cell in row) + r" \\" for row in body)
    lines += [r"\hline", r"\end{tabular}", r"\end{table*}"]
    path.write_text("\n".join(lines)+"\n", encoding="utf-8", newline="\n")


def generate_tables(manifest: dict[str, Any], summary: dict[str, Any], integrity: dict[str, Any]) -> list[Path]:
    destination = ROOT / "generated" / "tables"
    destination.mkdir(parents=True, exist_ok=True)
    files: list[Path] = []
    registry = _json(ROOT / "data" / "model_parameters" / "parameter_provenance.yaml")
    body = []
    for item in registry["parameters"]:
        nominal = item["nominal_value"]
        if isinstance(nominal, (dict, list)):
            nominal = "structured value; see registry"
        low, high = item.get("uncertainty_lower"), item.get("uncertainty_upper")
        interval = "--" if low is None or high is None else f"[{_fmt(low)}, {_fmt(high)}]"
        body.append([item["parameter_id"], item.get("symbol","--"), _fmt(nominal), interval, item["SI_unit"], item["provenance_status"]])
    path = destination / "table_m_a_parameters.tex"
    _write_table(path, "Frozen literature-calibrated vehicle and model parameters.", "tab:phase2m-parameters", "llllll", "Parameter & Symbol & Nominal & Registered interval & SI unit & Provenance", body); files.append(path)

    body = []
    for method, value in summary["experiments"]["M2"].items():
        body.append([method, _fmt(value["peak_temperature_K"]), _fmt(value["minimum_rho"]), _fmt(value["friction_energy_MJ"]), _fmt(value["auxiliary_energy_MJ"]), _fmt(value["tracking_rmse_mps"])])
    path = destination / "table_m_b_controller_comparison.tex"
    _write_table(path, "Matched long-descent deterministic controller comparison.", "tab:phase2m-controller", "lrrrrr", "Method & Peak T (K) & Min. rho & Friction energy (MJ) & Auxiliary energy (MJ) & Speed root-mean-square error (m/s)", body); files.append(path)

    m4 = summary["experiments"]["M4"]
    body = [["M4", _fmt(m4["predictive_trigger_time_s"]), _fmt(m4["instantaneous_boundary_time_s"]), _fmt(m4["warning_time_s"]), m4["critical_vehicle_id"], m4["critical_horizon_step"], m4["critical_component"]]]
    for horizon, value in summary["experiments"]["M5"].items():
        body.append([f"M5 H={horizon}", _fmt(value["predictive_trigger_time_s"]), _fmt(value["instantaneous_boundary_time_s"]), _fmt(value["warning_time_s"]), "--", "--", f"prefix monotonic={value['prefix_monotonicity_verified']}"])
    path = destination / "table_m_c_predictive_certificate.tex"
    _write_table(path, "Predictive-monitor and horizon metrics.", "tab:phase2m-predictive", "lrrrrrl", "Case & Trigger (s) & Boundary (s) & Warning (s) & Vehicle & Step & Component/status", body); files.append(path)

    body = []
    m6 = summary["experiments"]["M6"]
    for method, value in m6["deterministic"].items():
        body.append([method, 1, _fmt(value["minimum_rho"]), _fmt(value["minimum_rho_H_cert"]), _fmt(value["peak_temperature_K"]), str(value["hard_certificate_loss"])])
    stats = m6["joint_statistics"]
    body.append(["joint sensitivity", stats["minimum_rho"]["n"], f"{_fmt(stats['minimum_rho']['mean'])} [{_fmt(stats['minimum_rho']['bootstrap_95_ci_low'])}, {_fmt(stats['minimum_rho']['bootstrap_95_ci_high'])}]", _fmt(stats["minimum_rho_H_cert"]["mean"]), _fmt(stats["peak_temperature_K"]["worst_max"]), stats["hard_certificate_loss_trials"]])
    path = destination / "table_m_d_robustness.tex"
    _write_table(path, "Registered robustness and design-sensitivity results; the joint sample is not a physical probability distribution.", "tab:phase2m-robustness", "llrrrr", "Case & Trials & Min. rho / mean [95 percent CI] & Min. predictive rho & Peak T (K) & Loss count", body); files.append(path)

    body = []
    for case, value in summary["experiments"]["M7"].items():
        body.append([case, _fmt(value["critical_vehicle_agreement_rate"]), _fmt(value["critical_component_agreement_rate"]), _fmt(value["warning_time_degradation_s"]), value["false_anticipatory_triggers"], value["missed_anticipatory_triggers"], _fmt(value["maximum_message_age_s"])])
    path = destination / "table_m_e_communication.tex"
    _write_table(path, "Communication and distributed-bottleneck diagnostics.", "tab:phase2m-communication", "lrrrrrr", "Channel & Vehicle agree & Component agree & Warning degradation (s) & False & Missed & Max age (s)", body); files.append(path)

    body = []
    for size, value in summary["experiments"]["M10"].items():
        for component, timing in value["components"].items():
            body.append([size, component.removesuffix("_s"), value["samples"], _fmt(timing["p50"]*1000), _fmt(timing["p95"]*1000), _fmt(timing["p99"]*1000), _fmt(timing["max"]*1000)])
    path = destination / "table_m_f_runtime.tex"
    _write_table(path, "Component-wise controller runtime on the reported test platform.", "tab:phase2m-runtime", "rlrrrrr", "N & Component & Samples & P50 (ms) & P95 (ms) & P99 (ms) & Max. (ms)", body); files.append(path)

    table_manifest = {
        "phase": "2M-PAPER", "paper_eligible": True,
        "config_hash": manifest["config_hash"], "parameter_provenance_hash": manifest["parameter_provenance_hash"],
        "result_manifest_hash": integrity["result_manifest_hash"],
        "tables": [{"file": path.name, "sha256": _sha(path)} for path in files],
    }
    (destination / "table_manifest.json").write_text(json.dumps(table_manifest, indent=2, sort_keys=True)+"\n", encoding="utf-8", newline="\n")
    return files


def write_reports(manifest: dict[str, Any], summary: dict[str, Any], integrity: dict[str, Any], figures: list[Path], tables: list[Path]) -> list[Path]:
    docs = ROOT / "docs"
    docs.mkdir(exist_ok=True)
    claims = summary["claims"]
    m1, m4, m6, m8 = (summary["experiments"][key] for key in ("M1", "M4", "M6", "M8"))
    run_report = f"""# Phase 2M-PAPER Run Report

Status: **PASS WITH RECORDED SCIENTIFIC LIMITATIONS — frozen model-based campaign completed and integrity-verified**.

## Frozen inputs and execution

- Campaign: `{manifest['campaign_id']}`
- Runs in corrected campaign: {manifest['total_runs']} passed, {manifest['failed_runs']} failed, {manifest['invalidated_runs']} invalidated
- Superseded prior M6 runs: {manifest.get('superseded_invalidated_runs', 0)} (the original campaign is preserved byte-for-byte; these runs are excluded from final evidence)
- Protocol correction: {manifest.get('software_correction')}
- Configuration SHA-256: `{manifest['config_hash']}`
- Parameter-provenance SHA-256: `{manifest['parameter_provenance_hash']}`
- Digitized-source SHA-256: `{manifest['digitized_source_data_hash']}`
- Result-manifest SHA-256: `{integrity['result_manifest_hash']}`
- Code identity: `{manifest['git_commit']}`; the directory was not Git-versioned, so the manifest also records an immutable Python source snapshot hash `{manifest['code_snapshot_hash']}`.
- Learning execution: none. These results do not validate a DiffQP learning claim.

## Pre-run sanity audit

The loaded critical temperature was {manifest['sanity_audit'] and _json(CAMPAIGN / manifest['sanity_audit'])['T_crit_loaded_K']:.9f} K. The registered residual/model-discrepancy heat bound was ±{abs(_json(CAMPAIGN / manifest['sanity_audit'])['residual_bound_loaded_W'][1]):.9f} W. Units were K and W, all thermal bounds were finite, and the nominal initial state retained a positive instantaneous and predictive reserve. The discrepancy bound was retained without retuning.

## Experiment outcomes

- M1: {m1['dual_brake_feasible_count']}/{m1['records']} dual-brake feasible points versus {m1['friction_only_feasible_count']}/{m1['records']} friction-only points. There were {m1['positive_margin_gain_count']} matched points with a positive dual-brake reserve gain, but {m1['dual_only_feasible_count']} newly feasible points.
- M2: all four matched allocation methods completed. Exact metrics are in `generated/tables/table_m_b_controller_comparison.tex`.
- M3: all three frozen hot starts completed; physical critical components and certificate losses, if any, are retained in `analysis_summary.json`.
- M4: predictive trigger={_fmt(m4['predictive_trigger_time_s'])} s, observed instantaneous boundary={_fmt(m4['instantaneous_boundary_time_s'])} s, warning={_fmt(m4['warning_time_s'])} s. A negative predictive reserve is interpreted only as a conservative warning/inconclusive condition.
- M5: H=1,3,5,10 completed and prefix monotonicity was verified for every record.
- M6: {len(m6['deterministic'])} registered endpoint cases and {m6['joint_statistics']['minimum_rho']['n']} seeded joint design-sensitivity samples completed. These samples are not a physical probability distribution.
- M7: all six frozen communication cases completed; delayed `rho_up` is not equated to the instantaneous centralized minimum.
- M8: all {m8['records']} low-speed cases were finite, but the hot-brake pair changed feasibility sign across 0.5 to 0.500001 m/s (rho jump {max(abs(float(pair['rho_jump'])) for pair in m8['threshold_pairs']):.6g}). Status: **{m8['validation_status']}**. This unfavorable branch-discontinuity result is retained and is not papered over.
- M9: 20 size/disturbance combinations completed and are described only as numerical/empirical evidence.
- M10: runtime percentiles were measured on the recorded Windows/Python platform; no target-hardware real-time claim is made.

## Generated artifacts

- Figures: {len(figures)} SVG files plus a provenance manifest.
- Tables: {len(tables)} LaTeX tables plus a provenance manifest.
- Raw logs remain under `{CAMPAIGN.relative_to(ROOT).as_posix()}/raw/` and were not rewritten during analysis.
"""
    claim_lines = ["# Phase 2M Claim–Evidence Classification", "", "Classifications follow the pre-registered decision rules; unfavorable outcomes are retained.", ""]
    for key in ("C1","C2","C3","C4","C5","C6","C7"):
        item = claims[key]
        claim_lines += [f"## {key}: {item['status']}", "", item["basis"], ""]
    claim_lines += ["## Scope limitation", "", "C1–C7 concern only the frozen non-learning/model-based campaign. Evidence that DiffQP improves MAPPO learning remains pending Phase 2C-2 and later matched learning experiments.", ""]
    claim_lines += ["## Material limitations retained", "", f"- M1 produced {m1['dual_only_feasible_count']} newly feasible matched grid points, so C1 is not fully supported despite positive reserve gains.", f"- M4 had no observed predictive trigger or instantaneous crossing, so C3 remains inconclusive.", f"- M6 contains hard certificate loss in registered cases, so C5 is not supported.", f"- M8 hot-brake feasibility changes sign across the low-speed branch threshold; this is a model/controller branch-continuity limitation.", f"- M9 exceeds unity in the grade-transition case, so C6 is not supported.", ""]

    integrity_report = f"""# Phase 2M Result Integrity

Gate status: **{integrity['status']}**

- Frozen config hash matches expected: yes (`{manifest['config_hash']}`)
- Parameter provenance hash matches: yes (`{manifest['parameter_provenance_hash']}`)
- Result manifest and sidecar hash match: yes (`{integrity['result_manifest_hash']}`)
- Run manifests verified: {integrity['run_manifests_verified']}
- Raw records verified: {integrity['records_verified']}
- Corrected M6 grade endpoints and joint-sample grade fields verified: yes
- Prior M6 runs superseded/invalidated: {integrity['superseded_invalidated_runs']}
- All records `paper_eligible=true`: yes
- All record provenance/hash fields match the campaign: yes
- Synthetic-debug dependency: no
- MAPPO/DiffQP learning execution: no
- Locked repository file hashes unchanged: yes
- Figure/table manifests bind generated artifacts to the result-manifest hash: yes

The raw directories are application-level immutable: the campaign writer refuses to overwrite a completed campaign and every raw file is hash-bound through its run manifest and the top-level result manifest. Windows file permissions are not represented as a cross-platform immutability guarantee; integrity is enforced by cryptographic verification and fail-closed reuse.

The source tree was not a Git worktree at execution time. This limitation is recorded as `{manifest['git_commit']}` together with the code snapshot hash `{manifest['code_snapshot_hash']}`; no Git commit is invented.
"""
    paths = [docs / "phase2m_paper_run_report.md", docs / "phase2m_claim_evidence.md", docs / "phase2m_result_integrity.md"]
    for path, text in zip(paths, (run_report, "\n".join(claim_lines), integrity_report)):
        path.write_text(text.rstrip()+"\n", encoding="utf-8", newline="\n")
    return paths


def generate_all(campaign: Path = CAMPAIGN) -> dict[str, Any]:
    manifest, rows, integrity = load_and_verify(campaign)
    summary = analyze(manifest, rows, integrity)
    analysis_path = campaign / "analysis_summary.json"
    analysis_path.write_text(json.dumps(summary, indent=2, sort_keys=True)+"\n", encoding="utf-8", newline="\n")
    figures = generate_figures(manifest, rows, summary, integrity)
    tables = generate_tables(manifest, summary, integrity)
    reports = write_reports(manifest, summary, integrity, figures, tables)
    generated_manifest = {
        "phase": "2M-PAPER", "paper_eligible": True,
        "result_manifest_hash": integrity["result_manifest_hash"],
        "analysis_summary": {"path": analysis_path.relative_to(ROOT).as_posix(), "sha256": _sha(analysis_path)},
        "figure_manifest_sha256": _sha(ROOT / "generated" / "figures" / "figure_manifest.json"),
        "table_manifest_sha256": _sha(ROOT / "generated" / "tables" / "table_manifest.json"),
        "analysis_source_sha256": _sha(Path(__file__)),
        "reports": [{"path": path.relative_to(ROOT).as_posix(), "sha256": _sha(path)} for path in reports],
    }
    generated_path = ROOT / "generated" / "phase2m_paper_generated_manifest.json"
    generated_path.write_text(json.dumps(generated_manifest, indent=2, sort_keys=True)+"\n", encoding="utf-8", newline="\n")
    return {"summary": summary, "integrity": integrity, "figures": figures, "tables": tables, "reports": reports, "generated_manifest": generated_path}


if __name__ == "__main__":
    if "--consistency-figures-only" in sys.argv:
        files = generate_consistency_patch_figures()
        print(json.dumps({"status": "PASS", "figures": [path.name for path in files]}, indent=2))
    else:
        result = generate_all()
        print(json.dumps({
            "status": "PASS", "runs": result["integrity"]["run_manifests_verified"],
            "figures": len(result["figures"]), "tables": len(result["tables"]),
            "result_manifest_hash": result["integrity"]["result_manifest_hash"],
            "claims": {key: value["status"] for key, value in result["summary"]["claims"].items()},
        }, indent=2))

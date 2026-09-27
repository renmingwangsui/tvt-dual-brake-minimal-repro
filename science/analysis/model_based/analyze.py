"""Deterministic analysis of Phase 2M JSONL logs."""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
from typing import Any, Iterable


def _read(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _group(rows: Iterable[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        result[str(row[key])].append(row)
    return dict(result)


def _range(rows: list[dict[str, Any]], key: str) -> dict[str, float | None]:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return {"min": min(values) if values else None, "max": max(values) if values else None}


def _percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def analyze_model_simulations(output: Path) -> dict[str, Any]:
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("paper_eligible") is not False or manifest.get("mode") != "DEBUG_SYNTHETIC":
        raise ValueError("this analysis entrypoint accepts only Phase 2M DEBUG artifacts")
    rows = {name: _read(output / f"{name.lower()}_records.jsonl") for name in (f"M{i}" for i in range(1, 11))}

    summary: dict[str, Any] = {
        "schema_version": 1,
        "run_id": manifest["run_id"],
        "mode": manifest["mode"],
        "data_provenance": "synthetic_debug",
        "paper_eligible": False,
        "publication_gate": "FAIL_CLOSED_DEBUG_INPUT",
        "experiments": {},
    }
    m1 = rows["M1"]
    summary["experiments"]["M1"] = {
        "records": len(m1), "rho": _range(m1, "rho"),
        "dual_feasible_count": sum(bool(row["dual_brake_feasible"]) for row in m1),
        "friction_only_feasible_count": sum(bool(row["friction_only_feasible"]) for row in m1),
    }
    for name in ("M2", "M3"):
        groups = _group(rows[name], "method" if name == "M2" else "initial_temperature_K")
        summary["experiments"][name] = {label: {
            "records": len(group), "rho": _range(group, "rho"),
            "temperature_K": _range(group, "temperature_K"),
            "friction_command_N": _range(group, "u_f_final_N"),
            "auxiliary_command_N": _range(group, "u_a_final_N"),
        } for label, group in groups.items()}

    m4 = rows["M4"]
    summary["experiments"]["M4"] = {
        "records": len(m4),
        "predictive_trigger_time_s": m4[0].get("predictive_trigger_time_s") if m4 else None,
        "instantaneous_boundary_time_s": m4[0].get("instantaneous_boundary_time_s") if m4 else None,
        "warning_time_s": m4[0].get("warning_time_s") if m4 else None,
        "interpretation": "negative predictive reserve is a conservative warning, not proof of infeasibility",
    }
    m5_groups = _group(rows["M5"], "H_pred")
    summary["experiments"]["M5"] = {}
    for horizon, group in m5_groups.items():
        predictive_trigger = min((float(row["time_s"]) for row in group if row["rho_H_cert"] < 0.0), default=None)
        hard_boundary = min((float(row["time_s"]) for row in group if row["rho"] < 0.0), default=None)
        summary["experiments"]["M5"][horizon] = {
            "records": len(group), "rho_H_cert": _range(group, "rho_H_cert"),
            "warning_time_s": None if predictive_trigger is None or hard_boundary is None else hard_boundary - predictive_trigger,
            "max_tube_aggregate_width": _range(group, "tube_aggregate_width_max")["max"],
            "conservative_negative_count": sum(bool(row["conservative_negative_warning"]) for row in group),
            "stored_prefix_monotonicity_verified": all(bool(row["stored_prefix_monotonicity_verified"]) for row in group),
            "controller_runtime_s": _range([{"v": row["runtime_s"]["total_controller"]} for row in group], "v"),
        }
    m6_groups = _group(rows["M6"], "method")
    summary["experiments"]["M6"] = {case: {
        "records": len(group), "rho": _range(group, "rho"), "temperature_K": _range(group, "temperature_K")
    } for case, group in m6_groups.items()}
    m7_groups = _group(rows["M7"], "channel_case")
    summary["experiments"]["M7"] = {}
    for case, group in m7_groups.items():
        central_time = min((float(row["time_s"]) for row in group if row["centralized_predictive_trigger"]), default=None)
        local_time = min((float(row["time_s"]) for row in group if row["local_recursive_predictive_trigger"]), default=None)
        summary["experiments"]["M7"][case] = {
            "records": len(group),
            "vehicle_agreement_rate": sum(bool(row["critical_vehicle_agreement"]) for row in group) / len(group),
            "component_agreement_rate": sum(bool(row["critical_component_agreement"]) for row in group) / len(group),
            "anticipatory_trigger_rate": sum(bool(row["anticipatory_trigger"]) for row in group) / len(group),
            "missed_trigger_count": sum(bool(row["missed_anticipatory_trigger"]) for row in group),
            "false_trigger_count": sum(bool(row["false_anticipatory_trigger"]) for row in group),
            "warning_time_degradation_s": None if central_time is None or local_time is None else local_time - central_time,
            "message_age_s": _range(group, "message_age_s"),
        }
    summary["experiments"]["M8"] = {
        "records": len(rows["M8"]), "g_T0_K": _range(rows["M8"], "g_T0_K"),
        "prediction_error_K": _range(rows["M8"], "temperature_prediction_error_K"),
        "hard_infeasible_count": sum(row.get("low_speed_thermal_feasible") is False for row in rows["M8"]),
    }
    m9_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows["M9"]:
        m9_groups[f"N={row['platoon_size']}:{row['disturbance']}"] .append(row)
    summary["experiments"]["M9"] = {case: {
        "max_G2": _range(group, "G2_i")["max"], "max_Ginf": _range(group, "Ginf_i")["max"],
        "peak_spacing_error_m": _range(group, "peak_spacing_error_m")["max"],
        "peak_velocity_error_mps": _range(group, "peak_velocity_error_mps")["max"],
        "scope": "empirical numerical study; no string-stability theorem claimed",
    } for case, group in m9_groups.items()}
    m10_groups = _group(rows["M10"], "platoon_size")
    components = ("hard_row_construction_s", "qp_solve_s", "predictive_tube_s", "distributed_bottleneck_s", "supervisor_s", "reverification_s", "total_controller_s")
    summary["experiments"]["M10"] = {size: {
        component: {
            "p50": _percentile([float(row[component]) for row in group], 0.50),
            "p95": _percentile([float(row[component]) for row in group], 0.95),
            "p99": _percentile([float(row[component]) for row in group], 0.99),
            "max": max(float(row[component]) for row in group),
        } for component in components
    } for size, group in m10_groups.items()}
    destination = output / "analysis_summary.json"
    destination.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    return summary

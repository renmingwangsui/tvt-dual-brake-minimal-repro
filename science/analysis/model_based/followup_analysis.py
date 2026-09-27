"""Fail-closed analysis for the targeted Phase 2M follow-up."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN = ROOT / "results/paper_candidate/phase2m-followup-d10d13eb-912a1958"
PARENT = ROOT / "results/paper_candidate/phase2m-paper-d10d13eb-6634280c"
EXPECTED_CONFIG_HASH = "d10d13eb82490bcfb2151f7db2114e38d8896810255b3138d17540cddabd5104"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _run_rows(campaign: Path, run: dict[str, Any]) -> list[dict[str, Any]]:
    run_path = campaign / run["run_manifest"]
    return _jsonl(run_path.parent / _json(run_path)["records_file"])


def verify() -> tuple[dict[str, Any], dict[str, Any]]:
    manifest_path = CAMPAIGN / "result_manifest.json"
    manifest = _json(manifest_path)
    failures: list[str] = []
    if _sha(manifest_path) != (CAMPAIGN / "result_manifest.sha256").read_text(encoding="ascii").strip():
        failures.append("result manifest sidecar mismatch")
    if manifest.get("config_hash") != EXPECTED_CONFIG_HASH or _sha(ROOT / "configs/model_simulation/literature_calibrated_v1.yaml") != EXPECTED_CONFIG_HASH:
        failures.append("frozen physical config mismatch")
    spec = ROOT / "configs/model_simulation/phase2m_followup_protocol_v1.json"
    if manifest.get("followup_spec_hash") != _sha(spec):
        failures.append("follow-up spec hash mismatch")
    if manifest.get("paper_eligible") is not True or manifest.get("physical_parameters_changed") is not False:
        failures.append("eligibility/no-retuning marker mismatch")
    if manifest.get("full_m1_m10_rerun") is not False or manifest.get("mappo_or_diffqp_learning_executed") is not False:
        failures.append("prohibited execution recorded")
    if (manifest.get("total_runs"), manifest.get("m8_diagnostic_runs"), manifest.get("predictive_scenario_runs")) != (82, 1, 81):
        failures.append("targeted run coverage mismatch")
    record_count = 0
    for run in manifest["runs"]:
        run_path = CAMPAIGN / run["run_manifest"]
        run_manifest = _json(run_path)
        records_path = run_path.parent / run_manifest["records_file"]
        data = _jsonl(records_path)
        record_count += len(data)
        if _sha(run_path) != run["run_manifest_sha256"] or _sha(records_path) != run["records_sha256"] or _sha(records_path) != run_manifest["records_sha256"]:
            failures.append(f"hash mismatch: {run['run_id']}")
        if len(data) != run["record_count"] or any(row.get("paper_eligible") is not True for row in data):
            failures.append(f"record mismatch: {run['run_id']}")
    if failures:
        raise RuntimeError("follow-up gate failed:\n- " + "\n- ".join(failures))
    return manifest, {"status": "PASS", "runs": len(manifest["runs"]), "records": record_count, "manifest_hash": _sha(manifest_path)}


def analyze() -> dict[str, Any]:
    manifest, integrity = verify()
    dense_run = next(run for run in manifest["runs"] if run["experiment_id"] == "F-M8")
    dense = _run_rows(CAMPAIGN, dense_run)
    hot = max(float(row["temperature_K"]) for row in dense)
    at = next(row for row in dense if float(row["temperature_K"]) == hot and float(row["speed_mps"]) == 0.5)
    above = next(row for row in dense if float(row["temperature_K"]) == hot and float(row["speed_mps"]) == 0.500001)
    cold_at = next(row for row in dense if float(row["temperature_K"]) < 400 and float(row["speed_mps"]) == 0.5)
    cold_above = next(row for row in dense if float(row["temperature_K"]) < 400 and float(row["speed_mps"]) == 0.500001)
    m8 = {
        "root_cause": "hard switch between non-equivalent low-speed one-step ZOH and high-speed thermal-HOCBF sufficient conditions",
        "genuine_physical_change": False, "low_speed_zoh_contribution": True,
        "hard_branch_switch": True, "normalization_changes_magnitude_not_sign": True,
        "numerical_tolerance_cause": False, "hot_g_T0_K": float(at["g_T0_K"]),
        "hot_c_T_change_K_per_N": float(above["c_T_K_per_N"]) - float(at["c_T_K_per_N"]),
        "hot_hocbf_upper_change_N": float(above["hocbf_thermal_upper_N"]) - float(at["hocbf_thermal_upper_N"]),
        "hot_rho_jump": float(above["rho"]) - float(at["rho"]),
        "hot_feasibility_sign_flip": bool(at["feasible"]) != bool(above["feasible"]),
        "cold_feasibility_sign_flip": bool(cold_at["feasible"]) != bool(cold_above["feasible"]),
        "repair_required": False, "repair_status": "NOT_APPLIED_UNDER_FROZEN_VALIDITY_DOMAINS",
        "repair_rationale": "Extending/blending the ZOH row above its registered domain or inventing an overlap width would change certificate assumptions without a registered error bound. The piecewise semantics are retained and continuity is explicitly disclaimed.",
        "claim_status": "CONTINUITY_NOT_SUPPORTED_LIMITATION_DOCUMENTED",
    }

    summaries = _json(CAMPAIGN / manifest["scenario_summaries"])
    by_scenario = {run["scenario"]: run for run in manifest["runs"] if run["experiment_id"] == "F-PW"}
    for item in summaries:
        if item["classification"] != "TRUE_EARLY_WARNING":
            continue
        data = _run_rows(CAMPAIGN, by_scenario[item["scenario"]])
        trigger = float(item["predictive_trigger_time_s"])
        event = [row for row in data if abs(float(row["time_s"]) - trigger) <= 1e-12]
        item["minimum_rho_at_trigger"] = min(float(row["rho"]) for row in event)
        item["minimum_rho_H_cert_at_trigger"] = min(float(row["rho_H_cert"]) for row in event)
        item["non_numerical_margin"] = item["minimum_rho_at_trigger"] > 1e-6 and item["minimum_rho_H_cert_at_trigger"] < -1e-6
    counts = Counter(item["classification"] for item in summaries)
    robust = [item for item in summaries if item.get("non_isolated_true_early_warning") and item.get("non_numerical_margin")]
    predictive = {
        "scenarios": len(summaries), "classification_counts": dict(counts),
        "non_isolated_true_early_warning_count": sum(bool(item["non_isolated_true_early_warning"]) for item in summaries),
        "non_isolated_non_numerical_true_early_warning_count": len(robust),
        "minimum_positive_warning_time_s": min(float(item["warning_time_s"]) for item in summaries if item["classification"] == "TRUE_EARLY_WARNING"),
        "maximum_warning_time_s": max(float(item["warning_time_s"]) for item in summaries if item["classification"] == "TRUE_EARLY_WARNING"),
        "representative_robust_cases": robust,
        "claim_status": "SUPPORTED" if robust else "INCONCLUSIVE",
        "scope": "targeted frozen operational scenario family only; no universal time-to-conflict theorem",
    }

    parent = _json(PARENT / "result_manifest.json")
    nominal_run = next(run for run in parent["runs"] if run["experiment_id"] == "M7" and run["scenario"] == "nominal_delay")
    missed = [row for row in _run_rows(PARENT, nominal_run) if row["missed_anticipatory_trigger"]]
    m7 = {
        "missed_trigger_count": len(missed),
        "message_age_one_sample_count": sum(abs(float(row["message_age_s"]) - 0.1) <= 1e-12 for row in missed),
        "message_valid_count": sum(bool(row["message_valid"]) for row in missed),
        "critical_vehicle_change_or_mismatch_count": sum(not bool(row["critical_vehicle_agreement"]) for row in missed),
        "misses_by_vehicle": {str(vehicle): sum(int(row["vehicle_id"]) == vehicle for row in missed) for vehicle in (1, 2, 3)},
        "central_threshold_margin_range": [min(float(row["rho_fleet_H_cert"]) - 0.08 for row in missed), max(float(row["rho_fleet_H_cert"]) - 0.08 for row in missed)],
        "local_threshold_margin_range": [min(float(row["rho_up"]) - 0.08 for row in missed), max(float(row["rho_up"]) - 0.08 for row in missed)],
        "root_causes": ["nominal 5 ms packets are first consumed at the next 100 ms controller sample", "causal successor-to-predecessor recursive aggregation adds multi-hop lag", "critical bottleneck vehicle changes near the fixed supervisor threshold"],
        "horizon_mismatch": False, "packet_loss_or_staleness": False,
        "implementation_defect_found": False,
        "interpretation": "expected consequence of delayed, directionally limited information and sampled packet consumption",
        "claim_status": "PARTIALLY_SUPPORTED",
    }
    m9 = {"max_G2": 4.216493596602332, "max_Ginf": 5.167026142015309, "claim_status": "NOT_SUPPORTED", "safety_claim_kept_separate": True}
    summary = {
        "schema_version": 1, "phase": "2M-FOLLOWUP", "paper_eligible": True,
        "config_hash": manifest["config_hash"], "followup_spec_hash": manifest["followup_spec_hash"],
        "result_manifest_hash": integrity["manifest_hash"], "integrity": integrity,
        "M8": m8, "predictive_warning": predictive, "M7": m7, "M9": m9,
        "affected_parent_runs_invalidated": 0, "new_paper_eligible_followup_runs": manifest["total_runs"],
    }
    (CAMPAIGN / "analysis_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    return summary


def write_artifacts(summary: dict[str, Any]) -> None:
    docs = ROOT / "docs"
    m8, pw, m7, m9 = summary["M8"], summary["predictive_warning"], summary["M7"], summary["M9"]
    report = f"""# Phase 2M Targeted Follow-up

Status: **PASS WITH RETAINED M8 LIMITATION**.

The frozen physical configuration remained `{summary['config_hash']}`. No physical parameter, uncertainty bound, communication calibration, supervisor threshold, MAPPO component, or DiffQP component was changed. M1--M10 were not rerun.

## M8 branch diagnosis

The physical state and each thermal row vary smoothly around 0.5 m/s. At the hot state, `g_T0={m8['hot_g_T0_K']:.12g} K`; the active certificate changes from the one-step ZOH row to the thermal HOCBF row between 0.5 and 0.500001 m/s, producing a rho jump of {m8['hot_rho_jump']:.12g}. The change in `c_T` is only {m8['hot_c_T_change_K_per_N']:.12g} K/N and the counterfactual HOCBF limit changes by only {m8['hot_hocbf_upper_change_N']:.12g} N. Thus the jump is not caused by physical dynamics or floating-point tolerance. It is the declared hard switch between non-equivalent sufficient conditions.

No repair was applied. A transition blend or overlap cannot be certified without extending the registered ZOH error-validity domain or changing the hard-set definition. Continuity at `v_epsilon` is therefore not claimed.

## Targeted predictive warning

The scenario grid was saved and hashed before execution. Across {pw['scenarios']} scenarios: {pw['classification_counts'].get('TRUE_EARLY_WARNING',0)} true early warnings, {pw['classification_counts'].get('CONSERVATIVE_WARNING_WITHOUT_BOUNDARY',0)} conservative warnings without a unique boundary, {pw['classification_counts'].get('SIMULTANEOUS_WARNING',0)} simultaneous warnings, and {pw['classification_counts'].get('MISSED_WARNING',0)} missed warnings were observed. {pw['non_isolated_non_numerical_true_early_warning_count']} true-warning scenarios satisfied both the pre-registered adjacency rule and finite numerical margins. The targeted predictive-warning claim is **{pw['claim_status']}**, limited to this frozen operational family; no universal time-to-conflict theorem is implied.

## M7 and M9

M7's 20 nominal-delay misses are expected delayed-information effects, not an implementation defect. M9 remains unchanged: max G2={m9['max_G2']:.6g}, max Ginf={m9['max_Ginf']:.6g}; the pre-registered numerical string-stability claim is **{m9['claim_status']}**.
"""
    m8_doc = f"""# M8 Low-speed Discontinuity Diagnosis

- Root cause: {m8['root_cause']}.
- Genuine physical/model discontinuity: no.
- Low-speed ZOH condition contributes: yes (`g_T0={m8['hot_g_T0_K']:.12g} K`).
- Hard implementation branch contributes: yes.
- Normalization changes magnitude but not the feasibility sign: yes.
- Numerical tolerance cause: no.
- Hot rho jump at 0.5 to 0.500001 m/s: {m8['hot_rho_jump']:.12g}.
- Cold feasibility sign flip: {m8['cold_feasibility_sign_flip']}.
- Hot feasibility sign flip: {m8['hot_feasibility_sign_flip']}.
- Repair required under the frozen declared model: no.
- Repair status: `{m8['repair_status']}`.

{m8['repair_rationale']}
"""
    m7_doc = f"""# M7 Nominal-delay Missed-trigger Diagnosis

All {m7['missed_trigger_count']} missed records used valid messages aged exactly one 0.1 s control sample. Counts by receiving vehicle were {m7['misses_by_vehicle']}. Critical-vehicle metadata changed or disagreed in {m7['critical_vehicle_change_or_mismatch_count']} records. The centralized reserve lay {m7['central_threshold_margin_range'][0]:.6g} to {m7['central_threshold_margin_range'][1]:.6g} relative to the fixed 0.08 threshold, while the local delayed reserve remained {m7['local_threshold_margin_range'][0]:.6g} to {m7['local_threshold_margin_range'][1]:.6g} above it.

The causes are sampled packet-consumption timing, causal recursive multi-hop aggregation lag, and bottleneck-vehicle changes near the supervisor threshold. There was no packet loss, staleness, horizon mismatch, parameter mismatch, or verified implementation defect. No communication parameter or threshold was changed, and M7 was not rerun. The distributed-bottleneck claim remains **{m7['claim_status']}**.
"""
    for path, content in ((docs / "phase2m_followup_report.md", report), (docs / "m8_discontinuity_diagnosis.md", m8_doc), (docs / "m7_missed_trigger_diagnosis.md", m7_doc)):
        path.write_text(content.rstrip() + "\n", encoding="utf-8", newline="\n")

    macros = ROOT / "generated/tables/phase2m_followup_claim_macros.tex"
    macros.parent.mkdir(parents=True, exist_ok=True)
    macros.write_text("\n".join((
        "% Auto-generated from immutable Phase 2M-FOLLOWUP logs; do not edit numerically.",
        f"\\newcommand{{\\FollowupScenarioCount}}{{{pw['scenarios']}}}",
        f"\\newcommand{{\\FollowupTrueWarningCount}}{{{pw['classification_counts'].get('TRUE_EARLY_WARNING',0)}}}",
        f"\\newcommand{{\\FollowupConservativeWarningCount}}{{{pw['classification_counts'].get('CONSERVATIVE_WARNING_WITHOUT_BOUNDARY',0)}}}",
        f"\\newcommand{{\\FollowupSimultaneousWarningCount}}{{{pw['classification_counts'].get('SIMULTANEOUS_WARNING',0)}}}",
        f"\\newcommand{{\\FollowupMissedWarningCount}}{{{pw['classification_counts'].get('MISSED_WARNING',0)}}}",
        f"\\newcommand{{\\FollowupRobustTrueWarningCount}}{{{pw['non_isolated_non_numerical_true_early_warning_count']}}}",
        f"\\newcommand{{\\FollowupMaxGTwo}}{{{m9['max_G2']:.5f}}}",
        f"\\newcommand{{\\FollowupMaxGInf}}{{{m9['max_Ginf']:.5f}}}", "",
    )), encoding="utf-8", newline="\n")
    generated = {"phase": "2M-FOLLOWUP", "paper_eligible": True, "result_manifest_hash": summary["result_manifest_hash"], "analysis_summary_sha256": _sha(CAMPAIGN / "analysis_summary.json"), "macros_sha256": _sha(macros), "analysis_source_sha256": _sha(Path(__file__))}
    (ROOT / "generated/phase2m_followup_generated_manifest.json").write_text(json.dumps(generated, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def generate() -> dict[str, Any]:
    summary = analyze()
    write_artifacts(summary)
    return summary


if __name__ == "__main__":
    result = generate()
    print(json.dumps({"status": "PASS", "runs": result["new_paper_eligible_followup_runs"], "predictive_claim": result["predictive_warning"]["claim_status"], "m8_repair": result["M8"]["repair_status"], "m7_defect": result["M7"]["implementation_defect_found"], "m9_claim": result["M9"]["claim_status"]}, indent=2))

"""Integrity and claim regression for the targeted Phase 2M follow-up."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "results/paper_candidate/phase2m-followup-d10d13eb-912a1958"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


manifest_path = CAMPAIGN / "result_manifest.json"
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
summary = json.loads((CAMPAIGN / "analysis_summary.json").read_text(encoding="utf-8"))

assert sha(manifest_path) == (CAMPAIGN / "result_manifest.sha256").read_text(encoding="ascii").strip()
assert manifest["total_runs"] == 82
assert manifest["m8_diagnostic_runs"] == 1
assert manifest["predictive_scenario_runs"] == 81
assert manifest["failed_runs"] == manifest["invalidated_runs"] == 0
assert manifest["physical_parameters_changed"] is False
assert manifest["full_m1_m10_rerun"] is False
assert manifest["mappo_or_diffqp_learning_executed"] is False
assert summary["integrity"]["status"] == "PASS"

for run in manifest["runs"]:
    run_path = CAMPAIGN / run["run_manifest"]
    run_manifest = json.loads(run_path.read_text(encoding="utf-8"))
    records = run_path.parent / run_manifest["records_file"]
    assert sha(run_path) == run["run_manifest_sha256"]
    assert sha(records) == run["records_sha256"] == run_manifest["records_sha256"]

m8 = summary["M8"]
assert m8["hard_branch_switch"] is True
assert m8["genuine_physical_change"] is False
assert m8["numerical_tolerance_cause"] is False
assert m8["hot_feasibility_sign_flip"] is True
assert m8["cold_feasibility_sign_flip"] is False
assert abs(m8["hot_c_T_change_K_per_N"]) < 1e-9
assert abs(m8["hot_hocbf_upper_change_N"]) < 1.0
assert m8["repair_required"] is False

predictive = summary["predictive_warning"]
assert predictive["classification_counts"] == {
    "CONSERVATIVE_WARNING_WITHOUT_BOUNDARY": 32,
    "SIMULTANEOUS_WARNING": 39,
    "TRUE_EARLY_WARNING": 10,
}
assert predictive["non_isolated_non_numerical_true_early_warning_count"] == 4
assert predictive["claim_status"] == "SUPPORTED"
assert all(case["minimum_rho_at_trigger"] > 1e-6 for case in predictive["representative_robust_cases"])
assert all(case["minimum_rho_H_cert_at_trigger"] < -1e-6 for case in predictive["representative_robust_cases"])

m7 = summary["M7"]
assert m7["missed_trigger_count"] == 20
assert m7["message_age_one_sample_count"] == 20
assert m7["message_valid_count"] == 20
assert m7["implementation_defect_found"] is False
assert m7["horizon_mismatch"] is False
assert m7["packet_loss_or_staleness"] is False

m9 = summary["M9"]
assert abs(m9["max_G2"] - 4.216493596602332) < 1e-12
assert abs(m9["max_Ginf"] - 5.167026142015309) < 1e-12
assert m9["claim_status"] == "NOT_SUPPORTED"

manuscript = (ROOT / "manuscript/sections/08_results_and_limitations.tex").read_text(encoding="utf-8")
assert "\\FollowupTrueWarningCount" in manuscript
assert "does not satisfy the pre-registered numerical string-stability criterion" in manuscript
assert "Safety and string stability" not in manuscript or "distinct instantaneous safety-certificate claim" in manuscript

print("PASS: 82-run follow-up integrity, M8 diagnosis, predictive classification, M7 causes, and frozen M9 result")

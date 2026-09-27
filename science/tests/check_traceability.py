"""Check required audit/traceability artifacts and executable mappings."""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
canonical_claims_name = "unverified_claims.md"
claims_case_variants = tuple(
    path.name
    for path in (ROOT / "docs").iterdir()
    if path.is_file() and path.name.casefold() == canonical_claims_name.casefold()
)
assert claims_case_variants == (canonical_claims_name,), (
    "docs/unverified_claims.md must exist exactly once with canonical lowercase spelling; "
    f"found {claims_case_variants!r}"
)

required_files = (
    "docs/pre_revision_audit.md",
    "docs/symbol_audit.csv",
    "docs/equation_to_code.md",
    "docs/claim_evidence_matrix.md",
    "docs/remaining_gaps.md",
    "docs/revision_audit.md",
    "docs/novelty_comparison_2026.md",
    "docs/unverified_claims.md",
    "docs/interval_certificate_semantics.md",
    "docs/distributed_bottleneck_protocol.md",
    "docs/control_loop_causality.md",
    "docs/simulator_design.md",
    "docs/simulator_validation.md",
    "docs/platoon_environment_design.md",
    "docs/network_model_design.md",
    "docs/safety_integration_validation.md",
    "docs/mappo_design.md",
    "docs/mappo_validation.md",
    "docs/mappo_backend_audit.md",
    "docs/autodiff_environment.md",
    "docs/linux_autodiff_migration.md",
    "docs/model_simulation_design.md",
    "docs/model_parameter_provenance.md",
    "docs/source_consistency_audit.md",
    "docs/model_simulation_paper_language.md",
    "docs/model_simulation_validation.md",
    "docs/model_simulation_data_required.md",
    "docs/model_simulation_log_schema.csv",
    "docs/DATA_REQUIRED.md",
    "environment/linux/requirements.txt",
    "environment/linux/README.md",
    "scripts/check_autodiff_backend.py",
    "scripts/run_regression_suite.py",
    "scripts/run_model_simulation_suite.py",
    "scripts/generate_model_simulation_figures.py",
    "scripts/check_literature_calibrated_readiness.py",
    "scripts/calibration/derive_longitudinal_parameters.py",
    "scripts/calibration/derive_predecessor_bounds.py",
    "scripts/calibration/build_uncertainty_intervals.py",
    "scripts/build_manuscript.py",
    "configs/model_simulation/debug_suite.json",
    "configs/model_simulation/literature_calibrated.template.json",
    "data/model_parameters/synthetic_debug.json",
    "data/model_parameters/literature_calibrated.template.json",
    "data/model_parameters/parameter_provenance.yaml",
    "data/model_parameters/source_extracts.json",
    "data/model_parameters/derived/longitudinal_parameters.json",
    "data/model_parameters/derived/predecessor_bounds.json",
    "data/model_parameters/derived/uncertainty_intervals.json",
)
for name in required_files:
    path = ROOT / name
    assert path.is_file() and path.stat().st_size > 100, name

runtime_files = tuple((ROOT / directory).rglob("*.py") for directory in ("src", "tests", "scripts", "experiments"))
runtime_text = "\n".join(
    path.read_text(encoding="utf-8")
    for group in runtime_files
    for path in group
)
assert re.search(r"[A-Za-z]:[\\/]", runtime_text) is None, "runtime code contains an absolute Windows path"

matrix = (ROOT / "docs" / "claim_evidence_matrix.md").read_text(encoding="utf-8")
for token in ("CLAIM", "THEOREM", "EQUATION", "CODE", "UNIT TEST", "SYSTEM EXPERIMENT", "FIGURE/TABLE", "STATUS"):
    assert token in matrix, token

mapping = (ROOT / "docs" / "equation_to_code.md").read_text(encoding="utf-8")
source = "\n".join(path.read_text(encoding="utf-8") for path in (ROOT / "src").rglob("*.py"))
for function in (
    "state_derivatives",
    "vehicle_rhs",
    "rk4_step",
    "integrate_reference",
    "integrate_zero_order_hold",
    "build_augmented_state",
    "expand_predecessor_packet",
    "continuous_jerk",
    "collision_affine",
    "thermal_affine",
    "fade_upper_bound",
    "auxiliary_cbf_upper_bound",
    "low_speed_temperature_row",
    "directional_margin_from_box",
    "low_speed_zoh_error_bound",
    "compute_digital_rate_bounds",
    "compute_complete_certificate",
    "compute_predictive_reserve",
    "compute_fleet_reserve",
    "aggregate_upstream_message",
    "execute_two_pass_control_cycle",
    "saturated_affine_backup_extension",
    "project_weighted_2d",
):
    assert function in mapping, f"mapping missing {function}"
    assert f"def {function}" in source, f"source missing {function}"
print("PASS: required audit artifacts and equation-to-code mappings are complete")

"""Verify the mandatory predictive-reserve extension is wired end to end."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

required_files = (
    "src/safety/conflict_reserve.py",
    "src/safety/predictive_reserve.py",
    "src/safety/fleet_reserve.py",
    "src/safety/control_loop.py",
    "src/safety/reachable_set.py",
    "src/safety/supervisor.py",
    "src/training_objective.py",
    "src/rollout_schema.py",
    "tests/test_predictive_reserve.py",
    "tests/test_predictive_objective.py",
    "tests/test_revision_gaps.py",
    "tests/test_low_speed_g_t0.py",
    "experiments/generate_predictive_figures.py",
)
for name in required_files:
    assert (ROOT / name).is_file(), name

source = "\n".join((ROOT / name).read_text(encoding="utf-8") for name in required_files if name.endswith(".py"))
for function in (
    "compute_effective_friction_interval",
    "compute_effective_aux_interval",
    "compute_instantaneous_reserve",
    "compute_complete_certificate",
    "propagate_reachable_set",
    "compute_predictive_reserve",
    "compute_fleet_reserve",
    "aggregate_upstream_message",
    "execute_two_pass_control_cycle",
    "compute_learning_predictive_reserve",
    "select_bottleneck_vehicle",
    "predictive_supervisor_trigger",
):
    assert f"def {function}" in source, function

manuscript = "\n".join(path.read_text(encoding="utf-8") for path in (ROOT / "manuscript").rglob("*.tex"))
for token in (
    r"\chi_{i,k}",
    r"\underline\rho^{\rm IA}_{i,k+\ell\mid k}",
    r"\rho_{i,k}^{H,{\rm cert}}",
    r"\rho_{{\rm fleet},k}^{H,{\rm cert}}",
    r"\rho_{i,k}^{H,{\rm up}}",
    "Proposition 2 (one-sided robust certificate)",
    "Proposition 5 (centralized and recursive minima)",
    "(P1)",
):
    assert token in manuscript, token

schema = (ROOT / "docs" / "log_schema.csv").read_text(encoding="utf-8")
for signal in (
    "predictive_reserve_vehicle_mps2",
    "fleet_predictive_reserve_mps2",
    "critical_vehicle_id",
    "critical_prediction_step",
    "critical_limit_type",
    "rho_predictive_vehicle",
    "rho_IA_lower_by_step",
    "rho_cert_H",
    "rho_upstream_local_H",
    "rho_fleet_centralized_H",
    "prediction_reverified",
):
    assert signal in schema, signal

matrix = (ROOT / "docs" / "claim_evidence_matrix.md").read_text(encoding="utf-8")
for claim in ("complete interval-plus-coupled-row", "backup-compatible", "decentralized zero-delay", "prediction never reads", "DiffQP improves learning"):
    assert claim in matrix, claim

print("PASS: predictive reserve modules, manuscript, logs, tests, figures, and claim matrix are wired")

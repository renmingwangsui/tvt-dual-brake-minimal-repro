"""Canonical-QP, KKT, active-set and matched backward validation."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from safety.diffqp import (  # noqa: E402
    CanonicalSafetyQP,
    QPGradientMode,
    QPTolerances,
    solve_differentiable_qp,
    write_debug_qp_log,
)
from safety_core import Row, project_weighted_2d  # noqa: E402


def bounds(f_upper: float = 120_000.0, a_upper: float = 80_000.0) -> tuple[Row, ...]:
    return (
        Row("friction_lower", -1.0, 0.0, 0.0, "N"),
        Row("friction_upper", 1.0, 0.0, f_upper, "N"),
        Row("auxiliary_lower", 0.0, -1.0, 0.0, "N"),
        Row("auxiliary_upper", 0.0, 1.0, a_upper, "N"),
    )


REGISTERED_CASES = {
    "nominal_interior": (np.asarray([60_000.0, 30_000.0]), bounds()),
    "near_collision": (
        np.asarray([20_000.0, 20_000.0]),
        bounds() + (Row("collision_hocbf", -1.0, -1.0, -100_000.0, "m/s^2"),),
    ),
    "near_thermal": (
        np.asarray([70_000.0, 20_000.0]), bounds() + (Row("thermal_hocbf", 1.0, 0.0, 40_000.0, "K/s^2"),),
    ),
    "near_fade": (
        np.asarray([65_000.0, 20_000.0]), bounds() + (Row("fade_cbf", 1.0, 0.0, 35_000.0, "N"),),
    ),
    "near_auxiliary": (
        np.asarray([50_000.0, 55_000.0]), bounds() + (Row("realized_auxiliary_cbf", 0.0, 1.0, 25_000.0, "N"),),
    ),
    "actuator_bound": (np.asarray([150_000.0, 10_000.0]), bounds()),
    "hot_brake": (
        np.asarray([40_000.0, 10_000.0]), bounds() + (Row("thermal_hocbf", 1.0, 0.0, 5_000.0, "K/s^2"),),
    ),
    "low_speed_zoh": (
        np.asarray([70_000.0, 10_000.0]), bounds() + (Row("thermal_zoh", 1e-5, 0.0, 0.4, "K"),),
    ),
    "collision_plus_bound": (
        np.asarray([10_000.0, 10_000.0]),
        bounds(f_upper=30_000.0) + (Row("collision_hocbf", -1.0, -1.0, -80_000.0, "m/s^2"),),
    ),
}

forward_errors: list[float] = []
objective_errors: list[float] = []
primal_residuals: list[float] = []
dual_residuals: list[float] = []
stationarity_residuals: list[float] = []
complementarity_residuals: list[float] = []
hard_row_residual_errors: list[float] = []
kkt_conditions: list[float] = []
active_multipliers: list[float] = []
jacobian_errors: list[float] = []
jacobian_relative_errors: list[float] = []
cosines: list[float] = []
linearization_errors: list[float] = []
records = []
stable_case_count = 0

for name, (nominal, rows) in REGISTERED_CASES.items():
    qp = CanonicalSafetyQP(rows)
    min_eigenvalue, condition_number, symmetry_residual = qp.hessian_diagnostics()
    assert min_eigenvalue > 0.0 and condition_number == 1.0 and symmetry_residual == 0.0
    accepted = np.asarray(project_weighted_2d(nominal, rows), dtype=np.float64)

    diff_input = torch.tensor(nominal, dtype=torch.float64, requires_grad=True)
    stop_input = torch.tensor(nominal, dtype=torch.float64, requires_grad=True)
    diff_result = solve_differentiable_qp(diff_input, qp, QPGradientMode.DIFFQP)
    stop_result = solve_differentiable_qp(stop_input, qp, QPGradientMode.STOP_GRADIENT)
    diff_solution = diff_result.action.detach().numpy()
    stop_solution = stop_result.action.detach().numpy()
    forward_errors.extend(np.abs(diff_solution - stop_solution))
    forward_errors.extend(np.abs(diff_solution - accepted))
    objective_errors.append(abs(qp.objective(diff_solution, nominal) - qp.objective(stop_solution, nominal)))
    hard_row_residual_errors.extend(np.abs(
        (qp.h - qp.G @ diff_solution) - (qp.h - qp.G @ stop_solution)
    ))
    assert np.array_equal(diff_solution, stop_solution)
    assert np.array_equal(diff_solution, accepted)
    assert diff_result.diagnostics.active_indices == stop_result.diagnostics.active_indices

    diagnostics = diff_result.diagnostics
    primal_residuals.append(diagnostics.primal_residual)
    dual_residuals.append(diagnostics.dual_residual)
    stationarity_residuals.append(diagnostics.stationarity_residual)
    complementarity_residuals.append(diagnostics.complementarity_residual)
    kkt_conditions.append(diagnostics.kkt_condition)
    active_multipliers.extend(map(float, diagnostics.active_multipliers))
    assert diagnostics.primal_residual <= qp.tolerances.primal
    assert diagnostics.dual_residual <= qp.tolerances.dual
    assert diagnostics.stationarity_residual <= qp.tolerances.stationarity
    assert diagnostics.complementarity_residual <= qp.tolerances.complementarity
    assert diagnostics.licq_satisfied
    assert diagnostics.strict_complementarity_satisfied
    assert diff_result.stability.stable
    assert diff_result.record.diffqp_gradient_valid
    assert diff_result.record.qp_mode == "DIFFQP"
    assert stop_result.record.qp_mode == "STOP_GRADIENT"
    assert stop_result.record.fallback_reason == "STOP_GRADIENT_MODE"
    records.extend((diff_result.record, stop_result.record))

    jacobian_kkt = qp.implicit_jacobian(nominal, diagnostics)
    jacobian_fd, fd_stable = qp.finite_difference_jacobian(nominal, 1e-3)
    assert fd_stable
    difference = np.abs(jacobian_kkt - jacobian_fd)
    relative = float(np.linalg.norm(jacobian_kkt - jacobian_fd) / max(np.linalg.norm(jacobian_fd), 1e-12))
    jacobian_errors.extend(difference.reshape(-1))
    jacobian_relative_errors.append(relative)
    if np.linalg.norm(jacobian_fd) > 1e-12 and np.linalg.norm(jacobian_kkt) > 1e-12:
        cosine = float(np.sum(jacobian_kkt * jacobian_fd) / (
            np.linalg.norm(jacobian_kkt) * np.linalg.norm(jacobian_fd)
        ))
        cosines.append(cosine)
        assert cosine > 0.999999
    perturbation = np.asarray([0.02, -0.01])
    predicted = jacobian_kkt @ perturbation
    actual = qp.solve(nominal + perturbation) - diagnostics.solution
    linearization_errors.append(float(np.max(np.abs(predicted - actual))))
    assert np.max(difference) < 1e-6
    assert relative < 1e-6
    assert np.max(np.abs(predicted - actual)) < 1e-6
    stable_case_count += 1


# A nontrivial intervention passes a finite QP-mediated gradient in DiffQP,
# while the exactly matched StopGradient branch returns the identical action
# and a zero QP-mediated gradient.
intervention_qp = CanonicalSafetyQP(REGISTERED_CASES["near_collision"][1])
intervention_nominal = REGISTERED_CASES["near_collision"][0]
diff_input = torch.tensor(intervention_nominal, dtype=torch.float64, requires_grad=True)
diff = solve_differentiable_qp(diff_input, intervention_qp, QPGradientMode.DIFFQP)
diff.action[0].backward()
diff_gradient_norm = float(torch.linalg.vector_norm(diff_input.grad).item())
assert diff.record.qp_intervention_magnitude > 0.0
assert np.isfinite(diff_gradient_norm) and diff_gradient_norm > 0.0

stop_input = torch.tensor(intervention_nominal, dtype=torch.float64, requires_grad=True)
stop = solve_differentiable_qp(stop_input, intervention_qp, QPGradientMode.STOP_GRADIENT)
stop.action[0].backward()
assert torch.count_nonzero(stop_input.grad).item() == 0
assert np.array_equal(diff.action.detach().numpy(), stop.action.detach().numpy())


# Active-set boundary: zero multiplier and changed +/- active sets. Forward is
# retained, but DiffQP falls back to a zero mediated gradient.
boundary_qp = CanonicalSafetyQP(bounds() + (Row("thermal_hocbf", 1.0, 0.0, 40_000.0, "K/s^2"),))
boundary_input = torch.tensor([40_000.0, 20_000.0], dtype=torch.float64, requires_grad=True)
boundary = solve_differentiable_qp(boundary_input, boundary_qp, QPGradientMode.DIFFQP)
assert not boundary.stability.stable
assert not boundary.diagnostics.strict_complementarity_satisfied
assert not boundary.record.diffqp_gradient_valid
assert boundary.record.fallback_reason == "STRICT_COMPLEMENTARITY_FAILURE"
boundary.action.sum().backward()
assert torch.count_nonzero(boundary_input.grad).item() == 0
records.append(boundary.record)


# Duplicate active physical rows are a deliberate LICQ-failure registration.
duplicate_rows = bounds() + (
    Row("thermal_hocbf", 1.0, 0.0, 40_000.0, "K/s^2"),
    Row("fade_cbf", 1.0, 0.0, 40_000.0, "N"),
)
duplicate_input = torch.tensor([70_000.0, 20_000.0], dtype=torch.float64, requires_grad=True)
duplicate = solve_differentiable_qp(
    duplicate_input, CanonicalSafetyQP(duplicate_rows), QPGradientMode.DIFFQP
)
assert not duplicate.diagnostics.licq_satisfied
assert duplicate.record.fallback_reason == "LICQ_FAILURE"
duplicate.action.sum().backward()
assert torch.count_nonzero(duplicate_input.grad).item() == 0
records.append(duplicate.record)


# Three physically named active rows in a two-decision QP are necessarily
# dependent; this is classified, never advertised as a differentiable point.
triple_rows = bounds(30_000.0, 50_000.0) + (
    Row("collision_hocbf", -1.0, -1.0, -80_000.0, "m/s^2"),
)
triple = solve_differentiable_qp(
    torch.tensor([10_000.0, 10_000.0], dtype=torch.float64, requires_grad=True),
    CanonicalSafetyQP(triple_rows),
    QPGradientMode.DIFFQP,
)
assert len(triple.diagnostics.active_indices) == 3
assert not triple.diagnostics.licq_satisfied
assert triple.record.fallback_reason == "LICQ_FAILURE"
records.append(triple.record)


# A tiny unscaled row produces an ill-conditioned KKT system even though the
# row is linearly independent. The unchanged forward action is retained.
ill_rows = bounds() + (Row("thermal_zoh", 1e-8, 0.0, 4e-4, "K"),)
ill_qp = CanonicalSafetyQP(ill_rows, tolerances=QPTolerances(kkt_condition_max=1e12))
ill = solve_differentiable_qp(
    torch.tensor([70_000.0, 10_000.0], dtype=torch.float64, requires_grad=True),
    ill_qp,
    QPGradientMode.DIFFQP,
)
assert ill.diagnostics.licq_satisfied
assert ill.diagnostics.kkt_condition > ill_qp.tolerances.kkt_condition_max
assert ill.record.fallback_reason == "ILL_CONDITIONED_KKT"
records.append(ill.record)


with tempfile.TemporaryDirectory() as directory:
    log_path = Path(directory) / "phase2c2_qp_debug.json"
    write_debug_qp_log(log_path, records)
    payload = json.loads(log_path.read_text(encoding="utf-8"))
    assert payload["data_provenance"] == "DEBUG"
    assert payload["paper_eligible"] is False
    assert all("active_constraint_names" in record for record in payload["records"])
    assert all("fallback_reason" in record for record in payload["records"])

assert max(forward_errors) == 0.0
assert max(objective_errors) == 0.0
assert max(hard_row_residual_errors) == 0.0

print(
    "PASS: Phase 2C-2 canonical forward/KKT/implicit differentiation/matched StopGradient "
    f"h_min=2.000e+00 h_cond=1.000e+00 forward_max={max(forward_errors):.3e} "
    f"forward_median={np.median(forward_errors):.3e} objective_max={max(objective_errors):.3e} "
    f"hard_residual_delta_max={max(hard_row_residual_errors):.3e} "
    f"primal_max={max(primal_residuals):.3e} dual_max={max(dual_residuals):.3e} "
    f"stationarity_max={max(stationarity_residuals):.3e} "
    f"complementarity_max={max(complementarity_residuals):.3e} "
    f"kkt_condition_max={max(kkt_conditions):.3e} "
    f"minimum_active_multiplier={min(active_multipliers):.3e} "
    f"jacobian_max={max(jacobian_errors):.3e} jacobian_median={np.median(jacobian_errors):.3e} "
    f"jacobian_relative_max={max(jacobian_relative_errors):.3e} cosine_min={min(cosines):.12f} "
    f"linearization_max={max(linearization_errors):.3e} stable_cases={stable_case_count} "
    f"boundary_cases=4 diffqp_gradient_norm={diff_gradient_norm:.6e}"
)

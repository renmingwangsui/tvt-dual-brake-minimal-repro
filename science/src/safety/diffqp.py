"""Matched DiffQP/StopGradient wrapper for the accepted hard safety QP.

The accepted controller solves a two-decision strictly convex projection in
``(u_f, u_a)``.  Jerk uncertainty is handled by the certified rows and the
predictive monitor; the implemented forward QP has no independent jerk-slack
decision.  This module therefore preserves that exact mathematical problem:

    min_z  sum_i w_i (z_i - u_RL_i)^2
    s.t.   G z <= h,

where ``H=2*diag(w)`` is fixed, ``q(u_RL)=-H*u_RL``, and ``G,h`` are the
already-audited hard rows constructed from physical state/certificate data.
Only the ``q(u_RL)`` path is differentiated.  Certified row construction,
predictive monitoring, distributed aggregation and supervisor decisions stay
detached and non-learned.

Both gradient modes call :func:`safety_core.project_weighted_2d` for the
forward solve.  Finite differences are validation utilities only; production
backward solves the active-set KKT linear system and never writes ``.grad``.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import Tensor

from safety_core import Row, project_weighted_2d


class QPGradientMode(str, Enum):
    DIFFQP = "DIFFQP"
    STOP_GRADIENT = "STOP_GRADIENT"


@dataclass(frozen=True)
class QPTolerances:
    active: float = 1e-6
    primal: float = 1e-7
    dual: float = 1e-7
    stationarity: float = 1e-6
    complementarity: float = 1e-5
    strict_complementarity: float = 1e-9
    licq_singular_value: float = 1e-10
    kkt_condition_max: float = 1e12
    stability_epsilon_N: float = 1e-3


@dataclass(frozen=True)
class KKTDiagnostics:
    solution: np.ndarray
    objective: float
    active_indices: tuple[int, ...]
    active_names: tuple[str, ...]
    active_multipliers: np.ndarray
    primal_residual: float
    dual_residual: float
    stationarity_residual: float
    complementarity_residual: float
    licq_satisfied: bool
    active_rank: int
    active_singular_values: np.ndarray
    minimum_active_singular_value: float
    active_matrix_condition: float
    strict_complementarity_satisfied: bool
    minimum_active_multiplier: float
    kkt_condition: float


@dataclass
class QPDebugRecord:
    qp_mode: str
    active_set: tuple[int, ...]
    active_constraint_names: tuple[str, ...]
    active_multipliers: tuple[float, ...]
    primal_residual: float
    dual_residual: float
    stationarity_residual: float
    complementarity_residual: float
    licq_satisfied: bool
    strict_complementarity_satisfied: bool
    kkt_condition: float
    diffqp_gradient_valid: bool
    fallback_reason: str
    qp_intervention_magnitude: float
    actor_gradient_norm: float | None = None
    data_provenance: str = "DEBUG"
    paper_eligible: bool = False


@dataclass(frozen=True)
class ActiveSetStability:
    stable: bool
    base_active_set: tuple[int, ...]
    perturbed_active_sets: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class DifferentiableQPResult:
    action: Tensor
    diagnostics: KKTDiagnostics
    stability: ActiveSetStability
    record: QPDebugRecord


@dataclass(frozen=True)
class CanonicalSafetyQP:
    """Canonical matrix view backed by the accepted forward solver."""

    rows: tuple[Row, ...]
    weights: tuple[float, float] = (1.0, 1.0)
    tolerances: QPTolerances = QPTolerances()

    def __post_init__(self) -> None:
        if not self.rows:
            raise ValueError("canonical safety QP requires at least one hard row")
        if len(self.weights) != 2 or min(self.weights) <= 0.0:
            raise ValueError("canonical safety QP weights must be positive")

    @property
    def H(self) -> np.ndarray:
        return 2.0 * np.diag(np.asarray(self.weights, dtype=np.float64))

    @property
    def G(self) -> np.ndarray:
        return np.asarray([(row.g_f, row.g_a) for row in self.rows], dtype=np.float64)

    @property
    def h(self) -> np.ndarray:
        return np.asarray([row.h for row in self.rows], dtype=np.float64)

    @property
    def row_names(self) -> tuple[str, ...]:
        return tuple(row.name for row in self.rows)

    def q(self, nominal: Sequence[float]) -> np.ndarray:
        return -self.H @ np.asarray(nominal, dtype=np.float64)

    def objective(self, solution: Sequence[float], nominal: Sequence[float]) -> float:
        delta = np.asarray(solution, dtype=np.float64) - np.asarray(nominal, dtype=np.float64)
        return float(np.sum(np.asarray(self.weights, dtype=np.float64) * np.square(delta)))

    def solve(self, nominal: Sequence[float]) -> np.ndarray:
        # This is intentionally the same accepted solver used by the Phase 2B
        # controller.  Neither gradient mode has an alternate forward solver.
        return np.asarray(project_weighted_2d(nominal, self.rows, self.weights), dtype=np.float64)

    def hessian_diagnostics(self) -> tuple[float, float, float]:
        symmetry_residual = float(np.max(np.abs(self.H - self.H.T)))
        eigenvalues = np.linalg.eigvalsh(self.H)
        return float(np.min(eigenvalues)), float(np.linalg.cond(self.H)), symmetry_residual

    def diagnostics(self, nominal: Sequence[float], solution: Sequence[float] | None = None) -> KKTDiagnostics:
        z = self.solve(nominal) if solution is None else np.asarray(solution, dtype=np.float64)
        violations = self.G @ z - self.h
        active = tuple(int(index) for index in np.flatnonzero(np.abs(violations) <= self.tolerances.active))
        gradient = self.H @ z + self.q(nominal)
        if active:
            active_matrix = self.G[np.asarray(active, dtype=np.int64)]
            multipliers, *_ = np.linalg.lstsq(active_matrix.T, -gradient, rcond=None)
            singular_values = np.linalg.svd(active_matrix, compute_uv=False)
            rank = int(np.linalg.matrix_rank(active_matrix, tol=self.tolerances.licq_singular_value))
            minimum_singular = float(np.min(singular_values)) if singular_values.size else float("inf")
            active_condition = float(np.linalg.cond(active_matrix))
            stationarity = gradient + active_matrix.T @ multipliers
            complementarity = multipliers * violations[np.asarray(active, dtype=np.int64)]
            kkt = np.block([
                [self.H, active_matrix.T],
                [active_matrix, np.zeros((len(active), len(active)), dtype=np.float64)],
            ])
            kkt_condition = float(np.linalg.cond(kkt))
            minimum_multiplier = float(np.min(multipliers))
        else:
            active_matrix = np.empty((0, 2), dtype=np.float64)
            multipliers = np.empty(0, dtype=np.float64)
            singular_values = np.empty(0, dtype=np.float64)
            rank = 0
            minimum_singular = float("inf")
            active_condition = 1.0
            stationarity = gradient
            complementarity = np.empty(0, dtype=np.float64)
            kkt_condition = float(np.linalg.cond(self.H))
            minimum_multiplier = float("inf")
        licq = rank == len(active) and minimum_singular >= self.tolerances.licq_singular_value
        strict = not active or bool(np.all(multipliers > self.tolerances.strict_complementarity))
        return KKTDiagnostics(
            solution=z,
            objective=self.objective(z, nominal),
            active_indices=active,
            active_names=tuple(self.row_names[index] for index in active),
            active_multipliers=np.asarray(multipliers, dtype=np.float64),
            primal_residual=float(max(0.0, np.max(violations))),
            dual_residual=float(max(0.0, np.max(-multipliers))) if multipliers.size else 0.0,
            stationarity_residual=float(np.linalg.norm(stationarity, ord=np.inf)),
            complementarity_residual=float(np.max(np.abs(complementarity))) if complementarity.size else 0.0,
            licq_satisfied=licq,
            active_rank=rank,
            active_singular_values=singular_values,
            minimum_active_singular_value=minimum_singular,
            active_matrix_condition=active_condition,
            strict_complementarity_satisfied=strict,
            minimum_active_multiplier=minimum_multiplier,
            kkt_condition=kkt_condition,
        )

    def active_set_stability(self, nominal: Sequence[float]) -> ActiveSetStability:
        nominal_array = np.asarray(nominal, dtype=np.float64)
        base = self.diagnostics(nominal_array).active_indices
        perturbed: list[tuple[int, ...]] = []
        for dimension in range(2):
            for sign in (-1.0, 1.0):
                candidate = nominal_array.copy()
                candidate[dimension] += sign * self.tolerances.stability_epsilon_N
                perturbed.append(self.diagnostics(candidate).active_indices)
        return ActiveSetStability(all(item == base for item in perturbed), base, tuple(perturbed))

    def backward_validity(
        self, diagnostics: KKTDiagnostics, stability: ActiveSetStability
    ) -> tuple[bool, str]:
        t = self.tolerances
        if diagnostics.primal_residual > t.primal:
            return False, "KKT_PRIMAL_RESIDUAL"
        # Multiplier uniqueness and the meaning of multiplier-based residuals
        # require LICQ, so structural regularity is classified first.
        if not diagnostics.licq_satisfied:
            return False, "LICQ_FAILURE"
        if not diagnostics.strict_complementarity_satisfied:
            return False, "STRICT_COMPLEMENTARITY_FAILURE"
        if diagnostics.dual_residual > t.dual:
            return False, "KKT_DUAL_RESIDUAL"
        if diagnostics.stationarity_residual > t.stationarity:
            return False, "KKT_STATIONARITY_RESIDUAL"
        if diagnostics.complementarity_residual > t.complementarity:
            return False, "KKT_COMPLEMENTARITY_RESIDUAL"
        if not np.isfinite(diagnostics.kkt_condition) or diagnostics.kkt_condition > t.kkt_condition_max:
            return False, "ILL_CONDITIONED_KKT"
        if not stability.stable:
            return False, "BOUNDARY_ACTIVE_SET"
        return True, "NONE"

    def implicit_jacobian(self, nominal: Sequence[float], diagnostics: KKTDiagnostics) -> np.ndarray:
        active = diagnostics.active_indices
        if active:
            active_matrix = self.G[np.asarray(active, dtype=np.int64)]
            kkt = np.block([
                [self.H, active_matrix.T],
                [active_matrix, np.zeros((len(active), len(active)), dtype=np.float64)],
            ])
            rhs = np.vstack((self.H, np.zeros((len(active), 2), dtype=np.float64)))
            derivative = np.linalg.solve(kkt, rhs)
            return derivative[:2]
        return np.linalg.solve(self.H, self.H)

    def finite_difference_jacobian(self, nominal: Sequence[float], epsilon_N: float) -> tuple[np.ndarray, bool]:
        if epsilon_N <= 0.0:
            raise ValueError("finite-difference epsilon must be positive")
        nominal_array = np.asarray(nominal, dtype=np.float64)
        base_active = self.diagnostics(nominal_array).active_indices
        jacobian = np.zeros((2, 2), dtype=np.float64)
        stable = True
        for dimension in range(2):
            plus = nominal_array.copy()
            minus = nominal_array.copy()
            plus[dimension] += epsilon_N
            minus[dimension] -= epsilon_N
            plus_diagnostics = self.diagnostics(plus)
            minus_diagnostics = self.diagnostics(minus)
            stable = stable and plus_diagnostics.active_indices == base_active
            stable = stable and minus_diagnostics.active_indices == base_active
            jacobian[:, dimension] = (
                plus_diagnostics.solution - minus_diagnostics.solution
            ) / (2.0 * epsilon_N)
        return jacobian, stable


class _ImplicitSafetyQP(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx: object,
        nominal: Tensor,
        qp: CanonicalSafetyQP,
        mode: str,
        record_box: dict[str, object],
    ) -> Tensor:
        if nominal.ndim != 1 or nominal.numel() != 2:
            raise ValueError("differentiable safety QP expects one two-dimensional action")
        nominal_array = nominal.detach().cpu().numpy().astype(np.float64, copy=False)
        solution = qp.solve(nominal_array)
        diagnostics = qp.diagnostics(nominal_array, solution)
        stability = qp.active_set_stability(nominal_array)
        valid, fallback_reason = qp.backward_validity(diagnostics, stability)
        gradient_mode = QPGradientMode(mode)
        if gradient_mode is QPGradientMode.DIFFQP and valid:
            jacobian = qp.implicit_jacobian(nominal_array, diagnostics)
        else:
            jacobian = np.zeros((2, 2), dtype=np.float64)
        ctx.save_for_backward(torch.as_tensor(jacobian, dtype=nominal.dtype, device=nominal.device))
        record = QPDebugRecord(
            qp_mode=gradient_mode.value,
            active_set=diagnostics.active_indices,
            active_constraint_names=diagnostics.active_names,
            active_multipliers=tuple(map(float, diagnostics.active_multipliers)),
            primal_residual=diagnostics.primal_residual,
            dual_residual=diagnostics.dual_residual,
            stationarity_residual=diagnostics.stationarity_residual,
            complementarity_residual=diagnostics.complementarity_residual,
            licq_satisfied=diagnostics.licq_satisfied,
            strict_complementarity_satisfied=diagnostics.strict_complementarity_satisfied,
            kkt_condition=diagnostics.kkt_condition,
            diffqp_gradient_valid=gradient_mode is QPGradientMode.DIFFQP and valid,
            fallback_reason=(fallback_reason if gradient_mode is QPGradientMode.DIFFQP else "STOP_GRADIENT_MODE"),
            qp_intervention_magnitude=float(np.linalg.norm(solution - nominal_array)),
        )
        record_box["diagnostics"] = diagnostics
        record_box["stability"] = stability
        record_box["record"] = record
        return torch.as_tensor(solution, dtype=nominal.dtype, device=nominal.device)

    @staticmethod
    def backward(ctx: object, grad_output: Tensor) -> tuple[Tensor, None, None, None]:  # type: ignore[override]
        (jacobian,) = ctx.saved_tensors
        return jacobian.T @ grad_output, None, None, None


def solve_differentiable_qp(
    nominal: Tensor,
    qp: CanonicalSafetyQP,
    mode: QPGradientMode,
) -> DifferentiableQPResult:
    record_box: dict[str, object] = {}
    action = _ImplicitSafetyQP.apply(nominal, qp, mode.value, record_box)
    return DifferentiableQPResult(
        action=action,
        diagnostics=record_box["diagnostics"],  # type: ignore[arg-type]
        stability=record_box["stability"],  # type: ignore[arg-type]
        record=record_box["record"],  # type: ignore[arg-type]
    )


def write_debug_qp_log(path: Path, records: Sequence[QPDebugRecord]) -> None:
    payload = {
        "schema_version": 1,
        "data_provenance": "DEBUG",
        "paper_eligible": False,
        "records": [asdict(record) for record in records],
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


DIFFQP_BACKWARD_ACTIVE = True

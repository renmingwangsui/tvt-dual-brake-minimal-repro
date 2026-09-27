"""Certified monitoring reserve and separate differentiable learning surrogate."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

from safety_core import Interval

from .fleet_reserve import softmin
from .reachable_set import (
    AugmentedHybridState,
    IntervalHybridState,
    ReachabilityUncertainty,
    ReachableDynamicsParams,
    point_interval_state,
    propagate_interval_one_step,
)


@dataclass(frozen=True)
class ReserveLowerBound:
    """Computable IA lower bound; not the exact reachable-set infimum."""
    rho_ia_lower: float
    friction_interval_nonempty: bool
    auxiliary_interval_nonempty: bool
    critical_limit_type: str = "unspecified"

    @property
    def value(self) -> float:
        return self.rho_ia_lower


@dataclass(frozen=True)
class PredictiveReserveResult:
    rho_ia_lower_by_step: tuple[float, ...]
    rho_cert_H: float
    critical_step: int
    reachable_sets: tuple[IntervalHybridState, ...]
    interval_feasible_by_step: tuple[bool, ...]
    critical_limit_by_step: tuple[str, ...]
    critical_limit_type: str
    backup_control_widths_N: tuple[tuple[float, float], ...]
    first_control: tuple[float, float]
    rho_true_H: float | None = None
    prediction_mode: str = "certified_monitoring"

    @property
    def rho_by_step(self) -> tuple[float, ...]:
        return self.rho_ia_lower_by_step

    @property
    def rho_H(self) -> float:
        return self.rho_cert_H

    @property
    def reserve_by_step(self) -> tuple[float, ...]:
        return self.rho_ia_lower_by_step

    @property
    def min_reserve(self) -> float:
        return self.rho_cert_H


def compute_predictive_reserve(
    chi_k: AugmentedHybridState | IntervalHybridState,
    backup_policy_extension: Callable[[IntervalHybridState, int], Sequence[float | Interval]],
    horizon: int,
    params: ReachableDynamicsParams,
    uncertainty: ReachabilityUncertainty,
    reserve_lower_bound: Callable[[IntervalHybridState, int], ReserveLowerBound],
    provisional_executed_control: Sequence[float] | None = None,
) -> PredictiveReserveResult:
    """Certified IA tube: provisional QP command first, then Kappa_B(X).

    The returned quantity is a computable outward-rounded lower bound. A
    negative value is an inconclusive conservative warning, not proof that a
    physical uncertainty realization is infeasible.
    """
    if horizon < 1:
        raise ValueError("prediction horizon must be at least one")
    if provisional_executed_control is None:
        raise ValueError("prediction requires a computed provisional QP command")
    current = chi_k if isinstance(chi_k, IntervalHybridState) else point_interval_state(chi_k)
    tube: list[IntervalHybridState] = []
    reserves: list[float] = []
    interval_ok: list[bool] = []
    critical_limits: list[str] = []
    control_widths: list[tuple[float, float]] = []
    for step in range(1, horizon + 1):
        command = provisional_executed_control if step == 1 else backup_policy_extension(current, step)
        command_intervals = tuple(
            item if hasattr(item, "low") and hasattr(item, "high") else None
            for item in command
        )
        control_widths.append(tuple(
            0.0 if interval is None else float(interval.high - interval.low)
            for interval in command_intervals
        ))
        current = propagate_interval_one_step(current, command, params, uncertainty)
        bound = reserve_lower_bound(current, step)
        valid_intervals = bound.friction_interval_nonempty and bound.auxiliary_interval_nonempty
        tube.append(current)
        # A complete lower certificate already carries negative interval gaps.
        # Retain the flags for diagnostics; never overwrite rho with -infinity.
        reserves.append(bound.rho_ia_lower)
        interval_ok.append(valid_intervals)
        critical_limits.append(bound.critical_limit_type)
    critical_index = min(range(horizon), key=reserves.__getitem__)
    return PredictiveReserveResult(
        rho_ia_lower_by_step=tuple(reserves),
        rho_cert_H=reserves[critical_index],
        critical_step=critical_index + 1,
        reachable_sets=tuple(tube),
        interval_feasible_by_step=tuple(interval_ok),
        critical_limit_by_step=tuple(critical_limits),
        critical_limit_type=critical_limits[critical_index],
        backup_control_widths_N=tuple(control_widths),
        first_control=(float(provisional_executed_control[0]), float(provisional_executed_control[1])),
    )


def final_control_requires_reverification(
    provisional_control: Sequence[float], final_control: Sequence[float], tolerance_N: float = 1e-9
) -> bool:
    if tolerance_N < 0.0:
        raise ValueError("tolerance must be nonnegative")
    return any(abs(float(a) - float(b)) > tolerance_N for a, b in zip(provisional_control, final_control))


@dataclass(frozen=True)
class LearningPredictiveResult:
    reserve_by_step: tuple[float, ...]
    soft_horizon_reserve: float
    final_state: object


def compute_learning_predictive_reserve(
    state: object,
    policy: Callable[[object, int], object],
    differentiable_transition: Callable[[object, object, int], object],
    reserve_surrogate: Callable[[object, int], float],
    horizon: int,
    soft_temperature: float,
) -> LearningPredictiveResult:
    """Short differentiable-policy rollout; never used as the monitor certificate.

    The callbacks are intentionally backend-agnostic.  A Torch/JAX trainer can
    pass tensor-valued transitions and an autodiff-compatible softmin.  This
    pure-Python reference records the exact path and is regression tested with
    scalars; the production trainer remains experiment-owned.
    """
    if horizon < 1:
        raise ValueError("learning horizon must be at least one")
    current = state
    reserves: list[float] = []
    for step in range(1, horizon + 1):
        nominal_or_executed = policy(current, step)
        current = differentiable_transition(current, nominal_or_executed, step)
        reserves.append(float(reserve_surrogate(current, step)))
    return LearningPredictiveResult(tuple(reserves), softmin(reserves, soft_temperature), current)

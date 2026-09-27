"""Exact instantaneous collision--thermal--actuator conflict certificate.

The previous commands, gear, gear dwell time, and message age are part of the
state.  A nonnegative coupled reserve is not sufficient when either individual
command interval is empty; ``feasible`` implements the full conjunction.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Mapping, Sequence

from safety_core import Interval, VehicleParams
from .reachable_set import AugmentedHybridState


@dataclass(frozen=True)
class ReserveParameters:
    vehicle: VehicleParams
    friction_envelope_N: float
    auxiliary_envelope_N: float
    gear_power_limit_N: float
    realized_auxiliary_cbf_limit_N: float
    mode_auxiliary_limit_N: float
    thermal_hocbf_limit_N: float
    fade_cbf_limit_N: float
    low_speed_zoh_limit_N: float
    low_speed_zoh_g_T0_K: float
    friction_lower_N: float
    auxiliary_lower_N: float
    A_f: float
    A_a: float
    gamma_c_mps2: float
    mu_c_mps2: float
    Xi_hat_c_mps2: float
    alpha2_psi1_c_mps2: float

    def validate(self) -> None:
        self.vehicle.validate()
        if self.A_f <= 0.0 or self.A_a <= 0.0:
            raise ValueError("collision-row command coefficients must be positive")
        nonnegative = (
            self.friction_envelope_N,
            self.auxiliary_envelope_N,
            self.gear_power_limit_N,
            self.realized_auxiliary_cbf_limit_N,
            self.mode_auxiliary_limit_N,
            self.friction_lower_N,
            self.auxiliary_lower_N,
        )
        if min(nonnegative) < 0.0:
            raise ValueError("force limits must be nonnegative")
        if not isfinite(self.low_speed_zoh_g_T0_K):
            raise ValueError("low-speed state-only thermal reserve g_T0 must be finite")


@dataclass(frozen=True)
class InstantaneousReserveResult:
    L_f: float
    U_f: float
    L_a: float
    U_a: float
    D_c: float
    M: float
    friction_interval_nonempty: bool
    auxiliary_interval_nonempty: bool
    feasible: bool
    active_friction_limit: str
    active_auxiliary_limit: str
    low_speed_active: bool = False
    g_T0_K: float | None = None
    low_speed_thermal_feasible: bool = True

    def as_dict(self) -> dict[str, float | bool | str]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class CertificateScales:
    """Fixed positive normalizers; they change magnitude, never feasibility."""

    friction_force_N: float
    auxiliary_force_N: float
    collision_mps2: float
    low_speed_temperature_K: float = 1.0

    def validate(self) -> None:
        if min(
            self.friction_force_N,
            self.auxiliary_force_N,
            self.collision_mps2,
            self.low_speed_temperature_K,
        ) <= 0.0:
            raise ValueError("certificate scales must be strictly positive")


@dataclass(frozen=True)
class CompleteCertificateResult:
    delta_f_N: float
    delta_a_N: float
    M_mps2: float
    normalized_friction: float
    normalized_auxiliary: float
    normalized_collision: float
    rho: float
    feasible: bool
    critical_limit_type: str
    g_T0_K: float | None = None
    normalized_low_speed_thermal: float | None = None


def compute_interval_certificate_lower_bound(
    delta_f_N: Interval,
    delta_a_N: Interval,
    collision_reserve_mps2: Interval,
    scales: CertificateScales,
    low_speed_g_T0_K: Interval | None = None,
) -> tuple[float, str]:
    """Outward-interval lower bound, never advertised as the exact infimum."""
    scales.validate()
    terms = {
        "friction_interval": delta_f_N.low / scales.friction_force_N,
        "auxiliary_interval": delta_a_N.low / scales.auxiliary_force_N,
        "collision_demand": collision_reserve_mps2.low / scales.collision_mps2,
    }
    if low_speed_g_T0_K is not None:
        terms["low_speed_thermal_state"] = low_speed_g_T0_K.low / scales.low_speed_temperature_K
    critical = min(terms, key=terms.get)
    return terms[critical], critical


def _lower_margin(uncertainty: Mapping[str, float], key: str) -> float:
    value = float(uncertainty.get(key, 0.0))
    if value < 0.0:
        raise ValueError(f"{key} must be nonnegative")
    return value


def compute_effective_friction_interval(
    chi: AugmentedHybridState,
    params: ReserveParameters,
    uncertainty: Mapping[str, float] | None = None,
) -> tuple[float, float, str]:
    uncertainty = uncertainty or {}
    p = params.vehicle
    candidates = {
        "command_envelope": min(p.friction_command_limit_N, params.friction_envelope_N),
        "fade_cbf": params.fade_cbf_limit_N,
    }
    if chi.speed_mps <= p.low_speed_threshold_mps:
        candidates["low_speed_zoh"] = params.low_speed_zoh_limit_N
    else:
        candidates["thermal_hocbf"] = params.thermal_hocbf_limit_N
    name = min(candidates, key=candidates.get)
    upper = candidates[name] - _lower_margin(uncertainty, "friction_upper_margin_N")
    lower = params.friction_lower_N
    return lower, upper, name


def compute_effective_aux_interval(
    chi: AugmentedHybridState,
    params: ReserveParameters,
    uncertainty: Mapping[str, float] | None = None,
) -> tuple[float, float, str]:
    uncertainty = uncertainty or {}
    p = params.vehicle
    candidates = {
        "command_envelope": min(p.auxiliary_command_limit_N, params.auxiliary_envelope_N),
        "gear_power": params.gear_power_limit_N,
        "realized_force_cbf": params.realized_auxiliary_cbf_limit_N,
        "mode_limit": params.mode_auxiliary_limit_N,
    }
    name = min(candidates, key=candidates.get)
    upper = candidates[name] - _lower_margin(uncertainty, "auxiliary_upper_margin_N")
    lower = params.auxiliary_lower_N
    return lower, upper, name


def compute_instantaneous_reserve(
    chi: AugmentedHybridState,
    params: ReserveParameters,
    uncertainty: Mapping[str, float] | None = None,
) -> InstantaneousReserveResult:
    """Return the exact interval-box optimum and the full feasibility flag."""
    params.validate()
    uncertainty = uncertainty or {}
    L_f, U_f, active_f = compute_effective_friction_interval(chi, params, uncertainty)
    L_a, U_a, active_a = compute_effective_aux_interval(chi, params, uncertainty)
    D_c = (
        params.gamma_c_mps2
        + params.mu_c_mps2
        - params.Xi_hat_c_mps2
        - params.alpha2_psi1_c_mps2
        + _lower_margin(uncertainty, "collision_demand_margin_mps2")
    )
    friction_ok = L_f <= U_f
    auxiliary_ok = L_a <= U_a
    low_speed_active = chi.speed_mps <= params.vehicle.low_speed_threshold_mps
    g_T0 = params.low_speed_zoh_g_T0_K if low_speed_active else None
    low_speed_thermal_ok = not low_speed_active or g_T0 >= 0.0
    M = params.A_f * U_f + params.A_a * U_a - D_c
    feasible = friction_ok and auxiliary_ok and M >= 0.0 and low_speed_thermal_ok
    return InstantaneousReserveResult(
        L_f=L_f,
        U_f=U_f,
        L_a=L_a,
        U_a=U_a,
        D_c=D_c,
        M=M,
        friction_interval_nonempty=friction_ok,
        auxiliary_interval_nonempty=auxiliary_ok,
        feasible=feasible,
        active_friction_limit=active_f,
        active_auxiliary_limit=active_a,
        low_speed_active=low_speed_active,
        g_T0_K=g_T0,
        low_speed_thermal_feasible=low_speed_thermal_ok,
    )


def compute_complete_certificate(
    result: InstantaneousReserveResult,
    scales: CertificateScales,
) -> CompleteCertificateResult:
    """Exact signed certificate for two intervals plus one increasing row.

    ``M`` remains the physical collision demand/supply reserve even when an
    individual interval is empty.  The minimum normalized gap ``rho`` closes
    that logical gap.  In the low-speed branch the command-independent reserve
    ``g_T0`` is also included, so rho >= 0 iff the complete hard command set is
    nonempty even when the ZOH command coefficient is zero.
    """
    scales.validate()
    delta_f = result.U_f - result.L_f
    delta_a = result.U_a - result.L_a
    terms = {
        "friction_interval": delta_f / scales.friction_force_N,
        "auxiliary_interval": delta_a / scales.auxiliary_force_N,
        "collision_demand": result.M / scales.collision_mps2,
    }
    normalized_g_T0 = None
    if result.low_speed_active:
        if result.g_T0_K is None:
            raise ValueError("low-speed certificate requires g_T0")
        normalized_g_T0 = result.g_T0_K / scales.low_speed_temperature_K
        terms["low_speed_thermal_state"] = normalized_g_T0
    critical = min(terms, key=terms.get)
    rho = terms[critical]
    return CompleteCertificateResult(
        delta_f_N=delta_f,
        delta_a_N=delta_a,
        M_mps2=result.M,
        normalized_friction=terms["friction_interval"],
        normalized_auxiliary=terms["auxiliary_interval"],
        normalized_collision=terms["collision_demand"],
        rho=rho,
        feasible=rho >= 0.0,
        critical_limit_type=critical,
        g_T0_K=result.g_T0_K,
        normalized_low_speed_thermal=normalized_g_T0,
    )


def normalization_sensitivity(
    result: InstantaneousReserveResult,
    candidate_scales: Sequence[CertificateScales],
) -> tuple[CompleteCertificateResult, ...]:
    """Audit scale sensitivity without changing algorithms or physical margins."""
    if not candidate_scales:
        raise ValueError("at least one scale triple is required")
    return tuple(compute_complete_certificate(result, scales) for scales in candidate_scales)


def independent_box_lp_feasible(result: InstantaneousReserveResult, A_f: float, A_a: float) -> bool:
    """Independent vertex-enumeration LP decision used by the 10k-case test."""
    if result.L_f > result.U_f or result.L_a > result.U_a:
        return False
    if result.low_speed_active and not result.low_speed_thermal_feasible:
        return False
    vertices: Sequence[tuple[float, float]] = (
        (result.L_f, result.L_a),
        (result.L_f, result.U_a),
        (result.U_f, result.L_a),
        (result.U_f, result.U_a),
    )
    return any(A_f * u_f + A_a * u_a >= result.D_c for u_f, u_a in vertices)

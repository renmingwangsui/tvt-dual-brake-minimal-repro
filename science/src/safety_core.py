"""Executable equations and hard rows for Conflict-Aware Safe-MAPPO-DiffQP.

SI units are used throughout: m, s, kg, N, K, J.  A hard row is represented
as ``g @ [u_f, u_a] <= h`` and its logged residual is ``h-g@u``.
Nonnegative residual means that the implemented row is satisfied.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import exp
from typing import Callable, Dict, Iterable, List, Mapping, Sequence, Tuple

from simulator.thermal import thermal_rate_Kps
from simulator.truck_dynamics import first_order_lag_rate, longitudinal_acceleration_mps2


REQUIRED_UNCERTAINTY_COMPONENTS = (
    "measurement_position_m",
    "measurement_speed_mps",
    "measurement_acceleration_mps2",
    "predecessor_acceleration_mps2",
    "predecessor_jerk_mps3",
    "message_age_s",
    "packet_delay_s",
    "grade_rad",
    "grade_rate_radps",
    "rolling_force_N",
    "aerodynamic_force_N",
    "heat_residual_W",
    "heat_residual_rate_Wps",
    "tau_f_s",
    "tau_a_s",
)


@dataclass(frozen=True)
class VehicleParams:
    mass_kg: float
    tau_f_s: float
    tau_a_s: float
    heat_capacity_J_per_K: float
    cooling_W_per_K: float
    heat_fraction: float
    friction_cold_limit_N: float
    friction_command_limit_N: float
    auxiliary_command_limit_N: float
    fade_reference_K: float
    fade_slope_per_K: float
    fade_floor: float
    sample_time_s: float
    low_speed_threshold_mps: float
    temperature_buffer_K: float
    alpha_fade_per_s: float
    alpha_aux_per_s: float

    def validate(self) -> None:
        positive = (
            self.mass_kg,
            self.tau_f_s,
            self.tau_a_s,
            self.heat_capacity_J_per_K,
            self.friction_cold_limit_N,
            self.friction_command_limit_N,
            self.sample_time_s,
        )
        if any(value <= 0.0 for value in positive):
            raise ValueError("positive vehicle parameters must be strictly positive")
        if not 0.0 < self.heat_fraction <= 1.0:
            raise ValueError("heat_fraction must lie in (0,1]")
        if not 0.0 < self.fade_floor <= 1.0:
            raise ValueError("fade_floor must lie in (0,1]")


@dataclass(frozen=True)
class State:
    speed_mps: float
    friction_force_N: float
    auxiliary_force_N: float
    temperature_K: float
    acceleration_mps2: float


@dataclass(frozen=True)
class Predecessor:
    speed_mps: float
    acceleration_mps2: float
    jerk_mps3: float


@dataclass(frozen=True)
class Row:
    name: str
    g_f: float
    g_a: float
    h: float
    residual_unit: str

    def residual(self, command: Sequence[float]) -> float:
        return self.h - self.g_f * command[0] - self.g_a * command[1]


@dataclass(frozen=True)
class Interval:
    low: float
    high: float

    def __post_init__(self) -> None:
        if self.low > self.high:
            raise ValueError("invalid interval")

    def __add__(self, other: "Interval | float") -> "Interval":
        other = as_interval(other)
        return Interval(self.low + other.low, self.high + other.high)

    __radd__ = __add__

    def __neg__(self) -> "Interval":
        return Interval(-self.high, -self.low)

    def __sub__(self, other: "Interval | float") -> "Interval":
        return self + (-as_interval(other))

    def __rsub__(self, other: "Interval | float") -> "Interval":
        return as_interval(other) - self

    def __mul__(self, other: "Interval | float") -> "Interval":
        other = as_interval(other)
        values = (
            self.low * other.low,
            self.low * other.high,
            self.high * other.low,
            self.high * other.high,
        )
        return Interval(min(values), max(values))

    __rmul__ = __mul__

    def reciprocal(self) -> "Interval":
        if self.low <= 0.0 <= self.high:
            raise ZeroDivisionError("interval contains zero")
        return Interval(min(1.0 / self.low, 1.0 / self.high), max(1.0 / self.low, 1.0 / self.high))

    def __truediv__(self, other: "Interval | float") -> "Interval":
        return self * as_interval(other).reciprocal()

    def square(self) -> "Interval":
        if self.low <= 0.0 <= self.high:
            return Interval(0.0, max(self.low * self.low, self.high * self.high))
        return Interval(min(self.low * self.low, self.high * self.high), max(self.low * self.low, self.high * self.high))


def as_interval(value: Interval | float) -> Interval:
    return value if isinstance(value, Interval) else Interval(float(value), float(value))


def directional_margin(estimate: float, true_interval: Interval) -> float:
    """Smallest nonnegative gamma satisfying estimate-gamma <= every true value."""
    return max(0.0, estimate - true_interval.low)


def validate_uncertainty_box(box: Mapping[str, Interval]) -> None:
    """Require every deterministic uncertainty component declared in the paper."""
    missing = set(REQUIRED_UNCERTAINTY_COMPONENTS).difference(box)
    if missing:
        raise ValueError("uncertainty box lacks: " + ", ".join(sorted(missing)))


def directional_margin_from_box(
    estimate: float,
    box: Mapping[str, Interval],
    interval_evaluator: Callable[[Mapping[str, Interval]], Interval],
) -> float:
    """Evaluate gamma=max(0, Xi_hat-lower(Xi(B))) with interval arithmetic."""
    validate_uncertainty_box(box)
    return directional_margin(estimate, interval_evaluator(box))


def sampled_data_margin(rate_bound: float, dt_s: float, integration_error: float) -> float:
    if min(rate_bound, dt_s, integration_error) < 0.0:
        raise ValueError("sampled-data bound inputs must be nonnegative")
    return rate_bound * dt_s + integration_error


@dataclass(frozen=True)
class LowSpeedZOHErrorBound:
    state_variation_K: float
    parameter_uncertainty_K: float
    lag_uncertainty_K: float
    integration_error_K: float
    total_K: float


def low_speed_zoh_error_bound(
    p: VehicleParams,
    speed_abs_bound_mps: float,
    friction_force_abs_bound_N: float,
    friction_command_abs_bound_N: float,
    acceleration_abs_bound_mps2: float,
    ambient_rate_abs_bound_Kps: float,
    heat_residual_rate_abs_bound_Wps: float,
    thermal_parameter_relative_uncertainty: float,
    actuator_lag_relative_uncertainty: float,
    integration_error_K: float,
) -> LowSpeedZOHErrorBound:
    """Closed conservative one-step error for the frozen-v/Ta/qw ZOH row.

    It bounds variation of friction heat, ambient cooling, and residual heat;
    then adds registered relative thermal-parameter, lag, and integrator terms.
    The same expression is printed in the manuscript.
    """
    values = (
        speed_abs_bound_mps, friction_force_abs_bound_N,
        friction_command_abs_bound_N, acceleration_abs_bound_mps2,
        ambient_rate_abs_bound_Kps, heat_residual_rate_abs_bound_Wps,
        thermal_parameter_relative_uncertainty,
        actuator_lag_relative_uncertainty, integration_error_K,
    )
    if min(values) < 0.0 or actuator_lag_relative_uncertainty >= 1.0:
        raise ValueError("nonnegative bounds and lag relative uncertainty < 1 required")
    dt = p.sample_time_s
    tau_min = p.tau_f_s * (1.0 - actuator_lag_relative_uncertainty)
    bdot = (friction_command_abs_bound_N + friction_force_abs_bound_N) / tau_min
    heat_rate = p.heat_fraction * (
        friction_force_abs_bound_N * acceleration_abs_bound_mps2
        + speed_abs_bound_mps * bdot
    )
    power_rate = (
        heat_rate
        + p.cooling_W_per_K * ambient_rate_abs_bound_Kps
        + heat_residual_rate_abs_bound_Wps
    )
    state_term = 0.5 * dt * dt * power_rate / p.heat_capacity_J_per_K
    reference_power = (
        p.heat_fraction * friction_force_abs_bound_N * speed_abs_bound_mps
        + p.cooling_W_per_K * ambient_rate_abs_bound_Kps * dt
        + heat_residual_rate_abs_bound_Wps * dt
    )
    parameter_term = dt * thermal_parameter_relative_uncertainty * reference_power / p.heat_capacity_J_per_K
    lag_term = (
        0.5 * dt * dt * p.heat_fraction * speed_abs_bound_mps
        * (friction_command_abs_bound_N + friction_force_abs_bound_N)
        * actuator_lag_relative_uncertainty / (p.heat_capacity_J_per_K * tau_min)
    )
    total = state_term + parameter_term + lag_term + integration_error_K
    return LowSpeedZOHErrorBound(state_term, parameter_term, lag_term, integration_error_K, total)


@dataclass(frozen=True)
class DigitalRateBounds:
    collision: float
    thermal: float
    fade: float
    auxiliary: float


def affine_lhs_rate_bound(
    coefficient_abs: Sequence[float],
    coefficient_rate_abs: Sequence[float],
    command_abs: Sequence[float],
    uncontrolled_rate_abs: float,
) -> float:
    """Rate bound for ``a(t)^T u_k + xi(t)`` during one ZOH interval.

    The newly selected command is constant between samples, so ``u_dot=0`` on
    the open hold interval.  Command jumps are checked by the QP at the sample
    instant and do not require an independently calibrated command-slew limit.
    """
    if not (len(coefficient_abs) == len(coefficient_rate_abs) == len(command_abs)):
        raise ValueError("rate-bound vectors must have equal length")
    if min((*coefficient_abs, *coefficient_rate_abs, *command_abs, uncontrolled_rate_abs)) < 0.0:
        raise ValueError("rate-bound components must be nonnegative")
    return uncontrolled_rate_abs + sum(
        adot * u
        for adot, u in zip(coefficient_rate_abs, command_abs)
    )


def compute_digital_rate_bounds(components: Mapping[str, Mapping[str, Sequence[float] | float]]) -> DigitalRateBounds:
    """Compute all four barrier LHS rate bounds from registered components."""
    def one(name: str) -> float:
        c = components[name]
        return affine_lhs_rate_bound(
            c["coefficient_abs"], c["coefficient_rate_abs"], c["command_abs"],
            float(c["uncontrolled_rate_abs"]),
        )
    return DigitalRateBounds(one("collision"), one("thermal"), one("fade"), one("auxiliary"))


def fade_factor(temperature_K: float, p: VehicleParams) -> float:
    return max(p.fade_floor, 1.0 - p.fade_slope_per_K * (temperature_K - p.fade_reference_K))


def fade_derivative(temperature_K: float, p: VehicleParams) -> float:
    raw = 1.0 - p.fade_slope_per_K * (temperature_K - p.fade_reference_K)
    return -p.fade_slope_per_K if raw > p.fade_floor else 0.0


def state_derivatives(
    state: State,
    command: Sequence[float],
    p: VehicleParams,
    nonbraking_force_N: float,
    ambient_temperature_K: float,
    heat_residual_W: float,
) -> Dict[str, float]:
    u_f, u_a = command
    acceleration = longitudinal_acceleration_mps2(
        nonbraking_force_N,
        state.friction_force_N,
        state.auxiliary_force_N,
        p.mass_kg,
    )
    return {
        "v_dot": acceleration,
        "b_dot": first_order_lag_rate(state.friction_force_N, u_f, p.tau_f_s),
        "r_dot": first_order_lag_rate(state.auxiliary_force_N, u_a, p.tau_a_s),
        "T_dot": thermal_rate_Kps(
            state.temperature_K,
            ambient_temperature_K,
            state.friction_force_N,
            state.speed_mps,
            heat_residual_W,
            p.heat_capacity_J_per_K,
            p.cooling_W_per_K,
            p.heat_fraction,
        ),
    }


def continuous_jerk(
    state: State, command: Sequence[float], p: VehicleParams, nonbraking_force_rate_Nps: float
) -> float:
    u_f, u_a = command
    b_dot = first_order_lag_rate(state.friction_force_N, u_f, p.tau_f_s)
    r_dot = first_order_lag_rate(state.auxiliary_force_N, u_a, p.tau_a_s)
    return (
        nonbraking_force_rate_Nps - b_dot - r_dot
    ) / p.mass_kg


def collision_affine(
    state: State,
    predecessor: Predecessor,
    p: VehicleParams,
    gap_m: float,
    standstill_gap_m: float,
    headway_plus_lag_s: float,
    follower_safe_decel_mps2: float,
    predecessor_emergency_decel_mps2: float,
    nonbraking_force_rate_Nps: float,
    k1_per_s: float,
    k2_per_s: float,
    gamma_mps2: float = 0.0,
    mu_mps2: float = 0.0,
) -> Dict[str, float]:
    c = headway_plus_lag_s + state.speed_mps / follower_safe_decel_mps2
    rv = predecessor.speed_mps / predecessor_emergency_decel_mps2
    h = (
        gap_m
        - standstill_gap_m
        - headway_plus_lag_s * state.speed_mps
        - state.speed_mps**2 / (2.0 * follower_safe_decel_mps2)
        + predecessor.speed_mps**2 / (2.0 * predecessor_emergency_decel_mps2)
    )
    h_dot = (
        predecessor.speed_mps
        - state.speed_mps
        - c * state.acceleration_mps2
        + rv * predecessor.acceleration_mps2
    )
    xi_second = (
        predecessor.acceleration_mps2
        - state.acceleration_mps2
        - state.acceleration_mps2**2 / follower_safe_decel_mps2
        - c
        * (
            nonbraking_force_rate_Nps / p.mass_kg
            + state.friction_force_N / (p.mass_kg * p.tau_f_s)
            + state.auxiliary_force_N / (p.mass_kg * p.tau_a_s)
        )
        + predecessor.acceleration_mps2**2 / predecessor_emergency_decel_mps2
        + rv * predecessor.jerk_mps3
    )
    uncontrolled = xi_second + (k1_per_s + k2_per_s) * h_dot + k1_per_s * k2_per_s * h
    a_f = c / (p.mass_kg * p.tau_f_s)
    a_a = c / (p.mass_kg * p.tau_a_s)
    demand = gamma_mps2 + mu_mps2 - uncontrolled
    return {"h": h, "h_dot": h_dot, "uncontrolled": uncontrolled, "A_f": a_f, "A_a": a_a, "D_c": demand}


def thermal_affine(
    state: State,
    p: VehicleParams,
    critical_temperature_K: float,
    temperature_rate_Kps: float,
    ambient_temperature_rate_Kps: float,
    heat_residual_rate_Wps: float,
    k1_per_s: float,
    k2_per_s: float,
    gamma_Kps2: float = 0.0,
    mu_Kps2: float = 0.0,
) -> Dict[str, float]:
    h = critical_temperature_K - state.temperature_K
    h_dot = -temperature_rate_Kps
    xi_second = (
        p.heat_fraction * state.friction_force_N * state.speed_mps / (p.heat_capacity_J_per_K * p.tau_f_s)
        - p.heat_fraction * state.friction_force_N * state.acceleration_mps2 / p.heat_capacity_J_per_K
        + p.cooling_W_per_K * (temperature_rate_Kps - ambient_temperature_rate_Kps) / p.heat_capacity_J_per_K
        - heat_residual_rate_Wps / p.heat_capacity_J_per_K
    )
    uncontrolled = xi_second + (k1_per_s + k2_per_s) * h_dot + k1_per_s * k2_per_s * h
    b_f = p.heat_fraction * state.speed_mps / (p.heat_capacity_J_per_K * p.tau_f_s)
    rhs = uncontrolled - gamma_Kps2 - mu_Kps2
    return {"h": h, "h_dot": h_dot, "uncontrolled": uncontrolled, "B_f": b_f, "rhs": rhs}


def fade_upper_bound(
    state: State,
    p: VehicleParams,
    temperature_rate_Kps: float,
    gamma_Nps: float = 0.0,
    mu_Nps: float = 0.0,
) -> Dict[str, float]:
    h = p.friction_cold_limit_N * fade_factor(state.temperature_K, p) - state.friction_force_N
    upper = state.friction_force_N + p.tau_f_s * (
        p.friction_cold_limit_N * fade_derivative(state.temperature_K, p) * temperature_rate_Kps
        + p.alpha_fade_per_s * h
        - gamma_Nps
        - mu_Nps
    )
    return {"h": h, "upper_N": upper}


def auxiliary_cbf_upper_bound(
    state: State,
    p: VehicleParams,
    realized_limit_N: float,
    realized_limit_slope_N_per_mps: float,
    gamma_Nps: float = 0.0,
    mu_Nps: float = 0.0,
) -> Dict[str, float]:
    """Fixed-gear or conservative minimum-envelope CBF for h_A=rbar(v)-r."""
    h = realized_limit_N - state.auxiliary_force_N
    upper = state.auxiliary_force_N + p.tau_a_s * (
        realized_limit_slope_N_per_mps * state.acceleration_mps2
        + p.alpha_aux_per_s * h
        - gamma_Nps
        - mu_Nps
    )
    return {"h": h, "upper_N": upper}


def low_speed_temperature_row(
    state: State,
    p: VehicleParams,
    ambient_temperature_K: float,
    heat_residual_W: float,
    critical_temperature_K: float,
    model_error_margin_K: float,
) -> Dict[str, float]:
    """Exact ZOH one-step row c_T*u_f <= rhs under constant v, Ta, and q_w."""
    dt = p.sample_time_s
    lam = p.cooling_W_per_K / p.heat_capacity_J_per_K
    decay = exp(-lam * dt) if lam > 0.0 else 1.0
    i0 = (1.0 - decay) / lam if lam > 1e-12 else dt
    rate_delta = lam - 1.0 / p.tau_f_s
    if abs(rate_delta) < 1e-10:
        i1 = decay * dt
    else:
        i1 = decay * (exp(rate_delta * dt) - 1.0) / rate_delta
    base = (
        ambient_temperature_K
        + decay * (state.temperature_K - ambient_temperature_K)
        + heat_residual_W * i0 / p.heat_capacity_J_per_K
        + p.heat_fraction * state.speed_mps * state.friction_force_N * i1 / p.heat_capacity_J_per_K
    )
    coefficient = p.heat_fraction * state.speed_mps * (i0 - i1) / p.heat_capacity_J_per_K
    limit = critical_temperature_K - p.temperature_buffer_K - model_error_margin_K
    g_T0 = limit - base
    return {
        "coefficient_K_per_N": coefficient,
        "rhs_K": g_T0,
        "g_T0_K": g_T0,
        "base_K": base,
        "predicted_limit_K": limit,
    }


def thermal_constraint_mode(speed_mps: float, p: VehicleParams) -> str:
    """Select the undivided exact-ZOH row at and below the registered threshold."""
    return "zoh" if speed_mps <= p.low_speed_threshold_mps else "hocbf"


def effective_limits(
    p: VehicleParams,
    friction_upper_rows_N: Iterable[float],
    auxiliary_envelope_limit_N: float,
    auxiliary_cbf_limit_N: float,
) -> Dict[str, float]:
    lower_f = 0.0
    upper_f = min(
        p.friction_command_limit_N,
        *friction_upper_rows_N,
    )
    lower_a = 0.0
    upper_a = min(
        p.auxiliary_command_limit_N,
        auxiliary_envelope_limit_N,
        auxiliary_cbf_limit_N,
    )
    return {"L_f": lower_f, "U_f": upper_f, "L_a": lower_a, "U_a": upper_a}


def conflict_reserve(collision: Mapping[str, float], limits: Mapping[str, float]) -> float:
    if limits["L_f"] > limits["U_f"] or limits["L_a"] > limits["U_a"]:
        return float("-inf")
    return collision["A_f"] * limits["U_f"] + collision["A_a"] * limits["U_a"] - collision["D_c"]


def build_hard_rows(
    collision: Mapping[str, float],
    limits: Mapping[str, float],
    thermal: Mapping[str, float] | None,
    low_speed: Mapping[str, float] | None,
) -> List[Row]:
    if thermal is not None and low_speed is not None:
        raise ValueError("thermal HOCBF and low-speed ZOH rows are mutually exclusive")
    rows = [
        Row("friction_lower", -1.0, 0.0, -limits["L_f"], "N"),
        Row("friction_upper", 1.0, 0.0, limits["U_f"], "N"),
        Row("auxiliary_lower", 0.0, -1.0, -limits["L_a"], "N"),
        Row("auxiliary_upper", 0.0, 1.0, limits["U_a"], "N"),
        Row("collision_hocbf", -collision["A_f"], -collision["A_a"], -collision["D_c"], "m/s^2"),
    ]
    if thermal is not None:
        rows.append(Row("thermal_hocbf", thermal["B_f"], 0.0, thermal["rhs"], "K/s^2"))
    if low_speed is not None:
        coefficient = low_speed["coefficient_K_per_N"]
        g_T0 = low_speed.get("g_T0_K", low_speed["rhs_K"])
        name = "thermal_zoh_state" if coefficient == 0.0 else "thermal_zoh"
        rows.append(Row(name, coefficient, 0.0, g_T0, "K"))
    return rows


def all_residuals(rows: Iterable[Row], command: Sequence[float]) -> Dict[str, float]:
    return {row.name: row.residual(command) for row in rows}


def feasible(rows: Iterable[Row], command: Sequence[float], tolerance: float = 1e-8) -> bool:
    return all(row.residual(command) >= -tolerance for row in rows)


def project_weighted_2d(
    nominal: Sequence[float], rows: Sequence[Row], weights: Sequence[float] = (1.0, 1.0)
) -> Tuple[float, float]:
    """Exact candidate enumeration for a strictly convex 2-D diagonal-weight QP."""
    if min(weights) <= 0.0:
        raise ValueError("projection weights must be positive")
    p0, p1 = float(nominal[0]), float(nominal[1])
    candidates: List[Tuple[float, float]] = [(p0, p1)]
    for row in rows:
        denom = row.g_f * row.g_f / weights[0] + row.g_a * row.g_a / weights[1]
        if denom > 1e-18:
            violation = row.g_f * p0 + row.g_a * p1 - row.h
            candidates.append((p0 - violation * row.g_f / (weights[0] * denom), p1 - violation * row.g_a / (weights[1] * denom)))
    for i, first in enumerate(rows):
        for second in rows[i + 1 :]:
            det = first.g_f * second.g_a - first.g_a * second.g_f
            if abs(det) > 1e-18:
                candidates.append(((first.h * second.g_a - first.g_a * second.h) / det, (first.g_f * second.h - first.h * second.g_f) / det))
    valid = [candidate for candidate in candidates if feasible(rows, candidate, tolerance=1e-7)]
    if not valid:
        raise ValueError("hard projection polytope is empty")
    return min(valid, key=lambda u: weights[0] * (u[0] - p0) ** 2 + weights[1] * (u[1] - p1) ** 2)


def certified_backup(collision: Mapping[str, float], limits: Mapping[str, float]) -> Tuple[float, float]:
    """Auxiliary-first certified backup for the interval-plus-collision structure."""
    if conflict_reserve(collision, limits) < 0.0:
        raise ValueError("certificate loss: backup polytope is empty")
    u_a = limits["U_a"]
    required_f = (collision["D_c"] - collision["A_a"] * u_a) / collision["A_f"]
    u_f = max(limits["L_f"], required_f)
    if u_f > limits["U_f"]:
        raise ValueError("certificate loss after numerical backup construction")
    return u_f, u_a

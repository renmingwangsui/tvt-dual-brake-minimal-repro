"""10,000-case regression checks: direct derivatives, paper rows, QP rows."""
from __future__ import annotations

import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from safety_core import (  # noqa: E402
    Predecessor,
    State,
    VehicleParams,
    auxiliary_cbf_upper_bound,
    build_hard_rows,
    collision_affine,
    conflict_reserve,
    continuous_jerk,
    effective_limits,
    fade_derivative,
    fade_factor,
    fade_upper_bound,
    low_speed_temperature_row,
    state_derivatives,
    thermal_affine,
    thermal_constraint_mode,
)


RNG = random.Random(20260916)
N = 10_000


def params() -> VehicleParams:
    return VehicleParams(
        mass_kg=RNG.uniform(12_000.0, 40_000.0),
        tau_f_s=RNG.uniform(0.15, 0.9),
        tau_a_s=RNG.uniform(0.08, 0.6),
        heat_capacity_J_per_K=RNG.uniform(3e5, 1.2e6),
        cooling_W_per_K=RNG.uniform(200.0, 1800.0),
        heat_fraction=RNG.uniform(0.5, 0.95),
        friction_cold_limit_N=180_000.0,
        friction_command_limit_N=180_000.0,
        auxiliary_command_limit_N=120_000.0,
        fade_reference_K=350.0,
        fade_slope_per_K=0.0015,
        fade_floor=0.25,
        sample_time_s=0.05,
        low_speed_threshold_mps=0.5,
        temperature_buffer_K=5.0,
        alpha_fade_per_s=2.0,
        alpha_aux_per_s=3.0,
    )


def state(v: float | None = None) -> State:
    return State(
        speed_mps=RNG.uniform(0.0, 35.0) if v is None else v,
        friction_force_N=RNG.uniform(0.0, 100_000.0),
        auxiliary_force_N=RNG.uniform(0.0, 70_000.0),
        temperature_K=RNG.uniform(350.0, 600.0),
        acceleration_mps2=RNG.uniform(-4.0, 2.0),
    )


def close(a: float, b: float, scale: float = 1.0) -> None:
    tol = 2e-6 * max(scale, abs(a), abs(b), 1.0)
    if abs(a - b) > tol:
        raise AssertionError(f"identity mismatch: {a} vs {b}, tol={tol}")


for _ in range(N):
    p = params()
    s = state(v=RNG.uniform(2.0, 35.0))
    pred = Predecessor(RNG.uniform(2.0, 35.0), RNG.uniform(-5.0, 2.0), RNG.uniform(-5.0, 5.0))
    u = (RNG.uniform(0.0, 170_000.0), RNG.uniform(0.0, 110_000.0))
    dot_fr = RNG.uniform(-80_000.0, 80_000.0)
    a_s = RNG.uniform(2.0, 6.0)
    a_e = RNG.uniform(3.0, 8.0)
    q = RNG.uniform(0.8, 2.5)
    coll = collision_affine(s, pred, p, RNG.uniform(20.0, 150.0), 5.0, q, a_s, a_e, dot_fr, 1.2, 1.6)
    jerk = continuous_jerk(s, u, p, dot_fr)
    c = q + s.speed_mps / a_s
    direct_hddot = (
        pred.acceleration_mps2 - s.acceleration_mps2 - s.acceleration_mps2**2 / a_s
        - c * jerk + pred.acceleration_mps2**2 / a_e
        + pred.speed_mps * pred.jerk_mps3 / a_e
    )
    eps = 1e-6
    def hdot_at(sign):
        vv = s.speed_mps + sign * eps * s.acceleration_mps2
        aa = s.acceleration_mps2 + sign * eps * jerk
        vp = pred.speed_mps + sign * eps * pred.acceleration_mps2
        ap = pred.acceleration_mps2 + sign * eps * pred.jerk_mps3
        cc = q + vv / a_s
        rr = vp / a_e
        return vp - vv - cc * aa + rr * ap
    numerical_hddot = (hdot_at(1.0) - hdot_at(-1.0)) / (2.0 * eps)
    close(direct_hddot, numerical_hddot, scale=20.0)
    affine_hddot = coll["uncontrolled"] - (1.2 + 1.6) * coll["h_dot"] - 1.2 * 1.6 * coll["h"] + coll["A_f"] * u[0] + coll["A_a"] * u[1]
    close(direct_hddot, affine_hddot, scale=20.0)
    limits = {"L_f": 0.0, "U_f": 180_000.0, "L_a": 0.0, "U_a": 120_000.0}
    rows = build_hard_rows(coll, limits, None, None)
    collision_row = next(row for row in rows if row.name == "collision_hocbf")
    close(collision_row.residual(u), coll["A_f"] * u[0] + coll["A_a"] * u[1] - coll["D_c"], scale=20.0)

for _ in range(N):
    p = params()
    s = state(v=RNG.uniform(0.6, 35.0))
    u_f = RNG.uniform(0.0, 170_000.0)
    T_dot = RNG.uniform(-1.0, 4.0)
    Ta_dot = RNG.uniform(-0.05, 0.05)
    q_dot = RNG.uniform(-1000.0, 1000.0)
    thermal = thermal_affine(s, p, 700.0, T_dot, Ta_dot, q_dot, 0.8, 1.1)
    b_dot = (-s.friction_force_N + u_f) / p.tau_f_s
    direct_hddot = (
        -p.heat_fraction * (b_dot * s.speed_mps + s.friction_force_N * s.acceleration_mps2) / p.heat_capacity_J_per_K
        + p.cooling_W_per_K * (T_dot - Ta_dot) / p.heat_capacity_J_per_K
        - q_dot / p.heat_capacity_J_per_K
    )
    eps = 1e-6
    def thermal_hdot_at(sign):
        b = s.friction_force_N + sign * eps * b_dot
        v = s.speed_mps + sign * eps * s.acceleration_mps2
        temp = s.temperature_K + sign * eps * T_dot
        ambient = 300.0 + sign * eps * Ta_dot
        heat = sign * eps * q_dot
        return (
            -p.heat_fraction * b * v
            + p.cooling_W_per_K * (temp - ambient)
            - heat
        ) / p.heat_capacity_J_per_K
    numerical_hddot = (thermal_hdot_at(1.0) - thermal_hdot_at(-1.0)) / (2.0 * eps)
    close(direct_hddot, numerical_hddot, scale=5.0)
    affine_hddot = thermal["uncontrolled"] - (0.8 + 1.1) * thermal["h_dot"] - 0.8 * 1.1 * thermal["h"] - thermal["B_f"] * u_f
    close(direct_hddot, affine_hddot, scale=5.0)

for _ in range(N):
    p = params()
    s = state()
    T_dot = RNG.uniform(-1.0, 4.0)
    gamma = RNG.uniform(0.0, 100.0)
    mu = RNG.uniform(0.0, 100.0)
    fade = fade_upper_bound(s, p, T_dot, gamma, mu)
    u_f = RNG.uniform(0.0, 170_000.0)
    direct = (
        p.friction_cold_limit_N * fade_derivative(s.temperature_K, p) * T_dot
        + (s.friction_force_N - u_f) / p.tau_f_s
        + p.alpha_fade_per_s * fade["h"] - gamma - mu
    )
    close(fade["upper_N"] - u_f, p.tau_f_s * direct, scale=200_000.0)

for _ in range(N):
    p = params()
    s = state()
    limit = RNG.uniform(80_000.0, 130_000.0)
    slope = RNG.uniform(-3000.0, 1000.0)
    gamma = RNG.uniform(0.0, 1000.0)
    aux = auxiliary_cbf_upper_bound(s, p, limit, slope, gamma, 0.0)
    u_a = RNG.uniform(0.0, 120_000.0)
    direct = slope * s.acceleration_mps2 + (s.auxiliary_force_N - u_a) / p.tau_a_s + p.alpha_aux_per_s * aux["h"] - gamma
    close(aux["upper_N"] - u_a, p.tau_a_s * direct, scale=200_000.0)

for _ in range(N):
    p = params()
    s = state(v=RNG.uniform(1.0, 35.0))
    pred = Predecessor(25.0, -1.0, 0.0)
    coll = collision_affine(s, pred, p, 80.0, 5.0, 1.5, 4.0, 6.0, 0.0, 1.0, 1.0)
    f_upper = RNG.uniform(40_000.0, 170_000.0)
    a_upper = RNG.uniform(20_000.0, 110_000.0)
    limits = {"L_f": 0.0, "U_f": f_upper, "L_a": 0.0, "U_a": a_upper}
    margin = conflict_reserve(coll, limits)
    direct = coll["A_f"] * f_upper + coll["A_a"] * a_upper - coll["D_c"]
    close(margin, direct, scale=20.0)

# Boundary tests required by the audit text.
for v in (0.0, 0.25, 0.5, 0.500001):
    p = params()
    object.__setattr__(p, "low_speed_threshold_mps", 0.5)
    s = state(v=v)
    row = low_speed_temperature_row(s, p, 300.0, 0.0, 700.0, 1.0)
    if row["g_T0_K"] != row["rhs_K"]:
        raise AssertionError("g_T0 must be the undivided ZOH right-hand side")
    if v == 0.0 and abs(row["coefficient_K_per_N"]) > 1e-15:
        raise AssertionError("zero-speed thermal row must have zero command coefficient")
    if row["coefficient_K_per_N"] < -1e-15:
        raise AssertionError("low-speed thermal command coefficient has wrong sign")
    expected_mode = "zoh" if v <= 0.5 else "hocbf"
    if thermal_constraint_mode(v, p) != expected_mode:
        raise AssertionError(f"wrong thermal mode at v={v}")

print("PASS: 50,000 randomized derivative/affine/QP identities plus low-speed boundary cases")

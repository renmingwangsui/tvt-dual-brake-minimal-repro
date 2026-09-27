"""Shared continuous friction-brake thermal primitives in SI units."""
from __future__ import annotations

from math import exp, isfinite


def friction_heating_power_W(eta: float, friction_force_N: float, speed_mps: float) -> float:
    """Return the manuscript heat-input term ``eta*b*v``."""
    if not 0.0 < eta <= 1.0:
        raise ValueError("eta must lie in (0, 1]")
    if friction_force_N < 0.0 or speed_mps < 0.0:
        raise ValueError("braking force and forward speed must be nonnegative")
    return eta * friction_force_N * speed_mps


def thermal_rate_Kps(
    temperature_K: float,
    ambient_temperature_K: float,
    friction_force_N: float,
    speed_mps: float,
    residual_heat_W: float,
    heat_capacity_J_per_K: float,
    cooling_W_per_K: float,
    eta: float,
) -> float:
    """Evaluate ``C*T_dot = eta*b*v-k*(T-Ta)+q_w`` exactly once."""
    values = (
        temperature_K,
        ambient_temperature_K,
        residual_heat_W,
        heat_capacity_J_per_K,
        cooling_W_per_K,
    )
    if not all(isfinite(value) for value in values):
        raise ValueError("thermal inputs must be finite")
    if heat_capacity_J_per_K <= 0.0 or cooling_W_per_K < 0.0:
        raise ValueError("thermal capacity must be positive and cooling nonnegative")
    heating_W = friction_heating_power_W(eta, friction_force_N, speed_mps)
    return (
        heating_W
        - cooling_W_per_K * (temperature_K - ambient_temperature_K)
        + residual_heat_W
    ) / heat_capacity_J_per_K


def cooling_only_temperature_K(
    initial_temperature_K: float,
    ambient_temperature_K: float,
    elapsed_s: float,
    heat_capacity_J_per_K: float,
    cooling_W_per_K: float,
) -> float:
    """Analytical solution for constant ambient, zero braking and zero residual heat."""
    if elapsed_s < 0.0 or heat_capacity_J_per_K <= 0.0 or cooling_W_per_K < 0.0:
        raise ValueError("invalid cooling-only solution inputs")
    if cooling_W_per_K == 0.0:
        return initial_temperature_K
    return ambient_temperature_K + (initial_temperature_K - ambient_temperature_K) * exp(
        -cooling_W_per_K * elapsed_s / heat_capacity_J_per_K
    )

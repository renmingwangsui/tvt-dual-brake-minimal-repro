"""Pure validation and conversion functions used by Phase 2M-LIT tests.

These helpers do not supply physical values.  Callers must provide values from
traceable sources, and the provenance registry remains the authority.
"""
from __future__ import annotations

import math
from collections.abc import Sequence


def lb_to_kg(value_lb: float) -> float:
    if value_lb < 0:
        raise ValueError("mass cannot be negative")
    return float(value_lb) * 0.45359237


def percent_grade_to_rad(grade_percent: float) -> float:
    return math.atan(float(grade_percent) / 100.0)


def cda(drag_coefficient: float, frontal_area_m2: float) -> float:
    if drag_coefficient < 0 or frontal_area_m2 <= 0:
        raise ValueError("Cd must be nonnegative and area must be positive")
    return float(drag_coefficient) * float(frontal_area_m2)


def effective_heat_capacity(masses_kg: Sequence[float], heat_capacities_j_per_kg_k: Sequence[float]) -> float:
    """Return sum(m_j*c_p,j), with J/K dimensional meaning."""
    if len(masses_kg) != len(heat_capacities_j_per_kg_k) or not masses_kg:
        raise ValueError("matched nonempty mass and heat-capacity arrays required")
    if any(m <= 0 for m in masses_kg) or any(cp <= 0 for cp in heat_capacities_j_per_kg_k):
        raise ValueError("thermal masses and specific heats must be positive")
    return sum(float(m) * float(cp) for m, cp in zip(masses_kg, heat_capacities_j_per_kg_k))


def torque_to_tire_force(
    torque_nm: float,
    gear_ratio: float,
    final_drive_ratio: float,
    efficiency: float,
    rolling_radius_m: float,
) -> float:
    if torque_nm < 0 or gear_ratio <= 0 or final_drive_ratio <= 0:
        raise ValueError("torque must be nonnegative and ratios positive")
    if not (0 < efficiency <= 1) or rolling_radius_m <= 0:
        raise ValueError("efficiency must be in (0,1] and radius positive")
    return torque_nm * gear_ratio * final_drive_ratio * efficiency / rolling_radius_m


def validate_fade_curve(temperature_k: Sequence[float], factor: Sequence[float]) -> bool:
    if len(temperature_k) != len(factor) or len(temperature_k) < 2:
        return False
    return (
        all(b > a for a, b in zip(temperature_k, temperature_k[1:]))
        and all(0.0 <= value <= 1.0 for value in factor)
        and all(b <= a for a, b in zip(factor, factor[1:]))
    )


def validate_temperature_domain(temperature_k: Sequence[float], required_lower_k: float, required_upper_k: float) -> bool:
    return bool(temperature_k) and min(temperature_k) <= required_lower_k < required_upper_k <= max(temperature_k)


def validate_positive_thermal(capacity_j_per_k: float, cooling_w_per_k: float) -> bool:
    return capacity_j_per_k > 0 and cooling_w_per_k > 0


def validate_time_constant(value_s: float) -> bool:
    return math.isfinite(value_s) and value_s > 0


def enforce_force_power_envelope(force_n: float, speed_mps: float, force_cap_n: float, power_cap_w: float) -> bool:
    if min(force_n, speed_mps, force_cap_n, power_cap_w) < 0:
        return False
    return force_n <= force_cap_n and force_n * speed_mps <= power_cap_w + 1e-9


def interpolate_road_profile(position_m: Sequence[float], grade_rad: Sequence[float], query_m: float) -> float:
    if len(position_m) != len(grade_rad) or len(position_m) < 2:
        raise ValueError("matched road arrays with at least two points required")
    if any(b <= a for a, b in zip(position_m, position_m[1:])):
        raise ValueError("road positions must increase strictly")
    if query_m < position_m[0] or query_m > position_m[-1]:
        raise ValueError("query lies outside the registered road")
    for idx in range(len(position_m) - 1):
        x0, x1 = position_m[idx], position_m[idx + 1]
        if query_m <= x1:
            alpha = (query_m - x0) / (x1 - x0)
            return grade_rad[idx] + alpha * (grade_rad[idx + 1] - grade_rad[idx])
    raise AssertionError("unreachable")


def valid_uncertainty_box(lower: Sequence[float], nominal: Sequence[float], upper: Sequence[float]) -> bool:
    if not (len(lower) == len(nominal) == len(upper)) or not lower:
        return False
    return all(math.isfinite(x) and math.isfinite(n) and math.isfinite(y) and x <= n <= y for x, n, y in zip(lower, nominal, upper))

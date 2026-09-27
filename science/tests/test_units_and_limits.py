"""Dimensional and actuator-limit regression checks."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from safety_core import (  # noqa: E402
    REQUIRED_UNCERTAINTY_COMPONENTS,
    Interval,
    State,
    VehicleParams,
    auxiliary_cbf_upper_bound,
    directional_margin_from_box,
    compute_digital_rate_bounds,
    effective_limits,
    low_speed_zoh_error_bound,
)


# Dimension vector order: kg, m, s, K.
KG = (1, 0, 0, 0)
M = (0, 1, 0, 0)
S = (0, 0, 1, 0)
K = (0, 0, 0, 1)


def add(a, b):
    return tuple(x + y for x, y in zip(a, b))


def sub(a, b):
    return tuple(x - y for x, y in zip(a, b))


force = add(KG, sub(M, (0, 0, 2, 0)))
power = sub(add(force, M), S)
heat_capacity = sub(power, sub(K, S))
assert force == (1, 1, -2, 0)
assert power == (1, 2, -3, 0)
assert heat_capacity == (1, 2, -2, -1)
assert sub(power, heat_capacity) == sub(K, S)  # T_dot
assert sub(force, KG) == sub(M, (0, 0, 2, 0))  # acceleration

p = VehicleParams(20_000.0, 0.4, 0.2, 5e5, 500.0, 0.8, 180_000.0, 170_000.0, 100_000.0, 350.0, 0.001, 0.3, 0.1, 0.5, 5.0, 2.0, 3.0)
s = State(15.0, 40_000.0, 30_000.0, 450.0, -1.0)
aux = auxiliary_cbf_upper_bound(s, p, 90_000.0, -1000.0)
limits = effective_limits(p, [160_000.0, 150_000.0], 95_000.0, aux["upper_N"])
assert limits["L_f"] == 0.0
assert limits["U_f"] == 150_000.0
assert limits["L_a"] == 0.0
assert limits["U_a"] <= 95_000.0

box = {name: Interval(-1.0, 1.0) for name in REQUIRED_UNCERTAINTY_COMPONENTS}
gamma = directional_margin_from_box(2.0, box, lambda values: values["grade_rad"] + Interval(3.0, 4.0))
assert gamma == 0.0
box["grade_rad"] = Interval(-5.0, -4.0)
gamma = directional_margin_from_box(2.0, box, lambda values: values["grade_rad"])
assert gamma == 7.0

err = low_speed_zoh_error_bound(p, 0.5, 50000.0, 60000.0, 3.0, 0.2, 1000.0, 0.05, 0.1, 0.002)
assert err.total_K == err.state_variation_K + err.parameter_uncertainty_K + err.lag_uncertainty_K + err.integration_error_K
assert err.total_K > 0.002
components = {
    name: {
        "coefficient_abs": (2.0, 3.0),
        "coefficient_rate_abs": (0.5, 0.25), "command_abs": (6.0, 7.0),
        "uncontrolled_rate_abs": 8.0,
    }
    for name in ("collision", "thermal", "fade", "auxiliary")
}
rates = compute_digital_rate_bounds(components)
expected_rate = 8.0 + 0.5 * 6.0 + 0.25 * 7.0
assert rates.collision == rates.thermal == rates.fade == rates.auxiliary == expected_rate
print("PASS: units, limits, uncertainty, computed ZOH error, and four digital rate bounds")

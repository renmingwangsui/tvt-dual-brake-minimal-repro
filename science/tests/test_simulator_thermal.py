"""Independent continuous thermal-plant tests."""
from __future__ import annotations

from math import isclose
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from simulator.integrators import integrate_reference  # noqa: E402
from simulator.thermal import cooling_only_temperature_K, friction_heating_power_W, thermal_rate_Kps  # noqa: E402


C = 500_000.0
k = 700.0
eta = 0.8
ambient = 300.0

assert thermal_rate_Kps(400.0, ambient, 0.0, 20.0, 0.0, C, k, eta) < 0.0
assert friction_heating_power_W(eta, 20_000.0, 15.0) > 0.0
heated = thermal_rate_Kps(300.0, ambient, 20_000.0, 15.0, 0.0, C, k, eta)
assert heated > 0.0
assert friction_heating_power_W(eta, 20_000.0, 0.0) == 0.0
zero_speed = thermal_rate_Kps(ambient, ambient, 20_000.0, 0.0, 0.0, C, k, eta)
assert zero_speed == 0.0
assert thermal_rate_Kps(ambient, ambient, 0.0, 0.0, 500.0, C, k, eta) > 0.0
assert thermal_rate_Kps(ambient, ambient, 0.0, 0.0, -500.0, C, k, eta) < 0.0

initial = 430.0
elapsed = 120.0
numerical = integrate_reference(
    lambda _time, state: (thermal_rate_Kps(state[0], ambient, 0.0, 0.0, 0.0, C, k, eta),),
    (initial,),
    0.0,
    elapsed,
    maximum_step_s=2.0,
).state[0]
analytical = cooling_only_temperature_K(initial, ambient, elapsed, C, k)
assert isclose(numerical, analytical, rel_tol=2e-11, abs_tol=2e-9)

print("PASS: heating, cooling, zero-speed power, residual heat, and analytical cooling")

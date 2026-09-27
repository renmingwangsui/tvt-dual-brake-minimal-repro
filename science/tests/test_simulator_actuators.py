"""Friction/auxiliary lags, fade domain, envelope and gear transition tests."""
from __future__ import annotations

from math import isclose
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from simulator.auxiliary_brake import (  # noqa: E402
    SyntheticDebugAuxiliaryEnvelope,
    analytical_auxiliary_response_N,
    auxiliary_actuator_rate,
)
from simulator.fade import synthetic_debug_fade_curve  # noqa: E402
from simulator.gearbox import (  # noqa: E402
    GearState,
    GearboxParameters,
    advance_dwell_timer,
    can_shift,
    shift_gear,
)
from simulator.integrators import integrate_reference  # noqa: E402
from simulator.truck_dynamics import analytical_first_order_response, first_order_lag_rate  # noqa: E402


for initial, command, tau in ((0.0, 40_000.0, 0.4), (12_000.0, 5_000.0, 0.8)):
    elapsed = 1.7
    numerical = integrate_reference(
        lambda _time, state: (first_order_lag_rate(state[0], command, tau),),
        (initial,),
        0.0,
        elapsed,
        maximum_step_s=0.1,
    ).state[0]
    analytical = analytical_first_order_response(initial, command, elapsed, tau)
    assert isclose(numerical, analytical, rel_tol=2e-10, abs_tol=1e-7)

initial, command, tau, elapsed = 2_000.0, 30_000.0, 0.6, 2.0
numerical_aux = integrate_reference(
    lambda _time, state: (auxiliary_actuator_rate(state[0], command, tau),),
    (initial,),
    0.0,
    elapsed,
    maximum_step_s=0.1,
).state[0]
assert isclose(
    numerical_aux,
    analytical_auxiliary_response_N(initial, command, elapsed, tau),
    rel_tol=2e-10,
    abs_tol=1e-7,
)

envelope = SyntheticDebugAuxiliaryEnvelope(
    speed_points_mps=(0.0, 5.0, 15.0, 30.0),
    availability_factors=(0.0, 0.4, 1.0, 0.8),
    gear_force_limits_N={1: 20_000.0, 2: 40_000.0, 3: 70_000.0},
    power_limit_W=600_000.0,
)
assert envelope.force_limit_N(0.0, 2) == 0.0
assert envelope.force_limit_N(10.0, 3) > envelope.force_limit_N(10.0, 1)
assert envelope.force_limit_N(30.0, 3) <= 600_000.0 / 30.0
assert envelope.data_provenance == "synthetic_debug" and not envelope.paper_eligible

gearbox = GearboxParameters(
    gears=(1, 2, 3),
    permitted_shifts=frozenset({(1, 2), (2, 1), (2, 3), (3, 2)}),
    minimum_dwell_s=1.0,
)
state = GearState(2, 0.5)
assert not can_shift(state, 3, gearbox)
state = advance_dwell_timer(state, 0.5, gearbox)
assert can_shift(state, 3, gearbox)
shifted = shift_gear(state, 3, gearbox)
assert shifted == GearState(3, 0.0) and not can_shift(shifted, 1, gearbox)

fade = synthetic_debug_fade_curve()
assert fade.value(450.0) < fade.value(350.0)
assert fade.derivative_per_K(450.0) < 0.0
try:
    fade.value(700.0)
except ValueError:
    pass
else:
    raise AssertionError("fade interface must fail outside its registered domain")
clipped = synthetic_debug_fade_curve("conservative_clip")
assert clipped.value(700.0) == min(clipped.factors)
assert clipped.derivative_per_K(700.0) == 0.0

print("PASS: analytical actuator lags, auxiliary envelope, gear dwell, and fade domain policy")

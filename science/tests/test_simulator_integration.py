"""RK4 correctness, ZOH semantics, convergence, reference integration and DEBUG provenance."""
from __future__ import annotations

from math import exp
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from envs.road_profile import PiecewiseLinearGrade  # noqa: E402
from simulator.integrators import (  # noqa: E402
    SimulationResult,
    integrate_reference,
    integrate_zero_order_hold,
    integrate_zero_order_hold_reference,
    rk4_step,
)
from simulator.truck_dynamics import (  # noqa: E402
    EnvironmentInput,
    VehicleCommand,
    derivative_to_vector,
    load_synthetic_debug_config,
    state_from_vector,
    state_to_vector,
    vehicle_rhs,
)


config = load_synthetic_debug_config(ROOT / "configs" / "debug" / "synthetic_debug.yaml")
assert config.data_provenance == "synthetic_debug" and config.paper_eligible is False

# Classical RK4 has the expected one-step fourth-order polynomial for y'=y.
step = rk4_step(lambda _time, state: state, 0.0, (1.0,), 0.1)[0]
expected_polynomial = 1.0 + 0.1 + 0.1**2 / 2.0 + 0.1**3 / 6.0 + 0.1**4 / 24.0
assert abs(step - expected_polynomial) < 1e-15
assert abs(step - exp(0.1)) < 1e-7

road_data = config.raw["road"]
road = PiecewiseLinearGrade(tuple(road_data["positions_m"]), tuple(road_data["grades_rad"]))
p = config.parameters


def rhs(_time: float, vector: tuple[float, ...], command: tuple[float, float, float]) -> tuple[float, ...]:
    state = state_from_vector(vector)
    vehicle_command = VehicleCommand(*command)
    environment = EnvironmentInput(
        grade_rad=road.grade_rad(state.position_m),
        ambient_temperature_K=300.0,
        residual_heat_W=200.0,
    )
    return derivative_to_vector(vehicle_rhs(state, vehicle_command, environment, p))


scenarios = (
    (
        state_to_vector(state_from_vector((0.0, 18.0, 0.0, 0.0, 350.0))),
        ((16_000.0, 0.0, 0.0), (12_000.0, 0.0, 0.0), (8_000.0, 0.0, 0.0)) * 10,
    ),
    (
        state_to_vector(state_from_vector((90.0, 22.0, 5_000.0, 0.0, 400.0))),
        ((8_000.0, 10_000.0, 0.0), (0.0, 25_000.0, 0.0), (0.0, 35_000.0, 0.0)) * 10,
    ),
    (
        state_to_vector(state_from_vector((190.0, 25.0, 0.0, 5_000.0, 370.0))),
        ((8_000.0, 0.0, 10_000.0), (0.0, 0.0, 20_000.0), (0.0, 15_000.0, 25_000.0)) * 10,
    ),
)
steps = (0.04, 0.02, 0.01)
scenario_errors: list[list[tuple[float, ...]]] = []
for initial, commands in scenarios:
    reference = integrate_zero_order_hold_reference(
        rhs,
        initial,
        commands,
        config.control_step_s,
        data_provenance="synthetic_debug",
        paper_eligible=False,
    )
    one_scenario: list[tuple[float, ...]] = []
    for dt in steps:
        result = integrate_zero_order_hold(
            rhs,
            initial,
            commands,
            config.control_step_s,
            dt,
            data_provenance="synthetic_debug",
            paper_eligible=False,
        )
        one_scenario.append(tuple(
            abs(a - b) for a, b in zip(result.samples[-1].state, reference.samples[-1].state)
        ))
        assert result.data_provenance == "synthetic_debug" and not result.paper_eligible
    scenario_errors.append(one_scenario)

# Report the worst component error across three drive/friction/auxiliary scenarios.
errors = [
    tuple(max(scenario[step_index][component] for scenario in scenario_errors) for component in range(5))
    for step_index in range(len(steps))
]

# Quantitative component tolerances at dt=0.01 s: p, v, b, r, T.
tolerances = (2e-6, 2e-6, 2e-2, 2e-2, 1e-7)
print("CONVERGENCE_WORST_ERRORS_3_SCENARIOS dt/dt2/dt4=" + repr(errors))
assert all(error <= tolerance for error, tolerance in zip(errors[-1], tolerances)), (errors[-1], tolerances)
normalized_worst = [max(error / tolerance for error, tolerance in zip(row, tolerances)) for row in errors]
assert normalized_worst[1] < normalized_worst[0]
assert normalized_worst[2] < normalized_worst[1]
assert all(errors[2][index] < errors[1][index] for index in range(4))

# Independent scalar convergence order and adaptive reference comparison.
scalar_errors = []
for dt in (0.2, 0.1, 0.05):
    y = (1.0,)
    t = 0.0
    while t < 1.0 - 1e-15:
        h = min(dt, 1.0 - t)
        y = rk4_step(lambda _time, state: state, t, y, h)
        t += h
    scalar_errors.append(abs(y[0] - exp(1.0)))
assert scalar_errors[1] < scalar_errors[0] / 10.0
assert scalar_errors[2] < scalar_errors[1] / 10.0
adaptive = integrate_reference(lambda _time, state: state, (1.0,), 0.0, 1.0, maximum_step_s=0.2)
assert abs(adaptive.state[0] - exp(1.0)) < 1e-10

# Every derivative evaluation must observe exactly the command assigned to its control interval.
observed: list[tuple[float, float]] = []


def held_rhs(time_s: float, state: tuple[float, ...], command: float) -> tuple[float, ...]:
    observed.append((time_s, command))
    return (command,)


held = integrate_zero_order_hold(
    held_rhs,
    (0.0,),
    (1.0, 2.0, -1.0),
    0.1,
    0.03,
    data_provenance="synthetic_debug",
    paper_eligible=False,
)
assert abs(held.samples[-1].state[0] - 0.2) < 1e-12
for time_s, command in observed:
    # At a discontinuity RK4 observes the old interval's right limit and the
    # new interval's left limit, but never a command from a nonadjacent interval.
    if abs(time_s - 0.1) < 1e-14:
        allowed = {1.0, 2.0}
    elif abs(time_s - 0.2) < 1e-14:
        allowed = {2.0, -1.0}
    elif time_s < 0.1:
        allowed = {1.0}
    elif time_s < 0.2:
        allowed = {2.0}
    else:
        allowed = {-1.0}
    assert command in allowed, (time_s, command, allowed)

try:
    SimulationResult((), "synthetic_debug", True)
except ValueError:
    pass
else:
    raise AssertionError("DEBUG results must never be marked paper eligible")

print("PASS: RK4, adaptive reference, ZOH command holding, convergence, and DEBUG provenance")

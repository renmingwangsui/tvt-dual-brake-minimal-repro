"""Fixed-step RK4, adaptive Dormand--Prince reference integration, and ZOH rollout."""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Callable, Sequence, TypeVar


Vector = tuple[float, ...]
CommandT = TypeVar("CommandT")
Rhs = Callable[[float, Vector], Sequence[float]]
CommandRhs = Callable[[float, Vector, CommandT], Sequence[float]]


def _combine(y: Vector, h: float, terms: Sequence[tuple[float, Sequence[float]]]) -> Vector:
    return tuple(
        value + h * sum(coefficient * vector[index] for coefficient, vector in terms)
        for index, value in enumerate(y)
    )


def _as_vector(values: Sequence[float], expected: int) -> Vector:
    vector = tuple(float(value) for value in values)
    if len(vector) != expected or not all(isfinite(value) for value in vector):
        raise ValueError("RHS returned an invalid state derivative")
    return vector


def rk4_step(rhs: Rhs, time_s: float, state: Sequence[float], step_s: float) -> Vector:
    """One classical fourth-order Runge--Kutta step."""
    if step_s <= 0.0:
        raise ValueError("RK4 step must be positive")
    y = tuple(float(value) for value in state)
    n = len(y)
    k1 = _as_vector(rhs(time_s, y), n)
    k2 = _as_vector(rhs(time_s + step_s / 2.0, _combine(y, step_s, ((0.5, k1),))), n)
    k3 = _as_vector(rhs(time_s + step_s / 2.0, _combine(y, step_s, ((0.5, k2),))), n)
    k4 = _as_vector(rhs(time_s + step_s, _combine(y, step_s, ((1.0, k3),))), n)
    return _combine(y, step_s, ((1.0 / 6.0, k1), (1.0 / 3.0, k2), (1.0 / 3.0, k3), (1.0 / 6.0, k4)))


@dataclass(frozen=True)
class ReferenceIntegrationResult:
    state: Vector
    accepted_steps: int
    rejected_steps: int


def integrate_reference(
    rhs: Rhs,
    initial_state: Sequence[float],
    start_time_s: float,
    end_time_s: float,
    *,
    relative_tolerance: float = 1e-11,
    absolute_tolerance: float = 1e-12,
    maximum_step_s: float | None = None,
) -> ReferenceIntegrationResult:
    """Adaptive Dormand--Prince 5(4) integration used only as a validation reference."""
    if end_time_s < start_time_s or relative_tolerance <= 0.0 or absolute_tolerance <= 0.0:
        raise ValueError("invalid reference integration interval or tolerance")
    y = tuple(float(value) for value in initial_state)
    n = len(y)
    if end_time_s == start_time_s:
        return ReferenceIntegrationResult(y, 0, 0)
    interval = end_time_s - start_time_s
    max_step = interval if maximum_step_s is None else maximum_step_s
    if max_step <= 0.0:
        raise ValueError("maximum reference step must be positive")
    h = min(max_step, interval / 10.0)
    t = start_time_s
    accepted = rejected = 0
    while t < end_time_s:
        if end_time_s - t <= 1e-14 * max(1.0, abs(end_time_s)):
            t = end_time_s
            break
        h = min(h, end_time_s - t, max_step)
        k1 = _as_vector(rhs(t, y), n)
        k2 = _as_vector(rhs(t + h / 5.0, _combine(y, h, ((1.0 / 5.0, k1),))), n)
        k3 = _as_vector(rhs(t + 3.0 * h / 10.0, _combine(y, h, ((3.0 / 40.0, k1), (9.0 / 40.0, k2)))), n)
        k4 = _as_vector(rhs(t + 4.0 * h / 5.0, _combine(y, h, ((44.0 / 45.0, k1), (-56.0 / 15.0, k2), (32.0 / 9.0, k3)))), n)
        k5 = _as_vector(rhs(t + 8.0 * h / 9.0, _combine(y, h, ((19372.0 / 6561.0, k1), (-25360.0 / 2187.0, k2), (64448.0 / 6561.0, k3), (-212.0 / 729.0, k4)))), n)
        k6 = _as_vector(rhs(t + h, _combine(y, h, ((9017.0 / 3168.0, k1), (-355.0 / 33.0, k2), (46732.0 / 5247.0, k3), (49.0 / 176.0, k4), (-5103.0 / 18656.0, k5)))), n)
        fifth = _combine(y, h, ((35.0 / 384.0, k1), (500.0 / 1113.0, k3), (125.0 / 192.0, k4), (-2187.0 / 6784.0, k5), (11.0 / 84.0, k6)))
        k7 = _as_vector(rhs(t + h, fifth), n)
        fourth = _combine(y, h, ((5179.0 / 57600.0, k1), (7571.0 / 16695.0, k3), (393.0 / 640.0, k4), (-92097.0 / 339200.0, k5), (187.0 / 2100.0, k6), (1.0 / 40.0, k7)))
        error_ratio = max(
            abs(a - b) / (absolute_tolerance + relative_tolerance * max(abs(old), abs(a)))
            for old, a, b in zip(y, fifth, fourth)
        )
        if error_ratio <= 1.0:
            t += h
            y = fifth
            accepted += 1
        else:
            rejected += 1
        factor = 5.0 if error_ratio == 0.0 else max(0.2, min(5.0, 0.9 * error_ratio ** (-0.2)))
        h *= factor
        if h < 1e-14 * max(1.0, abs(t)):
            raise RuntimeError("reference integrator step underflow")
    return ReferenceIntegrationResult(y, accepted, rejected)


@dataclass(frozen=True)
class TrajectorySample:
    time_s: float
    state: Vector
    command_interval: int


@dataclass(frozen=True)
class SimulationResult:
    samples: tuple[TrajectorySample, ...]
    data_provenance: str
    paper_eligible: bool

    def __post_init__(self) -> None:
        if self.data_provenance == "synthetic_debug" and self.paper_eligible:
            raise ValueError("synthetic-debug results cannot be marked paper eligible")


def integrate_zero_order_hold(
    rhs: CommandRhs[CommandT],
    initial_state: Sequence[float],
    commands: Sequence[CommandT],
    control_step_s: float,
    simulation_step_s: float,
    *,
    data_provenance: str,
    paper_eligible: bool,
) -> SimulationResult:
    """Integrate with exact command changes only at controller boundaries."""
    if control_step_s <= 0.0 or simulation_step_s <= 0.0 or simulation_step_s > control_step_s:
        raise ValueError("require 0 < simulation_step_s <= control_step_s")
    if not commands:
        raise ValueError("at least one held command is required")
    y = tuple(float(value) for value in initial_state)
    t = 0.0
    samples = [TrajectorySample(t, y, 0)]
    for interval, command in enumerate(commands):
        end = (interval + 1) * control_step_s
        while t < end - 1e-15:
            h = min(simulation_step_s, end - t)
            y = rk4_step(lambda time, state: rhs(time, state, command), t, y, h)
            t = min(end, t + h)
            samples.append(TrajectorySample(t, y, interval))
    return SimulationResult(tuple(samples), data_provenance, paper_eligible)


def integrate_zero_order_hold_reference(
    rhs: CommandRhs[CommandT],
    initial_state: Sequence[float],
    commands: Sequence[CommandT],
    control_step_s: float,
    *,
    data_provenance: str,
    paper_eligible: bool,
) -> SimulationResult:
    """High-accuracy reference rollout with the same controller boundaries."""
    if control_step_s <= 0.0 or not commands:
        raise ValueError("positive control step and commands are required")
    y = tuple(float(value) for value in initial_state)
    samples = [TrajectorySample(0.0, y, 0)]
    for interval, command in enumerate(commands):
        start = interval * control_step_s
        end = (interval + 1) * control_step_s
        result = integrate_reference(
            lambda time, state: rhs(time, state, command),
            y,
            start,
            end,
            relative_tolerance=1e-12,
            absolute_tolerance=1e-13,
            maximum_step_s=control_step_s / 10.0,
        )
        y = result.state
        samples.append(TrajectorySample(end, y, interval))
    return SimulationResult(tuple(samples), data_provenance, paper_eligible)

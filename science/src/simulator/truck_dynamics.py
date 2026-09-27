"""Production simulator state types and the single shared continuous vehicle RHS."""
from __future__ import annotations

from dataclasses import dataclass
from bisect import bisect_right
import json
from math import cos, isfinite, sin
from pathlib import Path
from typing import Any, Mapping

from .thermal import thermal_rate_Kps


@dataclass(frozen=True)
class VehicleParameters:
    """Typed physical parameters; production instances must come from a validated manifest."""

    mass_kg: float
    length_m: float
    rolling_resistance_coefficient: float
    drag_area_m2: float
    tau_f_s: float
    tau_a_s: float
    thermal_capacity_J_per_K: float
    cooling_W_per_K: float
    heat_fraction: float
    friction_force_limit_N: float
    auxiliary_force_limit_N: float
    critical_temperature_K: float
    air_density_kg_per_m3: float = 1.225
    gravity_mps2: float = 9.80665
    auxiliary_speed_points_mps: tuple[float, ...] = ()
    auxiliary_force_points_N: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        positive = (
            self.mass_kg,
            self.length_m,
            self.drag_area_m2,
            self.tau_f_s,
            self.tau_a_s,
            self.thermal_capacity_J_per_K,
            self.friction_force_limit_N,
            self.auxiliary_force_limit_N,
            self.critical_temperature_K,
            self.air_density_kg_per_m3,
            self.gravity_mps2,
        )
        if not all(isfinite(value) for value in positive) or any(value <= 0.0 for value in positive):
            raise ValueError("strictly positive vehicle parameters must be finite")
        if self.rolling_resistance_coefficient < 0.0 or self.cooling_W_per_K < 0.0:
            raise ValueError("rolling resistance and cooling must be nonnegative")
        if not 0.0 < self.heat_fraction <= 1.0:
            raise ValueError("heat_fraction must lie in (0, 1]")
        if bool(self.auxiliary_speed_points_mps) != bool(self.auxiliary_force_points_N):
            raise ValueError("auxiliary speed and force grids must be provided together")
        if self.auxiliary_speed_points_mps:
            if len(self.auxiliary_speed_points_mps) < 2 or len(self.auxiliary_speed_points_mps) != len(self.auxiliary_force_points_N):
                raise ValueError("auxiliary envelope requires equal-length grids with at least two points")
            if any(b <= a for a, b in zip(self.auxiliary_speed_points_mps, self.auxiliary_speed_points_mps[1:])):
                raise ValueError("auxiliary speed points must be strictly increasing")
            if any(value < 0.0 for value in self.auxiliary_force_points_N):
                raise ValueError("auxiliary force envelope must be nonnegative")


@dataclass(frozen=True)
class VehicleState:
    position_m: float
    speed_mps: float
    friction_force_N: float
    auxiliary_force_N: float
    temperature_K: float

    def __post_init__(self) -> None:
        values = (
            self.position_m,
            self.speed_mps,
            self.friction_force_N,
            self.auxiliary_force_N,
            self.temperature_K,
        )
        if not all(isfinite(value) for value in values):
            raise ValueError("vehicle state must be finite")
        if min(self.speed_mps, self.friction_force_N, self.auxiliary_force_N) < 0.0:
            raise ValueError("forward speed and realized braking forces must be nonnegative")


@dataclass(frozen=True)
class VehicleCommand:
    drive_force_N: float
    friction_command_N: float
    auxiliary_command_N: float


@dataclass(frozen=True)
class EnvironmentInput:
    grade_rad: float
    ambient_temperature_K: float
    residual_heat_W: float = 0.0
    longitudinal_disturbance_N: float = 0.0


@dataclass(frozen=True)
class StateDerivative:
    position_mps: float
    speed_mps2: float
    friction_force_Nps: float
    auxiliary_force_Nps: float
    temperature_Kps: float


@dataclass(frozen=True)
class SimulatorConfig:
    parameters: VehicleParameters
    control_step_s: float
    simulation_step_s: float
    data_provenance: str
    paper_eligible: bool
    raw: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.control_step_s <= 0.0 or self.simulation_step_s <= 0.0:
            raise ValueError("integration steps must be positive")
        if self.simulation_step_s > self.control_step_s:
            raise ValueError("simulation_step_s must not exceed control_step_s")
        if self.data_provenance == "synthetic_debug" and self.paper_eligible:
            raise ValueError("synthetic-debug configurations can never be paper eligible")


def first_order_lag_rate(realized: float, command: float, time_constant_s: float) -> float:
    """Shared actuator primitive ``x_dot=(-x+u)/tau``."""
    if time_constant_s <= 0.0:
        raise ValueError("actuator time constant must be positive")
    return (-realized + command) / time_constant_s


def auxiliary_force_envelope_N(parameters: VehicleParameters, speed_mps: float) -> float:
    """Piecewise-linear registered retarder force envelope in vehicle speed."""
    if speed_mps < 0.0:
        raise ValueError("speed must be nonnegative")
    if not parameters.auxiliary_speed_points_mps:
        return parameters.auxiliary_force_limit_N
    speeds = parameters.auxiliary_speed_points_mps
    forces = parameters.auxiliary_force_points_N
    if speed_mps <= speeds[0]:
        return forces[0]
    if speed_mps >= speeds[-1]:
        return forces[-1]
    index = bisect_right(speeds, speed_mps) - 1
    x0, x1 = speeds[index:index + 2]
    y0, y1 = forces[index:index + 2]
    return y0 + (speed_mps - x0) * (y1 - y0) / (x1 - x0)


def auxiliary_force_envelope_slope_N_per_mps(parameters: VehicleParameters, speed_mps: float) -> float:
    if not parameters.auxiliary_speed_points_mps:
        return 0.0
    speeds = parameters.auxiliary_speed_points_mps
    forces = parameters.auxiliary_force_points_N
    if speed_mps <= speeds[0] or speed_mps >= speeds[-1]:
        return 0.0
    index = bisect_right(speeds, speed_mps) - 1
    return (forces[index + 1] - forces[index]) / (speeds[index + 1] - speeds[index])


def analytical_first_order_response(
    initial: float, command: float, elapsed_s: float, time_constant_s: float
) -> float:
    """Exact constant-command solution of the shared first-order actuator."""
    from math import exp

    if elapsed_s < 0.0 or time_constant_s <= 0.0:
        raise ValueError("elapsed time must be nonnegative and time constant positive")
    return command + (initial - command) * exp(-elapsed_s / time_constant_s)


def longitudinal_acceleration_mps2(
    nonbraking_force_N: float,
    friction_force_N: float,
    auxiliary_force_N: float,
    mass_kg: float,
) -> float:
    """Evaluate ``m*v_dot=F_r-b-r``."""
    if mass_kg <= 0.0:
        raise ValueError("mass must be positive")
    return (nonbraking_force_N - friction_force_N - auxiliary_force_N) / mass_kg


def nonbraking_force_N(
    state: VehicleState,
    command: VehicleCommand,
    environment: EnvironmentInput,
    parameters: VehicleParameters,
) -> float:
    """Evaluate the manuscript force convention for forward position ``p``.

    Positive grade is uphill. Thus both gravity and rolling resistance subtract
    from the positive-forward drive force; a negative downhill grade contributes
    positive forward force through ``-m*g*sin(theta)``.
    """
    return (
        command.drive_force_N
        - parameters.mass_kg * parameters.gravity_mps2 * sin(environment.grade_rad)
        - parameters.mass_kg
        * parameters.gravity_mps2
        * parameters.rolling_resistance_coefficient
        * cos(environment.grade_rad)
        - 0.5
        * parameters.air_density_kg_per_m3
        * parameters.drag_area_m2
        * state.speed_mps**2
        + environment.longitudinal_disturbance_N
    )


def vehicle_rhs(
    state: VehicleState,
    command: VehicleCommand,
    environment: EnvironmentInput,
    parameters: VehicleParameters,
    auxiliary_command_limit_N: float | None = None,
) -> StateDerivative:
    """Single continuous plant RHS used by simulator and reusable safety primitives."""
    if command.friction_command_N < 0.0 or command.auxiliary_command_N < 0.0:
        raise ValueError("braking commands must be nonnegative")
    if command.friction_command_N > parameters.friction_force_limit_N:
        raise ValueError("friction command exceeds configured force limit")
    mapped_auxiliary_limit = auxiliary_force_envelope_N(parameters, state.speed_mps)
    auxiliary_limit = (
        mapped_auxiliary_limit
        if auxiliary_command_limit_N is None
        else min(mapped_auxiliary_limit, auxiliary_command_limit_N)
    )
    if auxiliary_limit < 0.0 or command.auxiliary_command_N > auxiliary_limit:
        raise ValueError("auxiliary command exceeds active speed/gear/power limit")
    force = nonbraking_force_N(state, command, environment, parameters)
    return StateDerivative(
        position_mps=state.speed_mps,
        speed_mps2=longitudinal_acceleration_mps2(
            force, state.friction_force_N, state.auxiliary_force_N, parameters.mass_kg
        ),
        friction_force_Nps=first_order_lag_rate(
            state.friction_force_N, command.friction_command_N, parameters.tau_f_s
        ),
        auxiliary_force_Nps=first_order_lag_rate(
            state.auxiliary_force_N, command.auxiliary_command_N, parameters.tau_a_s
        ),
        temperature_Kps=thermal_rate_Kps(
            state.temperature_K,
            environment.ambient_temperature_K,
            state.friction_force_N,
            state.speed_mps,
            environment.residual_heat_W,
            parameters.thermal_capacity_J_per_K,
            parameters.cooling_W_per_K,
            parameters.heat_fraction,
        ),
    )


def state_to_vector(state: VehicleState) -> tuple[float, ...]:
    return (
        state.position_m,
        state.speed_mps,
        state.friction_force_N,
        state.auxiliary_force_N,
        state.temperature_K,
    )


def state_from_vector(values: tuple[float, ...] | list[float]) -> VehicleState:
    if len(values) != 5:
        raise ValueError("vehicle state vector must contain five entries")
    return VehicleState(*map(float, values))


def derivative_to_vector(derivative: StateDerivative) -> tuple[float, ...]:
    return (
        derivative.position_mps,
        derivative.speed_mps2,
        derivative.friction_force_Nps,
        derivative.auxiliary_force_Nps,
        derivative.temperature_Kps,
    )


def load_synthetic_debug_config(path: Path) -> SimulatorConfig:
    """Load the JSON-compatible YAML debug file and enforce non-paper provenance."""
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("data_provenance") != "synthetic_debug":
        raise ValueError("debug config must declare data_provenance: synthetic_debug")
    if document.get("paper_eligible") is not False:
        raise ValueError("debug config must declare paper_eligible: false")
    vehicle = document["vehicle"]
    parameters = VehicleParameters(**vehicle)
    simulation = document["simulation"]
    return SimulatorConfig(
        parameters=parameters,
        control_step_s=float(simulation["control_step_s"]),
        simulation_step_s=float(simulation["simulation_step_s"]),
        data_provenance="synthetic_debug",
        paper_eligible=False,
        raw=document,
    )

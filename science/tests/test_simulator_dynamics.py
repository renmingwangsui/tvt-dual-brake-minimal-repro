"""Continuous vehicle RHS, sign convention, and shared-primitive regression tests."""
from __future__ import annotations

import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from safety_core import State, VehicleParams, state_derivatives  # noqa: E402
from simulator.truck_dynamics import (  # noqa: E402
    EnvironmentInput,
    VehicleCommand,
    VehicleParameters,
    VehicleState,
    nonbraking_force_N,
    vehicle_rhs,
)


def parameters(rolling: float = 0.0, drag: float = 1e-12) -> VehicleParameters:
    return VehicleParameters(
        mass_kg=20_000.0,
        length_m=10.0,
        rolling_resistance_coefficient=rolling,
        drag_area_m2=drag,
        tau_f_s=0.4,
        tau_a_s=0.6,
        thermal_capacity_J_per_K=500_000.0,
        cooling_W_per_K=700.0,
        heat_fraction=0.8,
        friction_force_limit_N=120_000.0,
        auxiliary_force_limit_N=80_000.0,
        critical_temperature_K=650.0,
    )


p = parameters()
ambient = 300.0
stationary = VehicleState(0.0, 0.0, 0.0, 0.0, ambient)
zero = VehicleCommand(0.0, 0.0, 0.0)
flat = EnvironmentInput(0.0, ambient)
derivative = vehicle_rhs(stationary, zero, flat, p)
assert derivative == type(derivative)(0.0, 0.0, 0.0, 0.0, 0.0)

moving = VehicleState(0.0, 20.0, 0.0, 0.0, ambient)
drive = VehicleCommand(15_000.0, 0.0, 0.0)
assert vehicle_rhs(moving, drive, flat, p).speed_mps2 > 0.0
assert vehicle_rhs(moving, zero, EnvironmentInput(-0.05, ambient), p).speed_mps2 > 0.0
assert vehicle_rhs(moving, zero, EnvironmentInput(0.05, ambient), p).speed_mps2 < 0.0

baseline = vehicle_rhs(moving, drive, flat, p).speed_mps2
friction_state = VehicleState(0.0, 20.0, 8_000.0, 0.0, ambient)
auxiliary_state = VehicleState(0.0, 20.0, 0.0, 8_000.0, ambient)
assert vehicle_rhs(friction_state, drive, flat, p).speed_mps2 < baseline
assert vehicle_rhs(auxiliary_state, drive, flat, p).speed_mps2 < baseline

# Explicit force-sign identity, including rolling resistance and aerodynamic drag.
p_resistive = parameters(rolling=0.01, drag=6.0)
force = nonbraking_force_N(moving, drive, EnvironmentInput(-0.03, ambient, longitudinal_disturbance_N=120.0), p_resistive)
expected = (
    drive.drive_force_N
    - p_resistive.mass_kg * p_resistive.gravity_mps2 * math.sin(-0.03)
    - p_resistive.mass_kg * p_resistive.gravity_mps2 * 0.01 * math.cos(-0.03)
    - 0.5 * p_resistive.air_density_kg_per_m3 * 6.0 * moving.speed_mps**2
    + 120.0
)
assert abs(force - expected) < 1e-10

# Existing safety equations and simulator use the same lag/thermal/acceleration primitives.
safety_p = VehicleParams(
    mass_kg=p.mass_kg,
    tau_f_s=p.tau_f_s,
    tau_a_s=p.tau_a_s,
    heat_capacity_J_per_K=p.thermal_capacity_J_per_K,
    cooling_W_per_K=p.cooling_W_per_K,
    heat_fraction=p.heat_fraction,
    friction_cold_limit_N=p.friction_force_limit_N,
    friction_command_limit_N=p.friction_force_limit_N,
    auxiliary_command_limit_N=p.auxiliary_force_limit_N,
    fade_reference_K=350.0,
    fade_slope_per_K=0.001,
    fade_floor=0.3,
    sample_time_s=0.1,
    low_speed_threshold_mps=0.5,
    temperature_buffer_K=5.0,
    alpha_fade_per_s=1.0,
    alpha_aux_per_s=1.0,
)
shared_state = VehicleState(5.0, 17.0, 7_000.0, 4_000.0, 360.0)
shared_command = VehicleCommand(0.0, 20_000.0, 12_000.0)
environment = EnvironmentInput(0.0, 300.0, 250.0)
fr = 9_000.0
sim = vehicle_rhs(shared_state, shared_command, environment, p)
safety = state_derivatives(
    State(17.0, 7_000.0, 4_000.0, 360.0, 0.0),
    (20_000.0, 12_000.0),
    safety_p,
    fr,
    300.0,
    250.0,
)
assert abs(safety["v_dot"] - (fr - 7_000.0 - 4_000.0) / p.mass_kg) < 1e-15
assert abs(safety["b_dot"] - sim.friction_force_Nps) < 1e-15
assert abs(safety["r_dot"] - sim.auxiliary_force_Nps) < 1e-15
assert abs(safety["T_dot"] - sim.temperature_Kps) < 1e-15

print("PASS: simulator dynamics, force signs, braking effects, and shared RHS primitives")

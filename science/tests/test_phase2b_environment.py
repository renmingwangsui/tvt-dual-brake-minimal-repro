"""Phase 2B indexing, geometry, observations, leaders and synchronous update tests."""
from __future__ import annotations

from dataclasses import fields
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from envs.heavy_platoon_env import HeavyPlatoonEnv  # noqa: E402
from envs.leader_profile import (  # noqa: E402
    ConstantSpeedLeader,
    EmergencyBrakingPulseLeader,
    SmoothSpeedChangeLeader,
)
from envs.observations import LocalActorObservation, spacing_m  # noqa: E402
from envs.scenario import SUPPORTED_CONTROLLED_TRUCK_COUNTS, build_synthetic_debug_scenario  # noqa: E402


for n in SUPPORTED_CONTROLLED_TRUCK_COUNTS:
    scenario = build_synthetic_debug_scenario(n)
    env = HeavyPlatoonEnv(scenario)
    assert env.controlled_truck_count == n
    assert env.total_vehicle_count == n + 1
    assert len(env.states) == len(env.controllers) == len(env.previous_commands) == n
    assert len({parameter.mass_kg for parameter in scenario.parameters}) > 1 if n > 1 else True

assert spacing_m(100.0, 80.0, 10.0) == 10.0
assert spacing_m(100.0, 90.0, 10.0) == 0.0
assert spacing_m(100.0, 95.0, 10.0) == -5.0

local_fields = {item.name for item in fields(LocalActorObservation)}
prohibited = {
    "future_states",
    "true_uncertainty",
    "global_fleet_state",
    "centralized_fleet_minimum",
    "diagnostic_fleet_rho_H_cert",
    "other_vehicle_states",
}
assert local_fields.isdisjoint(prohibited)

constant = ConstantSpeedLeader(5.0, 20.0)
assert constant.state_at(2.0).position_m == 45.0
accelerating = SmoothSpeedChangeLeader(0.0, 20.0, 1.0, 2.0)
decelerating = SmoothSpeedChangeLeader(0.0, 20.0, -1.0, 2.0)
assert accelerating.state_at(1.0).acceleration_mps2 > 0.0
assert decelerating.state_at(1.0).acceleration_mps2 < 0.0
emergency = EmergencyBrakingPulseLeader(0.0, 20.0, 0.5, 1.0, 3.0)
assert emergency.state_at(1.0).acceleration_mps2 < 0.0
assert emergency.jerk_bound_mps3 > 0.0

env = HeavyPlatoonEnv(build_synthetic_debug_scenario(3, "constant_descent"))
initial = tuple(env.states)
result = env.step()
assert set(env.command_snapshot_time_by_vehicle.values()) == {0.0}
first_integrate = next(index for index, item in enumerate(env.execution_trace) if item.startswith("integrate_vehicle:"))
last_compute = max(index for index, item in enumerate(env.execution_trace) if item.startswith("compute_verified_command:"))
assert last_compute < first_integrate
assert tuple(env.states) != initial
assert all(record.augmented_state.position_m == initial[record.vehicle_id - 1].position_m for record in result.logs)
# Realized actuator states remain continuous and are not replaced by commands.
assert any(
    abs(state.friction_force_N - log.final_action[0]) > 1e-6
    or abs(state.auxiliary_force_N - log.final_action[1]) > 1e-6
    for state, log in zip(result.states, result.logs)
)

print("PASS: Phase 2B indexing, geometry, observation separation, leaders, and synchronous update")

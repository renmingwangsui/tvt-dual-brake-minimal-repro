"""Synthetic-debug heterogeneous platoon scenario construction."""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

from envs.leader_profile import (
    ConstantSpeedLeader,
    EmergencyBrakingPulseLeader,
    LeaderProfile,
    SmoothSpeedChangeLeader,
)
from envs.road_profile import ConstantGrade, RoadProfile
from network.v2v_channel import ChannelConfig
from simulator.truck_dynamics import VehicleParameters, VehicleState, load_synthetic_debug_config


SUPPORTED_CONTROLLED_TRUCK_COUNTS = (1, 3, 5, 10, 20, 40)


@dataclass(frozen=True)
class PlatoonScenario:
    controlled_truck_count: int
    parameters: tuple[VehicleParameters, ...]
    initial_states: tuple[VehicleState, ...]
    leader: LeaderProfile
    leader_length_m: float
    road: RoadProfile
    channel_config: ChannelConfig
    control_step_s: float
    simulation_step_s: float
    ambient_temperature_K: float
    residual_heat_W: float
    drive_force_N: tuple[float, ...]
    name: str
    data_provenance: str = "synthetic_debug"
    paper_eligible: bool = False

    def __post_init__(self) -> None:
        n = self.controlled_truck_count
        if n < 1 or len(self.parameters) != n or len(self.initial_states) != n or len(self.drive_force_N) != n:
            raise ValueError("N controlled trucks requires exactly N parameter/state/drive entries")
        if self.data_provenance == "synthetic_debug" and self.paper_eligible:
            raise ValueError("synthetic-debug scenarios cannot be paper eligible")
        if self.paper_eligible and self.data_provenance != "literature_calibrated":
            raise ValueError("paper-eligible scenarios require literature_calibrated provenance")

    @property
    def total_vehicle_count(self) -> int:
        return self.controlled_truck_count + 1


def build_synthetic_debug_scenario(
    controlled_truck_count: int,
    variant: str = "flat_steady",
    channel_config: ChannelConfig | None = None,
) -> PlatoonScenario:
    if controlled_truck_count < 1:
        raise ValueError("at least one controlled truck is required")
    root = Path(__file__).resolve().parents[2]
    config = load_synthetic_debug_config(root / "configs" / "debug" / "synthetic_debug.yaml")
    base = config.parameters
    parameters = tuple(
        replace(
            base,
            mass_kg=base.mass_kg * (0.96 + 0.02 * (index % 5)),
            tau_f_s=base.tau_f_s * (0.9 + 0.05 * (index % 4)),
            tau_a_s=base.tau_a_s * (0.9 + 0.04 * (index % 5)),
            thermal_capacity_J_per_K=base.thermal_capacity_J_per_K * (0.95 + 0.025 * (index % 5)),
        )
        for index in range(controlled_truck_count)
    )
    leader_speed = 0.2 if variant == "low_speed" else 18.0
    if variant == "leader_braking":
        leader: LeaderProfile = EmergencyBrakingPulseLeader(0.0, leader_speed, 0.2, 0.5, 2.0)
    elif variant == "smooth_acceleration":
        leader = SmoothSpeedChangeLeader(0.0, leader_speed, 0.8, 1.0)
    elif variant == "smooth_deceleration":
        leader = SmoothSpeedChangeLeader(0.0, leader_speed, -0.8, 1.0)
    else:
        leader = ConstantSpeedLeader(0.0, leader_speed)
    road: RoadProfile = ConstantGrade(0.0 if variant == "flat_steady" else -0.03)
    gap = 65.0
    leader_length = 5.0
    positions: list[float] = []
    predecessor_position = 0.0
    predecessor_length = leader_length
    for index, param in enumerate(parameters):
        position = predecessor_position - predecessor_length - gap
        positions.append(position)
        predecessor_position = position
        predecessor_length = param.length_m
    initial_states = tuple(
        VehicleState(
            position_m=positions[index],
            speed_mps=leader_speed - 0.1 * (index % 3),
            friction_force_N=0.0,
            auxiliary_force_N=0.0,
            temperature_K=(parameters[index].critical_temperature_K - 8.0 if variant == "hot_brake" else 350.0 + index),
        )
        for index in range(controlled_truck_count)
    )
    drive = tuple(
        param.mass_kg * param.gravity_mps2 * param.rolling_resistance_coefficient
        for param in parameters
    )
    return PlatoonScenario(
        controlled_truck_count=controlled_truck_count,
        parameters=parameters,
        initial_states=initial_states,
        leader=leader,
        leader_length_m=leader_length,
        road=road,
        channel_config=channel_config or ChannelConfig(),
        control_step_s=config.control_step_s,
        simulation_step_s=config.simulation_step_s,
        ambient_temperature_K=300.0,
        residual_heat_W=200.0,
        drive_force_N=drive,
        name=variant,
    )

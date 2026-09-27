"""Canonical augmented state and strictly separated observation interfaces."""
from __future__ import annotations

from dataclasses import dataclass

from network.packet import V2VPacket
from safety.reachable_set import AugmentedHybridState
from safety_core import Interval
from simulator.gearbox import GearState
from simulator.truck_dynamics import VehicleState


@dataclass(frozen=True)
class PredecessorReachableInformation:
    position_m: Interval
    speed_mps: Interval
    acceleration_mps2: Interval
    source_generation_time_s: float
    age_s: float
    valid: bool
    replacement_reason: str | None = None


def expand_predecessor_packet(
    packet: V2VPacket,
    current_time_s: float,
    declared_acceleration_margin_mps2: float,
) -> PredecessorReachableInformation:
    """Causally expand historical motion using declared acceleration/jerk bounds."""
    if declared_acceleration_margin_mps2 < 0.0:
        raise ValueError("acceleration margin must be nonnegative")
    age = packet.age_s(current_time_s)
    jerk = packet.declared_jerk_bound_mps3
    a_low = packet.acceleration_mps2 - declared_acceleration_margin_mps2 - jerk * age
    a_high = packet.acceleration_mps2 + declared_acceleration_margin_mps2 + jerk * age
    speeds = (
        max(0.0, packet.speed_mps + a_low * age),
        max(0.0, packet.speed_mps + a_high * age),
    )
    jerk_position = jerk * age**3 / 6.0
    center_position = packet.position_m + packet.speed_mps * age + 0.5 * packet.acceleration_mps2 * age**2
    acceleration_position = 0.5 * declared_acceleration_margin_mps2 * age**2
    return PredecessorReachableInformation(
        position_m=Interval(center_position - acceleration_position - jerk_position, center_position + acceleration_position + jerk_position),
        speed_mps=Interval(min(speeds), max(speeds)),
        acceleration_mps2=Interval(a_low, a_high),
        source_generation_time_s=packet.generation_time_s,
        age_s=age,
        valid=True,
    )


def conservative_predecessor_replacement(
    sensed_position_m: float,
    sensed_speed_mps: float,
    current_time_s: float,
    acceleration_bound_mps2: float,
    reason: str,
) -> PredecessorReachableInformation:
    if acceleration_bound_mps2 <= 0.0:
        raise ValueError("replacement acceleration bound must be positive")
    return PredecessorReachableInformation(
        position_m=Interval(sensed_position_m, sensed_position_m),
        speed_mps=Interval(sensed_speed_mps, sensed_speed_mps),
        acceleration_mps2=Interval(-acceleration_bound_mps2, acceleration_bound_mps2),
        source_generation_time_s=current_time_s,
        age_s=0.0,
        valid=False,
        replacement_reason=reason,
    )


def spacing_m(predecessor_position_m: float, follower_position_m: float, predecessor_length_m: float) -> float:
    """Bumper spacing ``d_i=p_{i-1}-p_i-L_{i-1}`` for forward coordinates."""
    if predecessor_length_m <= 0.0:
        raise ValueError("predecessor length must be positive")
    return predecessor_position_m - follower_position_m - predecessor_length_m


def build_augmented_state(
    physical_state: VehicleState,
    previous_command: tuple[float, float],
    gear_state: GearState,
    message_age_s: float,
    gap_m: float,
    predecessor_speed_mps: float,
    predecessor_acceleration_mps2: float,
) -> AugmentedHybridState:
    """The repository's only constructor for manuscript controller state chi."""
    if message_age_s < 0.0:
        raise ValueError("message age must be nonnegative")
    return AugmentedHybridState(
        position_m=physical_state.position_m,
        speed_mps=physical_state.speed_mps,
        friction_force_N=physical_state.friction_force_N,
        auxiliary_force_N=physical_state.auxiliary_force_N,
        temperature_K=physical_state.temperature_K,
        previous_friction_command_N=float(previous_command[0]),
        previous_auxiliary_command_N=float(previous_command[1]),
        gear_id=gear_state.gear,
        gear_dwell_time_s=gear_state.dwell_time_s,
        message_age_s=message_age_s,
        gap_m=gap_m,
        predecessor_speed_mps=predecessor_speed_mps,
        predecessor_acceleration_mps2=predecessor_acceleration_mps2,
    )


@dataclass(frozen=True)
class LocalActorObservation:
    own_speed_mps: float
    own_friction_force_N: float
    own_auxiliary_force_N: float
    own_temperature_K: float
    gap_m: float
    relative_speed_mps: float
    previous_friction_command_N: float
    previous_auxiliary_command_N: float
    gear_id: int
    message_age_s: float
    received_safety_intent: float


@dataclass(frozen=True)
class SafetyControllerInformation:
    chi: AugmentedHybridState
    predecessor_reachable: PredecessorReachableInformation
    message_valid: bool
    message_reason: str


@dataclass(frozen=True)
class CentralizedTrainingState:
    leader_state: tuple[float, float, float]
    controlled_states: tuple[VehicleState, ...]
    communication_validity: tuple[bool, ...]
    diagnostic_fleet_rho_H_cert: float | None = None


def local_actor_observation(
    chi: AugmentedHybridState, received_safety_intent: float = 0.0
) -> LocalActorObservation:
    return LocalActorObservation(
        own_speed_mps=chi.speed_mps,
        own_friction_force_N=chi.friction_force_N,
        own_auxiliary_force_N=chi.auxiliary_force_N,
        own_temperature_K=chi.temperature_K,
        gap_m=chi.gap_m,
        relative_speed_mps=chi.predecessor_speed_mps - chi.speed_mps,
        previous_friction_command_N=chi.previous_friction_command_N,
        previous_auxiliary_command_N=chi.previous_auxiliary_command_N,
        gear_id=chi.gear_id,
        message_age_s=chi.message_age_s,
        received_safety_intent=received_safety_intent,
    )

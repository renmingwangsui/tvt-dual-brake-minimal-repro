"""Conservative interval tube for the registered sampled heavy-truck model.

The tube is certified only for the explicit-Euler sampled model implemented
here.  Continuous-time truncation error must be included in the registered
uncertainty intervals before a vehicle-level claim is made.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

from safety_core import Interval, as_interval


@dataclass(frozen=True)
class AugmentedHybridState:
    position_m: float
    speed_mps: float
    friction_force_N: float
    auxiliary_force_N: float
    temperature_K: float
    previous_friction_command_N: float
    previous_auxiliary_command_N: float
    gear_id: int
    gear_dwell_time_s: float
    message_age_s: float
    gap_m: float
    predecessor_speed_mps: float
    predecessor_acceleration_mps2: float


@dataclass(frozen=True)
class IntervalHybridState:
    position_m: Interval
    speed_mps: Interval
    friction_force_N: Interval
    auxiliary_force_N: Interval
    temperature_K: Interval
    previous_friction_command_N: Interval
    previous_auxiliary_command_N: Interval
    gear_id: Interval
    gear_dwell_time_s: Interval
    message_age_s: Interval
    gap_m: Interval
    predecessor_speed_mps: Interval
    predecessor_acceleration_mps2: Interval

    def contains(self, state: AugmentedHybridState, tolerance: float = 1e-10) -> bool:
        pairs = (
            (self.position_m, state.position_m),
            (self.speed_mps, state.speed_mps),
            (self.friction_force_N, state.friction_force_N),
            (self.auxiliary_force_N, state.auxiliary_force_N),
            (self.temperature_K, state.temperature_K),
            (self.previous_friction_command_N, state.previous_friction_command_N),
            (self.previous_auxiliary_command_N, state.previous_auxiliary_command_N),
            (self.gear_id, float(state.gear_id)),
            (self.gear_dwell_time_s, state.gear_dwell_time_s),
            (self.message_age_s, state.message_age_s),
            (self.gap_m, state.gap_m),
            (self.predecessor_speed_mps, state.predecessor_speed_mps),
            (self.predecessor_acceleration_mps2, state.predecessor_acceleration_mps2),
        )
        return all(interval.low - tolerance <= value <= interval.high + tolerance for interval, value in pairs)

    def widths(self) -> dict[str, float]:
        """Per-coordinate interval widths for tube-inflation diagnostics."""
        return {
            name: float(interval.high - interval.low)
            for name, interval in self.__dict__.items()
        }

    def aggregate_width(self) -> float:
        """Dimensioned sum used only as a logged diagnostic, never a metric."""
        return sum(self.widths().values())

    def is_subset_of(self, other: "IntervalHybridState") -> bool:
        return all(
            getattr(other, name).low <= interval.low
            and interval.high <= getattr(other, name).high
            for name, interval in self.__dict__.items()
        )


@dataclass(frozen=True)
class ReachableDynamicsParams:
    sample_time_s: float
    nominal_mass_kg: float
    gravity_mps2: float
    drag_coefficient_N_per_mps2: float
    heat_capacity_J_per_K: float
    cooling_W_per_K: float
    heat_fraction: float
    nominal_tau_f_s: float
    nominal_tau_a_s: float

    def validate(self) -> None:
        if min(
            self.sample_time_s,
            self.nominal_mass_kg,
            self.gravity_mps2,
            self.heat_capacity_J_per_K,
            self.nominal_tau_f_s,
            self.nominal_tau_a_s,
        ) <= 0.0:
            raise ValueError("positive reachable-dynamics parameters required")


@dataclass(frozen=True)
class ReachabilityUncertainty:
    grade_rad: Interval
    predecessor_acceleration_mps2: Interval
    predecessor_jerk_mps3: Interval
    mass_scale: Interval
    drag_scale: Interval
    rolling_force_N: Interval
    tau_f_scale: Interval
    tau_a_scale: Interval
    thermal_gain_scale: Interval
    cooling_scale: Interval
    heat_residual_W: Interval
    ambient_temperature_K: Interval
    communication_delay_s: Interval
    auxiliary_availability_scale: Interval

    def validate(self) -> None:
        positive = (
            self.mass_scale,
            self.tau_f_scale,
            self.tau_a_scale,
            self.thermal_gain_scale,
            self.cooling_scale,
            self.auxiliary_availability_scale,
        )
        if any(interval.low <= 0.0 for interval in positive):
            raise ValueError("multiplicative uncertainty intervals must remain positive")
        if self.communication_delay_s.low < 0.0:
            raise ValueError("communication delay must be nonnegative")

    def is_subset_of(self, other: "ReachabilityUncertainty") -> bool:
        return all(
            getattr(other, name).low <= interval.low
            and interval.high <= getattr(other, name).high
            for name, interval in self.__dict__.items()
        )


def point_interval_state(state: AugmentedHybridState) -> IntervalHybridState:
    return IntervalHybridState(
        **{name: as_interval(float(value)) for name, value in state.__dict__.items()}
    )


def _nonnegative(interval: Interval) -> Interval:
    return Interval(max(0.0, interval.low), max(0.0, interval.high))


def interval_clip(value: Interval, lower: float, upper: float) -> Interval:
    """Inclusion-monotone interval extension of scalar saturation."""
    if lower > upper:
        raise ValueError("invalid saturation bounds")
    return Interval(max(lower, min(upper, value.low)), max(lower, min(upper, value.high)))


def saturated_affine_backup_extension(
    state: IntervalHybridState,
    friction_speed_gain_N_per_mps: float,
    auxiliary_speed_gain_N_per_mps: float,
    friction_bias_N: float,
    auxiliary_bias_N: float,
    friction_bounds_N: tuple[float, float],
    auxiliary_bounds_N: tuple[float, float],
) -> tuple[Interval, Interval]:
    """Sound set extension Kappa_B(X) for an affine-plus-saturation backup.

    This reference policy is intentionally simple. Production policies with
    min/max branches must provide an equally sound interval extension.
    """
    raw_f = friction_bias_N + friction_speed_gain_N_per_mps * state.speed_mps
    raw_a = auxiliary_bias_N + auxiliary_speed_gain_N_per_mps * state.speed_mps
    return (
        interval_clip(raw_f, *friction_bounds_N),
        interval_clip(raw_a, *auxiliary_bounds_N),
    )


def propagate_interval_one_step(
    state: IntervalHybridState,
    control: Sequence[float | Interval],
    params: ReachableDynamicsParams,
    uncertainty: ReachabilityUncertainty,
) -> IntervalHybridState:
    params.validate()
    uncertainty.validate()
    dt = params.sample_time_s
    u_f = as_interval(control[0])
    u_a_requested = as_interval(control[1])
    u_a = u_a_requested * uncertainty.auxiliary_availability_scale
    mass = params.nominal_mass_kg * uncertainty.mass_scale
    tau_f = params.nominal_tau_f_s * uncertainty.tau_f_scale
    tau_a = params.nominal_tau_a_s * uncertainty.tau_a_scale
    drag = params.drag_coefficient_N_per_mps2 * uncertainty.drag_scale * state.speed_mps.square()
    grade_force = mass * params.gravity_mps2 * uncertainty.grade_rad
    acceleration = (
        grade_force
        - drag
        - uncertainty.rolling_force_N
        - state.friction_force_N
        - state.auxiliary_force_N
    ) / mass
    next_position = state.position_m + dt * state.speed_mps
    next_speed = _nonnegative(state.speed_mps + dt * acceleration)
    next_b = _nonnegative(state.friction_force_N + dt * (u_f - state.friction_force_N) / tau_f)
    next_r = _nonnegative(state.auxiliary_force_N + dt * (u_a - state.auxiliary_force_N) / tau_a)
    heat = (
        params.heat_fraction
        * uncertainty.thermal_gain_scale
        * state.friction_force_N
        * state.speed_mps
    )
    cooling = params.cooling_W_per_K * uncertainty.cooling_scale * (
        state.temperature_K - uncertainty.ambient_temperature_K
    )
    next_temperature = state.temperature_K + dt * (
        heat - cooling + uncertainty.heat_residual_W
    ) / params.heat_capacity_J_per_K
    next_pred_speed = _nonnegative(
        state.predecessor_speed_mps + dt * uncertainty.predecessor_acceleration_mps2
    )
    next_pred_acc = (
        uncertainty.predecessor_acceleration_mps2 + dt * uncertainty.predecessor_jerk_mps3
    )
    next_gap = state.gap_m + dt * (state.predecessor_speed_mps - state.speed_mps)
    return IntervalHybridState(
        position_m=next_position,
        speed_mps=next_speed,
        friction_force_N=next_b,
        auxiliary_force_N=next_r,
        temperature_K=next_temperature,
        previous_friction_command_N=u_f,
        previous_auxiliary_command_N=u_a_requested,
        gear_id=state.gear_id,
        gear_dwell_time_s=state.gear_dwell_time_s + dt,
        message_age_s=state.message_age_s + dt + uncertainty.communication_delay_s,
        gap_m=next_gap,
        predecessor_speed_mps=next_pred_speed,
        predecessor_acceleration_mps2=next_pred_acc,
    )


ControlPolicy = Callable[[IntervalHybridState, int], Sequence[float | Interval]]


def propagate_reachable_set(
    chi_k: AugmentedHybridState | IntervalHybridState,
    control_policy: ControlPolicy,
    horizon: int,
    params: ReachableDynamicsParams,
    uncertainty: ReachabilityUncertainty,
) -> list[IntervalHybridState]:
    if horizon < 1:
        raise ValueError("prediction horizon must be at least one")
    current = chi_k if isinstance(chi_k, IntervalHybridState) else point_interval_state(chi_k)
    tube: list[IntervalHybridState] = []
    for step in range(1, horizon + 1):
        command = control_policy(current, step)
        current = propagate_interval_one_step(current, command, params, uncertainty)
        tube.append(current)
    return tube


def tube_width_profile(tube: Sequence[IntervalHybridState]) -> tuple[dict[str, float], ...]:
    return tuple(state.widths() for state in tube)

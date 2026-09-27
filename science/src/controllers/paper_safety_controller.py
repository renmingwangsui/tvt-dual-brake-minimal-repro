"""Literature-calibrated adapter over the frozen audited safety controller.

The certificate, QP, reachable-set, supervisor, and two-pass implementations
remain those in ``integrated_safety_controller``.  This adapter supplies the
frozen Phase 2M-LIT-3 uncertainty box and evaluates the tabulated fade and
retarder envelopes without modifying the hash-locked safety core.
"""
from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass, replace
from time import perf_counter_ns

import controllers.integrated_safety_controller as frozen_controller
from controllers.integrated_safety_controller import (
    ControllerCycleInput,
    IntegratedCycleResult,
    IntegratedSafetyController,
)
from safety.conflict_reserve import CertificateScales
from safety.conflict_reserve import ReserveParameters, compute_complete_certificate, compute_instantaneous_reserve
from safety.reachable_set import ReachabilityUncertainty
from safety.supervisor import SupervisorThresholds
from safety_core import (
    Interval, Predecessor, Row, State, VehicleParams, auxiliary_cbf_upper_bound,
    build_hard_rows, collision_affine, compute_digital_rate_bounds,
    directional_margin, effective_limits, low_speed_temperature_row,
    sampled_data_margin, state_derivatives, thermal_affine,
)


@dataclass(frozen=True)
class PaperSafetyControllerConfig:
    vehicle: VehicleParams
    critical_temperature_K: float
    horizon: int
    scales: CertificateScales
    supervisor_thresholds: SupervisorThresholds
    maximum_message_age_s: float
    fade_temperature_points_K: tuple[float, ...]
    fade_factor_points: tuple[float, ...]
    auxiliary_speed_points_mps: tuple[float, ...]
    auxiliary_force_points_N: tuple[float, ...]
    mass_scale_bounds: tuple[float, float]
    tau_f_scale_bounds: tuple[float, float]
    tau_a_scale_bounds: tuple[float, float]
    thermal_gain_scale_bounds: tuple[float, float]
    cooling_scale_bounds: tuple[float, float]
    residual_heat_bounds_W: tuple[float, float]
    ambient_temperature_bounds_K: tuple[float, float]
    rolling_force_bounds_N: tuple[float, float]
    predecessor_jerk_bound_mps3: float
    follower_safe_deceleration_mps2: float
    predecessor_emergency_deceleration_mps2: float
    headway_plus_lag_s: float
    standstill_gap_m: float
    collision_gains_per_s: tuple[float, float]
    low_speed_model_error_margin_K: float
    auxiliary_availability_bounds: tuple[float, float] = (1.0, 1.0)
    reachability_uncertainty_scale: float = 0.0
    data_provenance: str = "literature_calibrated"
    paper_eligible: bool = True

    def __post_init__(self) -> None:
        if self.horizon < 1 or self.maximum_message_age_s <= 0.0:
            raise ValueError("positive paper horizon and message-age limit required")
        if self.data_provenance != "literature_calibrated" or not self.paper_eligible:
            raise ValueError("paper controller requires literature-calibrated provenance")
        paired = (
            (self.fade_temperature_points_K, self.fade_factor_points),
            (self.auxiliary_speed_points_mps, self.auxiliary_force_points_N),
        )
        if any(len(x) < 2 or len(x) != len(y) for x, y in paired):
            raise ValueError("paper fade and auxiliary maps require aligned grids")
        if any(b <= a for a, b in zip(self.fade_temperature_points_K, self.fade_temperature_points_K[1:])):
            raise ValueError("fade temperatures must increase")
        if any(b > a for a, b in zip(self.fade_factor_points, self.fade_factor_points[1:])):
            raise ValueError("fade capability must be monotone nonincreasing")
        if any(b <= a for a, b in zip(self.auxiliary_speed_points_mps, self.auxiliary_speed_points_mps[1:])):
            raise ValueError("auxiliary speed points must increase")
        for lower, upper in (
            self.mass_scale_bounds, self.tau_f_scale_bounds, self.tau_a_scale_bounds,
            self.thermal_gain_scale_bounds, self.cooling_scale_bounds,
            self.residual_heat_bounds_W, self.ambient_temperature_bounds_K,
            self.rolling_force_bounds_N, self.auxiliary_availability_bounds,
        ):
            if lower > upper:
                raise ValueError("invalid paper uncertainty interval")


def _piecewise_value_and_slope(
    x: float, xs: tuple[float, ...], ys: tuple[float, ...]
) -> tuple[float, float]:
    if x <= xs[0]:
        return ys[0], 0.0
    if x >= xs[-1]:
        return ys[-1], 0.0
    index = bisect_right(xs, x) - 1
    slope = (ys[index + 1] - ys[index]) / (xs[index + 1] - xs[index])
    return ys[index] + slope * (x - xs[index]), slope


class PaperIntegratedSafetyController(IntegratedSafetyController):
    """Frozen controller with literature maps and registered uncertainty."""

    config: PaperSafetyControllerConfig

    def __init__(self, config: PaperSafetyControllerConfig) -> None:
        self.config = config
        self.previous_mode = frozen_controller.SupervisorMode.NORMAL
        self.recovery_counter = 0

    def _uncertainty(self, cycle: ControllerCycleInput) -> ReachabilityUncertainty:
        reachable = cycle.information.predecessor_reachable
        return ReachabilityUncertainty(
            grade_rad=Interval(cycle.grade_rad, cycle.grade_rad),
            predecessor_acceleration_mps2=reachable.acceleration_mps2,
            predecessor_jerk_mps3=Interval(
                -self.config.predecessor_jerk_bound_mps3,
                self.config.predecessor_jerk_bound_mps3,
            ),
            mass_scale=Interval(*self.config.mass_scale_bounds),
            drag_scale=Interval(1.0, 1.0),
            rolling_force_N=Interval(*self.config.rolling_force_bounds_N),
            tau_f_scale=Interval(*self.config.tau_f_scale_bounds),
            tau_a_scale=Interval(*self.config.tau_a_scale_bounds),
            thermal_gain_scale=Interval(*self.config.thermal_gain_scale_bounds),
            cooling_scale=Interval(*self.config.cooling_scale_bounds),
            heat_residual_W=Interval(*self.config.residual_heat_bounds_W),
            ambient_temperature_K=Interval(*self.config.ambient_temperature_bounds_K),
            communication_delay_s=Interval(
                0.0,
                min(cycle.information.chi.message_age_s, self.config.maximum_message_age_s),
            ),
            auxiliary_availability_scale=Interval(*self.config.auxiliary_availability_bounds),
        )

    def _robust_auxiliary_limit(self, speed_mps: float) -> float:
        # Constant lower envelope over a one-sample speed reach.  This retains
        # the frozen map while avoiding differentiation through its corners.
        dv = 4.0 * self.config.vehicle.sample_time_s
        candidates = [max(0.0, speed_mps - dv), speed_mps, speed_mps + dv]
        candidates.extend(
            point for point in self.config.auxiliary_speed_points_mps
            if candidates[0] <= point <= candidates[-1]
        )
        return min(
            _piecewise_value_and_slope(
                value,
                self.config.auxiliary_speed_points_mps,
                self.config.auxiliary_force_points_N,
            )[0]
            for value in candidates
        )

    def _build_context(self, cycle: ControllerCycleInput):
        p = self.config.vehicle
        chi = cycle.information.chi
        acceleration = (
            cycle.nonbraking_force_N - chi.friction_force_N - chi.auxiliary_force_N
        ) / p.mass_kg
        state = State(
            chi.speed_mps, chi.friction_force_N, chi.auxiliary_force_N,
            chi.temperature_K, acceleration,
        )
        predecessor = Predecessor(
            chi.predecessor_speed_mps, chi.predecessor_acceleration_mps2, 0.0
        )
        directional_collision = directional_margin(0.0, Interval(0.0, 0.0))
        k1, k2 = self.config.collision_gains_per_s
        collision = collision_affine(
            state, predecessor, p, chi.gap_m, self.config.standstill_gap_m,
            self.config.headway_plus_lag_s,
            self.config.follower_safe_deceleration_mps2,
            self.config.predecessor_emergency_deceleration_mps2,
            0.0, k1, k2, gamma_mps2=directional_collision,
        )
        zero_rate_components = {
            name: {
                "coefficient_abs": (0.0, 0.0),
                "coefficient_rate_abs": (0.0, 0.0),
                "command_abs": (p.friction_command_limit_N, p.auxiliary_command_limit_N),
                "uncontrolled_rate_abs": 0.0,
            }
            for name in ("collision", "thermal", "fade", "auxiliary")
        }
        digital_rates = compute_digital_rate_bounds(zero_rate_components)
        margins = {
            "directional_collision_mps2": directional_collision,
            "digital_collision": sampled_data_margin(digital_rates.collision, p.sample_time_s, 0.0),
            "digital_thermal": sampled_data_margin(digital_rates.thermal, p.sample_time_s, 0.0),
            "digital_fade": sampled_data_margin(digital_rates.fade, p.sample_time_s, 0.0),
            "digital_auxiliary": sampled_data_margin(digital_rates.auxiliary, p.sample_time_s, 0.0),
        }
        rates = state_derivatives(
            state,
            (chi.previous_friction_command_N, chi.previous_auxiliary_command_N),
            p, cycle.nonbraking_force_N, cycle.ambient_temperature_K,
            cycle.residual_heat_W,
        )
        temperature = cycle.information.chi.temperature_K
        phi, derivative = _piecewise_value_and_slope(
            temperature,
            self.config.fade_temperature_points_K,
            self.config.fade_factor_points,
        )
        fade_h = p.friction_cold_limit_N * phi - state.friction_force_N
        fade_upper = state.friction_force_N + p.tau_f_s * (
            p.friction_cold_limit_N * derivative * rates["T_dot"]
            + p.alpha_fade_per_s * fade_h
        )
        fade = {"h": fade_h, "upper_N": fade_upper}
        auxiliary_limit = min(cycle.auxiliary_limit_N, self._robust_auxiliary_limit(chi.speed_mps))
        auxiliary = auxiliary_cbf_upper_bound(state, p, auxiliary_limit, 0.0)
        low_speed = None
        thermal = None
        if chi.speed_mps <= p.low_speed_threshold_mps:
            low_speed = low_speed_temperature_row(
                state, p, cycle.ambient_temperature_K, cycle.residual_heat_W,
                self.config.critical_temperature_K,
                self.config.low_speed_model_error_margin_K,
            )
            if low_speed["coefficient_K_per_N"] > 0.0:
                thermal_upper = low_speed["g_T0_K"] / low_speed["coefficient_K_per_N"]
            else:
                thermal_upper = p.friction_command_limit_N
        else:
            thermal = thermal_affine(
                state, p, self.config.critical_temperature_K - p.temperature_buffer_K,
                rates["T_dot"], 0.0, 0.0, 0.8, 1.1,
            )
            thermal_upper = thermal["rhs"] / thermal["B_f"] if thermal["B_f"] > 0.0 else p.friction_command_limit_N
        limits = effective_limits(
            p, (thermal_upper, fade["upper_N"]), auxiliary_limit, auxiliary["upper_N"]
        )
        rows = build_hard_rows(collision, limits, thermal, low_speed)
        rows.extend((
            Row("friction_command_envelope", 1.0, 0.0, p.friction_command_limit_N, "N"),
            Row("auxiliary_command_envelope", 0.0, 1.0, auxiliary_limit, "N"),
            Row("fade_cbf", 1.0, 0.0, fade["upper_N"], "N"),
            Row("realized_auxiliary_cbf", 0.0, 1.0, auxiliary["upper_N"], "N"),
        ))
        low_limit = thermal_upper if low_speed is not None else p.friction_command_limit_N
        g_t0 = low_speed["g_T0_K"] if low_speed is not None else 1.0
        reserve_params = ReserveParameters(
            vehicle=p,
            friction_envelope_N=p.friction_command_limit_N,
            auxiliary_envelope_N=auxiliary_limit,
            gear_power_limit_N=auxiliary_limit,
            realized_auxiliary_cbf_limit_N=auxiliary["upper_N"],
            mode_auxiliary_limit_N=auxiliary_limit,
            thermal_hocbf_limit_N=thermal_upper,
            fade_cbf_limit_N=fade["upper_N"],
            low_speed_zoh_limit_N=low_limit,
            low_speed_zoh_g_T0_K=g_t0,
            friction_lower_N=0.0,
            auxiliary_lower_N=0.0,
            A_f=collision["A_f"],
            A_a=collision["A_a"],
            gamma_c_mps2=collision["D_c"],
            mu_c_mps2=0.0,
            Xi_hat_c_mps2=0.0,
            alpha2_psi1_c_mps2=0.0,
        )
        instantaneous = compute_instantaneous_reserve(chi, reserve_params)
        certificate = compute_complete_certificate(instantaneous, self.config.scales)
        return frozen_controller._HardContext(
            state, collision, thermal, low_speed, fade, auxiliary, limits, rows,
            instantaneous, certificate, margins,
        )

    def run_cycle(self, cycle: ControllerCycleInput) -> IntegratedCycleResult:
        # Time the feature/state access separately and intercept both exact 2-D
        # projections to distinguish QP1 from a triggered anticipatory QP2.
        feature_start = perf_counter_ns()
        _ = tuple(cycle.observation.__dict__.values()) + tuple(cycle.information.chi.__dict__.values())
        feature_s = (perf_counter_ns() - feature_start) * 1e-9
        projection_times: list[float] = []
        original_projection = frozen_controller.project_weighted_2d

        def timed_projection(*args, **kwargs):
            start = perf_counter_ns()
            try:
                return original_projection(*args, **kwargs)
            finally:
                projection_times.append((perf_counter_ns() - start) * 1e-9)

        frozen_controller.project_weighted_2d = timed_projection
        try:
            result = super().run_cycle(cycle)
        finally:
            frozen_controller.project_weighted_2d = original_projection
        timings = dict(result.timings_s)
        timings["feature_state_construction"] = feature_s
        timings["qp1"] = projection_times[0] if projection_times else timings.get("qp_solve", 0.0)
        timings["qp2"] = sum(projection_times[1:])
        return replace(
            result,
            timings_s=timings,
            data_provenance="literature_calibrated",
            paper_eligible=True,
        )

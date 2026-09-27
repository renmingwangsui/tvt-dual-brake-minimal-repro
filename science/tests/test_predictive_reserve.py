"""Mandatory predictive/fleet reserve regression suite."""
from __future__ import annotations

import random
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from safety.conflict_reserve import (  # noqa: E402
    CertificateScales,
    ReserveParameters,
    compute_complete_certificate,
    compute_instantaneous_reserve,
    independent_box_lp_feasible,
)
from safety.fleet_reserve import compute_fleet_reserve, softmin  # noqa: E402
from safety.predictive_reserve import (  # noqa: E402
    PredictiveReserveResult,
    ReserveLowerBound,
    compute_predictive_reserve,
)
from safety.reachable_set import (  # noqa: E402
    AugmentedHybridState,
    IntervalHybridState,
    ReachabilityUncertainty,
    ReachableDynamicsParams,
    point_interval_state,
    propagate_interval_one_step,
    propagate_reachable_set,
)
from safety.supervisor import (  # noqa: E402
    SupervisorMode,
    SupervisorThresholds,
    predictive_supervisor_trigger,
)
from safety_core import Interval, VehicleParams  # noqa: E402


def vehicle_params() -> VehicleParams:
    return VehicleParams(
        mass_kg=28000.0,
        tau_f_s=0.35,
        tau_a_s=0.55,
        heat_capacity_J_per_K=1.6e6,
        cooling_W_per_K=1000.0,
        heat_fraction=0.82,
        friction_cold_limit_N=180000.0,
        friction_command_limit_N=170000.0,
        auxiliary_command_limit_N=120000.0,
        fade_reference_K=500.0,
        fade_slope_per_K=0.0015,
        fade_floor=0.35,
        sample_time_s=0.1,
        low_speed_threshold_mps=0.5,
        temperature_buffer_K=5.0,
        alpha_fade_per_s=1.0,
        alpha_aux_per_s=1.0,
    )


def base_state() -> AugmentedHybridState:
    return AugmentedHybridState(
        position_m=0.0,
        speed_mps=20.0,
        friction_force_N=30000.0,
        auxiliary_force_N=20000.0,
        temperature_K=550.0,
        previous_friction_command_N=35000.0,
        previous_auxiliary_command_N=25000.0,
        gear_id=6,
        gear_dwell_time_s=3.0,
        message_age_s=0.05,
        gap_m=35.0,
        predecessor_speed_mps=19.0,
        predecessor_acceleration_mps2=-0.5,
    )


def reserve_params(rng: random.Random) -> ReserveParameters:
    return ReserveParameters(
        vehicle=vehicle_params(),
        friction_envelope_N=rng.uniform(20000.0, 180000.0),
        auxiliary_envelope_N=rng.uniform(10000.0, 130000.0),
        gear_power_limit_N=rng.uniform(5000.0, 130000.0),
        realized_auxiliary_cbf_limit_N=rng.uniform(5000.0, 130000.0),
        mode_auxiliary_limit_N=rng.uniform(5000.0, 130000.0),
        thermal_hocbf_limit_N=rng.uniform(10000.0, 190000.0),
        fade_cbf_limit_N=rng.uniform(10000.0, 190000.0),
        low_speed_zoh_limit_N=rng.uniform(10000.0, 190000.0),
        low_speed_zoh_g_T0_K=rng.uniform(1.0, 20.0),
        friction_lower_N=rng.uniform(0.0, 30000.0),
        auxiliary_lower_N=rng.uniform(0.0, 20000.0),
        A_f=rng.uniform(1e-6, 5e-5),
        A_a=rng.uniform(1e-6, 5e-5),
        gamma_c_mps2=rng.uniform(0.0, 3.0),
        mu_c_mps2=rng.uniform(0.0, 2.0),
        Xi_hat_c_mps2=rng.uniform(-3.0, 3.0),
        alpha2_psi1_c_mps2=rng.uniform(-2.0, 2.0),
    )


def dynamics_params() -> ReachableDynamicsParams:
    p = vehicle_params()
    return ReachableDynamicsParams(
        sample_time_s=p.sample_time_s,
        nominal_mass_kg=p.mass_kg,
        gravity_mps2=9.81,
        drag_coefficient_N_per_mps2=5.0,
        heat_capacity_J_per_K=p.heat_capacity_J_per_K,
        cooling_W_per_K=p.cooling_W_per_K,
        heat_fraction=p.heat_fraction,
        nominal_tau_f_s=p.tau_f_s,
        nominal_tau_a_s=p.tau_a_s,
    )


def zero_uncertainty() -> ReachabilityUncertainty:
    z = Interval(0.0, 0.0)
    one = Interval(1.0, 1.0)
    return ReachabilityUncertainty(
        grade_rad=z,
        predecessor_acceleration_mps2=Interval(-0.5, -0.5),
        predecessor_jerk_mps3=z,
        mass_scale=one,
        drag_scale=one,
        rolling_force_N=Interval(500.0, 500.0),
        tau_f_scale=one,
        tau_a_scale=one,
        thermal_gain_scale=one,
        cooling_scale=one,
        heat_residual_W=z,
        ambient_temperature_K=Interval(300.0, 300.0),
        communication_delay_s=z,
        auxiliary_availability_scale=one,
    )


def broad_uncertainty(scale: float = 1.0) -> ReachabilityUncertainty:
    return ReachabilityUncertainty(
        grade_rad=Interval(-0.01 * scale, 0.03 * scale),
        predecessor_acceleration_mps2=Interval(-2.0 * scale, 0.5 * scale),
        predecessor_jerk_mps3=Interval(-1.0 * scale, 1.0 * scale),
        mass_scale=Interval(1.0 - 0.08 * scale, 1.0 + 0.08 * scale),
        drag_scale=Interval(1.0 - 0.1 * scale, 1.0 + 0.1 * scale),
        rolling_force_N=Interval(300.0, 700.0 + 200.0 * scale),
        tau_f_scale=Interval(1.0 - 0.08 * scale, 1.0 + 0.08 * scale),
        tau_a_scale=Interval(1.0 - 0.08 * scale, 1.0 + 0.08 * scale),
        thermal_gain_scale=Interval(1.0 - 0.05 * scale, 1.0 + 0.05 * scale),
        cooling_scale=Interval(1.0 - 0.05 * scale, 1.0 + 0.05 * scale),
        heat_residual_W=Interval(-1000.0 * scale, 1000.0 * scale),
        ambient_temperature_K=Interval(295.0, 305.0 + 5.0 * scale),
        communication_delay_s=Interval(0.0, 0.05 * scale),
        auxiliary_availability_scale=Interval(1.0 - 0.1 * scale, 1.0),
    )


def point_from_interval(state: IntervalHybridState) -> AugmentedHybridState:
    def value(interval: Interval) -> float:
        assert interval.low == interval.high
        return interval.low
    return AugmentedHybridState(
        position_m=value(state.position_m),
        speed_mps=value(state.speed_mps),
        friction_force_N=value(state.friction_force_N),
        auxiliary_force_N=value(state.auxiliary_force_N),
        temperature_K=value(state.temperature_K),
        previous_friction_command_N=value(state.previous_friction_command_N),
        previous_auxiliary_command_N=value(state.previous_auxiliary_command_N),
        gear_id=int(value(state.gear_id)),
        gear_dwell_time_s=value(state.gear_dwell_time_s),
        message_age_s=value(state.message_age_s),
        gap_m=value(state.gap_m),
        predecessor_speed_mps=value(state.predecessor_speed_mps),
        predecessor_acceleration_mps2=value(state.predecessor_acceleration_mps2),
    )


def test_instantaneous_reserve_exact_box_solution() -> None:
    rng = random.Random(20260917)
    chi0 = base_state()
    for _ in range(10000):
        chi = replace(
            chi0,
            previous_friction_command_N=rng.uniform(0.0, 180000.0),
            previous_auxiliary_command_N=rng.uniform(0.0, 130000.0),
        )
        params = reserve_params(rng)
        result = compute_instantaneous_reserve(chi, params, {})
        certificate = compute_complete_certificate(
            result, CertificateScales(100000.0, 80000.0, 2.0)
        )
        lp = independent_box_lp_feasible(result, params.A_f, params.A_a)
        assert result.feasible == lp
        assert (certificate.rho >= 0.0) == lp
        assert certificate.M_mps2 == result.M


def test_predictive_horizon_one_matches_instantaneous() -> None:
    rng = random.Random(7)
    params = reserve_params(rng)
    command = (42000.0, 30000.0)
    next_interval = propagate_interval_one_step(
        point_interval_state(base_state()), command, dynamics_params(), zero_uncertainty()
    )
    expected = compute_instantaneous_reserve(point_from_interval(next_interval), params, {})

    def lower(tube: IntervalHybridState, step: int) -> ReserveLowerBound:
        actual = compute_instantaneous_reserve(point_from_interval(tube), params, {})
        certificate = compute_complete_certificate(actual, CertificateScales(100000.0, 80000.0, 2.0))
        return ReserveLowerBound(certificate.rho, actual.friction_interval_nonempty, actual.auxiliary_interval_nonempty, certificate.critical_limit_type)

    predicted = compute_predictive_reserve(
        base_state(), lambda state, step: command, 1, dynamics_params(), zero_uncertainty(), lower, command
    )
    expected_rho = compute_complete_certificate(expected, CertificateScales(100000.0, 80000.0, 2.0)).rho
    assert predicted.min_reserve == expected_rho
    assert predicted.critical_step == 1


def sample_interval(rng: random.Random, interval: Interval) -> Interval:
    value = rng.uniform(interval.low, interval.high)
    return Interval(value, value)


def sample_uncertainty(rng: random.Random, box: ReachabilityUncertainty) -> ReachabilityUncertainty:
    return ReachabilityUncertainty(**{
        name: sample_interval(rng, value) for name, value in box.__dict__.items()
    })


def test_reachable_set_contains_monte_carlo() -> None:
    rng = random.Random(101)
    box = broad_uncertainty(1.0)
    policy = lambda state, step: (45000.0 + 1000.0 * step, 32000.0)
    tube = propagate_reachable_set(base_state(), policy, 5, dynamics_params(), box)
    for _ in range(1000):
        current = point_interval_state(base_state())
        for step in range(1, 6):
            current = propagate_interval_one_step(
                current, policy(current, step), dynamics_params(), sample_uncertainty(rng, box)
            )
            assert tube[step - 1].contains(point_from_interval(current), tolerance=1e-7)


def test_predictive_reserve_monotonic_uncertainty() -> None:
    def lower(tube: IntervalHybridState, step: int) -> ReserveLowerBound:
        value = 90000.0 - 70.0 * tube.temperature_K.high - 20.0 * tube.speed_mps.high + 10.0 * tube.gap_m.low
        return ReserveLowerBound(value, True, True)
    policy = lambda state, step: (50000.0, 35000.0)
    narrow = compute_predictive_reserve(
        base_state(), policy, 5, dynamics_params(), broad_uncertainty(0.5), lower, (50000.0, 35000.0)
    )
    wide = compute_predictive_reserve(
        base_state(), policy, 5, dynamics_params(), broad_uncertainty(1.0), lower, (50000.0, 35000.0)
    )
    assert wide.min_reserve <= narrow.min_reserve + 1e-9


def dummy_result(reserve: float, step: int) -> PredictiveReserveResult:
    return PredictiveReserveResult(
        (reserve,), reserve, step, (), (True,), ("thermal_hocbf",),
        "thermal_hocbf", ((0.0, 0.0),), (0.0, 0.0)
    )


def test_horizon_monotonicity() -> None:
    def lower(tube: IntervalHybridState, step: int) -> ReserveLowerBound:
        return ReserveLowerBound(1.0 - 0.1 * step, True, True, "collision_demand")
    policy = lambda state, step: (50000.0, 35000.0)
    h4 = compute_predictive_reserve(base_state(), policy, 4, dynamics_params(), broad_uncertainty(0.5), lower, (50000.0, 35000.0))
    h5 = compute_predictive_reserve(base_state(), policy, 5, dynamics_params(), broad_uncertainty(0.5), lower, (50000.0, 35000.0))
    assert h5.rho_H <= h4.rho_H
    assert h5.critical_limit_type == "collision_demand"


def test_fleet_reserve_exact_min() -> None:
    values = [dummy_result(2.5, 2), dummy_result(-1.0, 4), dummy_result(0.2, 3)]
    assert compute_fleet_reserve(values).fleet_reserve == -1.0


def test_bottleneck_vehicle() -> None:
    values = [dummy_result(2.5, 2), dummy_result(-1.0, 4), dummy_result(0.2, 3)]
    result = compute_fleet_reserve(values)
    assert result.critical_vehicle == 1
    assert result.critical_step == 4


def test_predictive_trigger() -> None:
    thresholds = SupervisorThresholds(2.0, 3.0, 0.5, 1.0, 3)
    decision = predictive_supervisor_trigger(4.0, 1.5, True, False, thresholds)
    assert decision.mode is SupervisorMode.ANTICIPATORY
    assert decision.anticipatory_triggered and not decision.hard_triggered


def test_softmin_converges() -> None:
    values = [1.0, 2.0, 4.0]
    assert abs(softmin(values, 1e-3) - min(values)) < 1e-6


def main() -> None:
    tests = [
        test_instantaneous_reserve_exact_box_solution,
        test_predictive_horizon_one_matches_instantaneous,
        test_reachable_set_contains_monte_carlo,
        test_predictive_reserve_monotonic_uncertainty,
        test_horizon_monotonicity,
        test_fleet_reserve_exact_min,
        test_bottleneck_vehicle,
        test_predictive_trigger,
        test_softmin_converges,
    ]
    for test in tests:
        test()
    print("PASS: 10k box-LP equivalence, interval tube, predictive/fleet reserve, trigger, and softmin")


if __name__ == "__main__":
    main()

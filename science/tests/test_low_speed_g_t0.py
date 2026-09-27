"""Regression tests for the state-only low-speed thermal reserve g_T0."""
from __future__ import annotations

from dataclasses import replace
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from safety.conflict_reserve import (  # noqa: E402
    CertificateScales,
    ReserveParameters,
    compute_complete_certificate,
    compute_instantaneous_reserve,
    compute_interval_certificate_lower_bound,
    independent_box_lp_feasible,
)
from safety.reachable_set import AugmentedHybridState  # noqa: E402
from safety_core import (  # noqa: E402
    Interval,
    State,
    build_hard_rows,
    feasible,
    low_speed_temperature_row,
    thermal_constraint_mode,
)
from test_predictive_reserve import vehicle_params  # noqa: E402


EPSILON = 0.5
SPEEDS = (0.0, 1e-9, EPSILON, EPSILON + 1e-6)


def augmented(speed_mps: float) -> AugmentedHybridState:
    return AugmentedHybridState(
        position_m=0.0,
        speed_mps=speed_mps,
        friction_force_N=0.0,
        auxiliary_force_N=0.0,
        temperature_K=300.0,
        previous_friction_command_N=0.0,
        previous_auxiliary_command_N=0.0,
        gear_id=1,
        gear_dwell_time_s=2.0,
        message_age_s=0.0,
        gap_m=50.0,
        predecessor_speed_mps=0.0,
        predecessor_acceleration_mps2=0.0,
    )


def reserve_parameters(g_T0_K: float) -> ReserveParameters:
    p = replace(
        vehicle_params(),
        low_speed_threshold_mps=EPSILON,
    )
    return ReserveParameters(
        vehicle=p,
        friction_envelope_N=160_000.0,
        auxiliary_envelope_N=110_000.0,
        gear_power_limit_N=100_000.0,
        realized_auxiliary_cbf_limit_N=100_000.0,
        mode_auxiliary_limit_N=100_000.0,
        thermal_hocbf_limit_N=80_000.0,
        fade_cbf_limit_N=150_000.0,
        low_speed_zoh_limit_N=90_000.0,
        low_speed_zoh_g_T0_K=g_T0_K,
        friction_lower_N=0.0,
        auxiliary_lower_N=0.0,
        A_f=1e-5,
        A_a=1e-5,
        gamma_c_mps2=0.0,
        mu_c_mps2=0.0,
        Xi_hat_c_mps2=0.0,
        alpha2_psi1_c_mps2=0.0,
    )


def test_zoh_row_hot_cold_and_speed_boundary() -> None:
    p = replace(vehicle_params(), low_speed_threshold_mps=EPSILON)
    for speed in SPEEDS:
        cold = low_speed_temperature_row(
            State(speed, 0.0, 0.0, 300.0, 0.0), p, 300.0, 0.0, 700.0, 1.0
        )
        hot = low_speed_temperature_row(
            State(speed, 0.0, 0.0, 800.0, 0.0), p, 300.0, 0.0, 700.0, 1.0
        )
        assert cold["g_T0_K"] > 0.0
        assert hot["g_T0_K"] < 0.0
        assert cold["g_T0_K"] == cold["rhs_K"]
        if speed == 0.0:
            assert cold["coefficient_K_per_N"] == 0.0
            assert hot["coefficient_K_per_N"] == 0.0
        else:
            assert cold["coefficient_K_per_N"] > 0.0
        expected_mode = "zoh" if speed <= EPSILON else "hocbf"
        assert thermal_constraint_mode(speed, p) == expected_mode


def test_state_only_row_is_hard_infeasibility() -> None:
    p = replace(vehicle_params(), low_speed_threshold_mps=EPSILON)
    hot = low_speed_temperature_row(
        State(0.0, 0.0, 0.0, 800.0, 0.0), p, 300.0, 0.0, 700.0, 1.0
    )
    collision = {"A_f": 1e-5, "A_a": 1e-5, "D_c": 0.0}
    limits = {"L_f": 0.0, "U_f": 100_000.0, "L_a": 0.0, "U_a": 100_000.0}
    rows = build_hard_rows(collision, limits, None, hot)
    assert rows[-1].name == "thermal_zoh_state"
    assert rows[-1].g_f == 0.0 and rows[-1].h == hot["g_T0_K"]
    assert not feasible(rows, (0.0, 0.0))
    try:
        build_hard_rows(
            collision,
            limits,
            {"B_f": 1.0, "rhs": 1.0},
            hot,
        )
    except ValueError as exc:
        assert "mutually exclusive" in str(exc)
    else:
        raise AssertionError("thermal HOCBF and low-speed ZOH rows must not coexist")


def test_complete_certificate_partition_and_rho() -> None:
    scales = CertificateScales(100_000.0, 100_000.0, 2.0, 10.0)
    for speed in (0.0, 1e-9, EPSILON):
        hot = compute_instantaneous_reserve(augmented(speed), reserve_parameters(-2.0))
        hot_certificate = compute_complete_certificate(hot, scales)
        assert hot.low_speed_active
        assert hot.active_friction_limit == "low_speed_zoh"
        assert hot.friction_interval_nonempty and hot.auxiliary_interval_nonempty and hot.M >= 0.0
        assert not hot.low_speed_thermal_feasible and not hot.feasible
        assert hot_certificate.rho == -0.2
        assert hot_certificate.critical_limit_type == "low_speed_thermal_state"
        assert not independent_box_lp_feasible(hot, 1e-5, 1e-5)

        cold = compute_instantaneous_reserve(augmented(speed), reserve_parameters(2.0))
        cold_certificate = compute_complete_certificate(cold, scales)
        assert cold.low_speed_thermal_feasible and cold.feasible
        assert cold_certificate.rho >= 0.0

    high = compute_instantaneous_reserve(
        augmented(EPSILON + 1e-6), reserve_parameters(-2.0)
    )
    high_certificate = compute_complete_certificate(high, scales)
    assert not high.low_speed_active
    assert high.g_T0_K is None
    assert high.active_friction_limit == "thermal_hocbf"
    assert high.feasible and high_certificate.rho >= 0.0
    assert high_certificate.normalized_low_speed_thermal is None


def test_interval_lower_bound_includes_g_T0() -> None:
    lower, critical = compute_interval_certificate_lower_bound(
        Interval(10.0, 20.0),
        Interval(10.0, 20.0),
        Interval(1.0, 2.0),
        CertificateScales(100.0, 100.0, 2.0, 10.0),
        Interval(-3.0, 1.0),
    )
    assert lower == -0.3
    assert critical == "low_speed_thermal_state"


def main() -> None:
    test_zoh_row_hot_cold_and_speed_boundary()
    test_state_only_row_is_hard_infeasibility()
    test_complete_certificate_partition_and_rho()
    test_interval_lower_bound_includes_g_T0()
    print("PASS: g_T0 hard infeasibility at zero/near-zero/epsilon and hot/cold boundaries")


if __name__ == "__main__":
    main()

"""Regression tests for interval semantics, causality, aggregation and set-valued backup."""
from __future__ import annotations

import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from safety.conflict_reserve import (  # noqa: E402
    CertificateScales, InstantaneousReserveResult,
    compute_interval_certificate_lower_bound, normalization_sensitivity,
)
from safety.control_loop import TwoPassCallbacks, execute_two_pass_control_cycle  # noqa: E402
from safety.fleet_reserve import UpstreamSafetyMessage, aggregate_upstream_message  # noqa: E402
from safety.predictive_reserve import ReserveLowerBound, compute_predictive_reserve  # noqa: E402
from safety.reachable_set import (  # noqa: E402
    saturated_affine_backup_extension, propagate_reachable_set,
)
from safety.supervisor import BackupDomainConditions  # noqa: E402
from safety_core import Interval  # noqa: E402
from test_predictive_reserve import base_state, broad_uncertainty, dynamics_params  # noqa: E402


def test_interval_lower_bound_semantics() -> None:
    scales = CertificateScales(100.0, 50.0, 2.0)
    df, da, margin = Interval(-5.0, 20.0), Interval(10.0, 30.0), Interval(-0.4, 1.0)
    lower, _ = compute_interval_certificate_lower_bound(df, da, margin, scales)
    rng = random.Random(31)
    for _ in range(5000):
        rho = min(
            rng.uniform(df.low, df.high) / scales.friction_force_N,
            rng.uniform(da.low, da.high) / scales.auxiliary_force_N,
            rng.uniform(margin.low, margin.high) / scales.collision_mps2,
        )
        assert lower <= rho + 1e-12
    assert lower < 0.0  # warning only; sampled points may still be feasible


def test_causal_two_pass_order_and_reverification() -> None:
    events: list[str] = []
    def event(name: str, value: object):
        events.append(name); return value
    callbacks = TwoPassCallbacks(
        measure_and_align=lambda: event("measure", "state"),
        build_margins_and_intervals=lambda state: event("margins", "hard"),
        sample_nominal_actor=lambda state, hard: event("actor", (1.0, 2.0)),
        solve_normal_qp=lambda nominal, hard: event("normal_qp", (1.5, 2.5)),
        predict_from_provisional=lambda state, action: event("predict0", (events[-1], tuple(action))),
        aggregate_bottleneck=lambda prediction: event("aggregate", "message"),
        select_final_command=lambda provisional, prediction, aggregate: event("supervisor", ("anticipatory", (2.0, 3.0))),
        fast_verify_final_prediction=lambda state, action: event("reverify", tuple(action)),
        validate_hard_rows=lambda state, action: event("validate", None),
        apply_command=lambda action: event("apply", None),
        log_cycle=lambda record: event("log", None),
    )
    result = execute_two_pass_control_cycle(callbacks)
    assert events == ["measure", "margins", "actor", "normal_qp", "predict0", "aggregate", "supervisor", "reverify", "validate", "apply", "log"]
    assert result.provisional_prediction[0] == "normal_qp"
    assert result.prediction_reverified and result.final_command == (2.0, 3.0)


def test_set_valued_backup_and_nested_closed_loop_boxes() -> None:
    policy = lambda state, step: saturated_affine_backup_extension(
        state, 1000.0, -500.0, 20000.0, 60000.0,
        (0.0, 120000.0), (0.0, 100000.0),
    )
    narrow_u = broad_uncertainty(0.5)
    wide_u = broad_uncertainty(1.0)
    assert narrow_u.is_subset_of(wide_u)
    narrow = propagate_reachable_set(base_state(), policy, 5, dynamics_params(), narrow_u)
    wide = propagate_reachable_set(base_state(), policy, 5, dynamics_params(), wide_u)
    assert all(a.is_subset_of(b) for a, b in zip(narrow, wide))

    predicted = compute_predictive_reserve(
        base_state(), policy, 4, dynamics_params(), narrow_u,
        lambda state, step: ReserveLowerBound(1.0 - 0.1 * step, True, True, "collision_demand"),
        (25000.0, 30000.0),
    )
    assert predicted.backup_control_widths_N[0] == (0.0, 0.0)
    assert any(max(widths) > 0.0 for widths in predicted.backup_control_widths_N[1:])


def test_distributed_bottleneck_protocol() -> None:
    fallback = UpstreamSafetyMessage(-9.0, -1, 0, "stale_message_bound", 0.0, True)
    successor = UpstreamSafetyMessage(0.4, 2, 4, "thermal_hocbf", 9.5)
    accepted = aggregate_upstream_message(1, 1.0, 2, "collision_demand", 10.0, successor, 1.0, fallback)
    assert (accepted.critical_vehicle_id, accepted.critical_future_step) == (2, 4)
    lost = aggregate_upstream_message(1, 1.0, 2, "collision_demand", 10.0, None, 1.0, fallback)
    assert lost.fallback_used and lost.rho_upstream_H == -9.0
    stale = aggregate_upstream_message(1, 1.0, 2, "collision_demand", 10.0, successor, 0.2, fallback)
    assert stale.fallback_used
    changed = aggregate_upstream_message(1, 0.1, 3, "gear_power", 10.0, successor, 1.0, fallback)
    assert changed.critical_vehicle_id == 1 and changed.limiting_physical_constraint == "gear_power"
    tail = UpstreamSafetyMessage(0.3, 3, 5, "auxiliary_rate", 20.0)
    middle = aggregate_upstream_message(2, 0.5, 2, "fade_cbf", 20.0, tail, 0.1, fallback)
    head = aggregate_upstream_message(1, 0.8, 1, "collision_demand", 20.0, middle, 0.1, fallback)
    assert (head.rho_upstream_H, head.critical_vehicle_id, head.critical_future_step) == (0.3, 3, 5)


def test_normalization_sign_invariance_and_margin_logging() -> None:
    result = InstantaneousReserveResult(
        0.0, 10.0, 0.0, 20.0, 0.0, 1.0, True, True, True,
        "thermal_hocbf", "gear_power",
    )
    certificates = normalization_sensitivity(result, (
        CertificateScales(10.0, 20.0, 1.0),
        CertificateScales(100.0, 10.0, 0.5),
        CertificateScales(5.0, 100.0, 4.0),
    ))
    assert all(item.rho >= 0.0 for item in certificates)
    assert all((item.delta_f_N, item.delta_a_N, item.M_mps2) == (10.0, 20.0, 1.0) for item in certificates)
    assert len({item.rho for item in certificates}) > 1
    assert len({item.critical_limit_type for item in certificates}) > 1


def test_complete_backup_domain_guard() -> None:
    valid = BackupDomainConditions(
        h_collision_m=1.0,
        h_temperature_K=2.0,
        h_fade_N=3.0,
        h_auxiliary_N=4.0,
        psi1_collision_mps=0.5,
        psi1_temperature_Kps=0.25,
        rho_instantaneous=0.1,
        speed_mps=5.0,
        gear_dwell_valid=True,
        communication_uncertainty_valid=True,
    )
    assert valid.contains_current_state()
    assert not BackupDomainConditions(**{**valid.__dict__, "h_auxiliary_N": -1.0}).contains_current_state()
    assert not BackupDomainConditions(**{**valid.__dict__, "communication_uncertainty_valid": False}).contains_current_state()


def main() -> None:
    for test in (
        test_interval_lower_bound_semantics,
        test_causal_two_pass_order_and_reverification,
        test_set_valued_backup_and_nested_closed_loop_boxes,
        test_distributed_bottleneck_protocol,
        test_normalization_sign_invariance_and_margin_logging,
        test_complete_backup_domain_guard,
    ):
        test()
    print("PASS: IA lower semantics, causal loop, set-valued backup, distributed min and normalization")


if __name__ == "__main__":
    main()

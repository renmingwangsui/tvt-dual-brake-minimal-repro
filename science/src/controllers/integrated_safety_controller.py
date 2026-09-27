"""Non-reorderable Phase 2B adapter around the existing audited safety stack."""
from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter_ns
from typing import Sequence

from controllers.debug_nominal import NominalDebugController
from envs.observations import LocalActorObservation, SafetyControllerInformation
from safety.conflict_reserve import (
    CertificateScales,
    ReserveParameters,
    compute_complete_certificate,
    compute_instantaneous_reserve,
    compute_interval_certificate_lower_bound,
)
from safety.control_loop import TwoPassCallbacks, TwoPassCycleResult, execute_two_pass_control_cycle
from safety.fleet_reserve import UpstreamSafetyMessage, aggregate_upstream_message
from safety.predictive_reserve import PredictiveReserveResult, ReserveLowerBound, compute_predictive_reserve
from safety.reachable_set import (
    ReachabilityUncertainty,
    ReachableDynamicsParams,
    saturated_affine_backup_extension,
)
from safety.supervisor import SupervisorDecision, SupervisorMode, SupervisorThresholds, predictive_supervisor_trigger
from safety_core import (
    Interval,
    Predecessor,
    Row,
    State,
    VehicleParams,
    all_residuals,
    auxiliary_cbf_upper_bound,
    build_hard_rows,
    certified_backup,
    collision_affine,
    compute_digital_rate_bounds,
    directional_margin,
    effective_limits,
    fade_upper_bound,
    low_speed_temperature_row,
    project_weighted_2d,
    sampled_data_margin,
    state_derivatives,
    thermal_affine,
)
from simulator.truck_dynamics import VehicleParameters


@dataclass(frozen=True)
class DebugSafetyControllerConfig:
    vehicle: VehicleParams
    critical_temperature_K: float
    horizon: int
    scales: CertificateScales
    supervisor_thresholds: SupervisorThresholds
    maximum_message_age_s: float
    reachability_uncertainty_scale: float = 0.02
    data_provenance: str = "synthetic_debug"
    paper_eligible: bool = False

    def __post_init__(self) -> None:
        if self.horizon < 1 or self.maximum_message_age_s < 0.0:
            raise ValueError("invalid safety-controller horizon or message age")
        if self.data_provenance != "synthetic_debug" or self.paper_eligible:
            raise ValueError("debug safety configuration cannot be paper eligible")

    @classmethod
    def from_vehicle(
        cls, parameters: VehicleParameters, sample_time_s: float
    ) -> "DebugSafetyControllerConfig":
        vehicle = VehicleParams(
            mass_kg=parameters.mass_kg,
            tau_f_s=parameters.tau_f_s,
            tau_a_s=parameters.tau_a_s,
            heat_capacity_J_per_K=parameters.thermal_capacity_J_per_K,
            cooling_W_per_K=parameters.cooling_W_per_K,
            heat_fraction=parameters.heat_fraction,
            friction_cold_limit_N=parameters.friction_force_limit_N,
            friction_command_limit_N=parameters.friction_force_limit_N,
            auxiliary_command_limit_N=parameters.auxiliary_force_limit_N,
            fade_reference_K=400.0,
            fade_slope_per_K=0.001,
            fade_floor=0.35,
            sample_time_s=sample_time_s,
            low_speed_threshold_mps=0.5,
            temperature_buffer_K=5.0,
            alpha_fade_per_s=1.0,
            alpha_aux_per_s=1.0,
        )
        return cls(
            vehicle=vehicle,
            critical_temperature_K=parameters.critical_temperature_K,
            horizon=3,
            scales=CertificateScales(100_000.0, 80_000.0, 2.0, 2.0),
            supervisor_thresholds=SupervisorThresholds(0.08, 0.15, 0.02, 0.05, 2),
            maximum_message_age_s=0.3,
        )


@dataclass(frozen=True)
class ControllerCycleInput:
    vehicle_id: int
    time_s: float
    information: SafetyControllerInformation
    observation: LocalActorObservation
    nominal_controller: NominalDebugController
    nonbraking_force_N: float
    ambient_temperature_K: float
    residual_heat_W: float
    grade_rad: float
    auxiliary_limit_N: float
    predecessor_jerk_bound_mps3: float
    received_successor: UpstreamSafetyMessage | None
    is_tail_vehicle: bool = False


@dataclass
class _HardContext:
    state: State
    collision: dict[str, float]
    thermal: dict[str, float] | None
    low_speed: dict[str, float] | None
    fade: dict[str, float]
    auxiliary: dict[str, float]
    limits: dict[str, float]
    rows: list[Row]
    instantaneous: object
    certificate: object
    margins: dict[str, float]
    qp_status: str = "not_solved"
    hard_polytope_empty: bool = False
    final_residuals: dict[str, float] = field(default_factory=dict)
    final_residuals_valid: bool = False


@dataclass(frozen=True)
class IntegratedCycleResult:
    two_pass: TwoPassCycleResult
    instantaneous_certificate: object
    instantaneous_result: object
    upstream_message: UpstreamSafetyMessage
    hard_rows: tuple[Row, ...]
    hard_residuals: dict[str, float]
    residuals_valid: bool
    qp_status: str
    callback_order: tuple[str, ...]
    supervisor_decision: SupervisorDecision
    barriers: dict[str, float]
    timings_s: dict[str, float] = field(default_factory=dict)
    data_provenance: str = "synthetic_debug"
    paper_eligible: bool = False


class IntegratedSafetyController:
    def __init__(self, config: DebugSafetyControllerConfig) -> None:
        self.config = config
        self.previous_mode = SupervisorMode.NORMAL
        self.recovery_counter = 0

    def _reachability_parameters(self) -> ReachableDynamicsParams:
        p = self.config.vehicle
        return ReachableDynamicsParams(
            sample_time_s=p.sample_time_s,
            nominal_mass_kg=p.mass_kg,
            gravity_mps2=9.80665,
            drag_coefficient_N_per_mps2=3.0,
            heat_capacity_J_per_K=p.heat_capacity_J_per_K,
            cooling_W_per_K=p.cooling_W_per_K,
            heat_fraction=p.heat_fraction,
            nominal_tau_f_s=p.tau_f_s,
            nominal_tau_a_s=p.tau_a_s,
        )

    def _uncertainty(self, cycle: ControllerCycleInput) -> ReachabilityUncertainty:
        scale = self.config.reachability_uncertainty_scale
        reachable = cycle.information.predecessor_reachable
        one = Interval(1.0 - scale, 1.0 + scale)
        return ReachabilityUncertainty(
            grade_rad=Interval(cycle.grade_rad - 0.002, cycle.grade_rad + 0.002),
            predecessor_acceleration_mps2=reachable.acceleration_mps2,
            predecessor_jerk_mps3=Interval(-cycle.predecessor_jerk_bound_mps3, cycle.predecessor_jerk_bound_mps3),
            mass_scale=one,
            drag_scale=one,
            rolling_force_N=Interval(0.0, 1_000.0),
            tau_f_scale=one,
            tau_a_scale=one,
            thermal_gain_scale=one,
            cooling_scale=one,
            heat_residual_W=Interval(cycle.residual_heat_W - 100.0, cycle.residual_heat_W + 100.0),
            ambient_temperature_K=Interval(cycle.ambient_temperature_K - 1.0, cycle.ambient_temperature_K + 1.0),
            communication_delay_s=Interval(0.0, min(cycle.information.chi.message_age_s, self.config.maximum_message_age_s)),
            auxiliary_availability_scale=Interval(0.95, 1.0),
        )

    def _build_context(self, cycle: ControllerCycleInput) -> _HardContext:
        p = self.config.vehicle
        chi = cycle.information.chi
        acceleration = (
            cycle.nonbraking_force_N - chi.friction_force_N - chi.auxiliary_force_N
        ) / p.mass_kg
        state = State(
            chi.speed_mps,
            chi.friction_force_N,
            chi.auxiliary_force_N,
            chi.temperature_K,
            acceleration,
        )
        predecessor = Predecessor(
            chi.predecessor_speed_mps,
            chi.predecessor_acceleration_mps2,
            0.0,
        )
        directional_collision = directional_margin(0.0, Interval(-0.01, 0.01))
        collision = collision_affine(
            state, predecessor, p, chi.gap_m, 5.0, 1.2, 4.0, 6.0, 0.0, 1.0, 1.2,
            gamma_mps2=directional_collision,
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
            p,
            cycle.nonbraking_force_N,
            cycle.ambient_temperature_K,
            cycle.residual_heat_W,
        )
        fade = fade_upper_bound(state, p, rates["T_dot"])
        auxiliary = auxiliary_cbf_upper_bound(state, p, cycle.auxiliary_limit_N, 0.0)
        low_speed = None
        thermal = None
        if chi.speed_mps <= p.low_speed_threshold_mps:
            low_speed = low_speed_temperature_row(
                state,
                p,
                cycle.ambient_temperature_K,
                cycle.residual_heat_W,
                self.config.critical_temperature_K,
                0.5,
            )
            if low_speed["coefficient_K_per_N"] > 0.0:
                thermal_upper = low_speed["g_T0_K"] / low_speed["coefficient_K_per_N"]
            else:
                thermal_upper = p.friction_command_limit_N
        else:
            thermal = thermal_affine(
                state,
                p,
                self.config.critical_temperature_K - p.temperature_buffer_K,
                rates["T_dot"],
                0.0,
                0.0,
                0.8,
                1.1,
            )
            thermal_upper = thermal["rhs"] / thermal["B_f"] if thermal["B_f"] > 0.0 else p.friction_command_limit_N
        limits = effective_limits(
            p,
            (thermal_upper, fade["upper_N"]),
            cycle.auxiliary_limit_N,
            auxiliary["upper_N"],
        )
        rows = build_hard_rows(collision, limits, thermal, low_speed)
        rows.extend((
            Row("friction_command_envelope", 1.0, 0.0, p.friction_command_limit_N, "N"),
            Row("auxiliary_command_envelope", 0.0, 1.0, cycle.auxiliary_limit_N, "N"),
            Row("fade_cbf", 1.0, 0.0, fade["upper_N"], "N"),
            Row("realized_auxiliary_cbf", 0.0, 1.0, auxiliary["upper_N"], "N"),
        ))
        low_limit = thermal_upper if low_speed is not None else p.friction_command_limit_N
        g_t0 = low_speed["g_T0_K"] if low_speed is not None else 1.0
        reserve_params = ReserveParameters(
            vehicle=p,
            friction_envelope_N=p.friction_command_limit_N,
            auxiliary_envelope_N=cycle.auxiliary_limit_N,
            gear_power_limit_N=cycle.auxiliary_limit_N,
            realized_auxiliary_cbf_limit_N=auxiliary["upper_N"],
            mode_auxiliary_limit_N=cycle.auxiliary_limit_N,
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
        return _HardContext(state, collision, thermal, low_speed, fade, auxiliary, limits, rows, instantaneous, certificate, margins)

    def _predict(
        self, cycle: ControllerCycleInput, context: _HardContext, provisional: Sequence[float]
    ) -> PredictiveReserveResult:
        scales = self.config.scales
        certificate = context.certificate

        def backup(state: object, step: int) -> tuple[Interval, Interval]:
            return saturated_affine_backup_extension(
                state,
                180.0,
                260.0,
                context.limits["L_f"],
                context.limits["L_a"],
                (context.limits["L_f"], max(context.limits["L_f"], context.limits["U_f"])),
                (context.limits["L_a"], max(context.limits["L_a"], context.limits["U_a"])),
            )

        def lower(state: object, step: int) -> ReserveLowerBound:
            width_b = state.friction_force_N.high - state.friction_force_N.low
            width_r = state.auxiliary_force_N.high - state.auxiliary_force_N.low
            width_gap = state.gap_m.high - state.gap_m.low
            delta_f = Interval(certificate.delta_f_N - width_b, certificate.delta_f_N + width_b)
            delta_a = Interval(certificate.delta_a_N - width_r, certificate.delta_a_N + width_r)
            collision = Interval(
                certificate.M_mps2 - 0.02 * width_gap - 1e-5 * (width_b + width_r),
                certificate.M_mps2 + 0.02 * width_gap + 1e-5 * (width_b + width_r),
            )
            low = None
            if certificate.g_T0_K is not None:
                width_t = state.temperature_K.high - state.temperature_K.low
                low = Interval(certificate.g_T0_K - width_t, certificate.g_T0_K + width_t)
            value, critical = compute_interval_certificate_lower_bound(
                delta_f, delta_a, collision, scales, low
            )
            return ReserveLowerBound(value, delta_f.low >= 0.0, delta_a.low >= 0.0, critical)

        return compute_predictive_reserve(
            cycle.information.chi,
            backup,
            self.config.horizon,
            self._reachability_parameters(),
            self._uncertainty(cycle),
            lower,
            provisional,
        )

    def evaluate_supervisor(
        self,
        instantaneous_rho: float,
        predictive_rho: float,
        instantaneous_feasible: bool,
        hard_polytope_empty: bool,
    ) -> SupervisorDecision:
        return predictive_supervisor_trigger(
            instantaneous_rho,
            predictive_rho,
            instantaneous_feasible,
            hard_polytope_empty,
            self.config.supervisor_thresholds,
            self.previous_mode,
            self.recovery_counter,
        )

    def run_cycle(self, cycle: ControllerCycleInput) -> IntegratedCycleResult:
        events: list[str] = []
        timings_ns: dict[str, int] = {
            "hard_row_construction": 0,
            "qp_solve": 0,
            "predictive_tube": 0,
            "distributed_bottleneck": 0,
            "supervisor": 0,
            "reverification": 0,
        }
        total_start_ns = perf_counter_ns()
        context_box: dict[str, _HardContext] = {}
        aggregate_box: dict[str, UpstreamSafetyMessage] = {}
        decision_box: dict[str, SupervisorDecision] = {}

        def measure() -> SafetyControllerInformation:
            events.extend(("measure_physical_state", "process_sensing_and_messages", "construct_augmented_chi"))
            return cycle.information

        def build(_information: SafetyControllerInformation) -> _HardContext:
            events.extend(("construct_uncertainty", "construct_margins", "construct_hard_rows", "instantaneous_certificate"))
            start_ns = perf_counter_ns()
            context = self._build_context(cycle)
            timings_ns["hard_row_construction"] += perf_counter_ns() - start_ns
            context_box["value"] = context
            return context

        def nominal(_state: object, _context: object) -> tuple[float, float]:
            events.append("nominal_debug_controller")
            return cycle.nominal_controller.command(cycle.observation)

        def solve(command: Sequence[float], context: _HardContext) -> tuple[float, float]:
            events.append("normal_hard_qp")
            start_ns = perf_counter_ns()
            try:
                result = project_weighted_2d(command, context.rows)
                context.qp_status = "optimal"
                return result
            except ValueError:
                context.qp_status = "empty_hard_polytope"
                context.hard_polytope_empty = True
                return (
                    max(0.0, min(context.limits["U_f"], float(command[0]))),
                    max(0.0, min(context.limits["U_a"], float(command[1]))),
                )
            finally:
                timings_ns["qp_solve"] += perf_counter_ns() - start_ns

        def predict(_state: object, action: Sequence[float]) -> PredictiveReserveResult:
            events.append("predictive_tube")
            start_ns = perf_counter_ns()
            try:
                return self._predict(cycle, context_box["value"], action)
            finally:
                timings_ns["predictive_tube"] += perf_counter_ns() - start_ns

        def aggregate(prediction: PredictiveReserveResult) -> UpstreamSafetyMessage:
            events.append("distributed_bottleneck")
            start_ns = perf_counter_ns()
            try:
                if cycle.is_tail_vehicle:
                    message = UpstreamSafetyMessage(
                        prediction.rho_cert_H,
                        cycle.vehicle_id,
                        prediction.critical_step,
                        prediction.critical_limit_type,
                        cycle.time_s,
                        False,
                    )
                    aggregate_box["value"] = message
                    return message
                fallback = UpstreamSafetyMessage(-1.0, -1, 0, "stale_message_bound", cycle.time_s, True)
                message = aggregate_upstream_message(
                    cycle.vehicle_id,
                    prediction.rho_cert_H,
                    prediction.critical_step,
                    prediction.critical_limit_type,
                    cycle.time_s,
                    cycle.received_successor,
                    self.config.maximum_message_age_s,
                    fallback,
                )
                aggregate_box["value"] = message
                return message
            finally:
                timings_ns["distributed_bottleneck"] += perf_counter_ns() - start_ns

        def select(
            provisional: Sequence[float], prediction: PredictiveReserveResult, upstream: UpstreamSafetyMessage
        ) -> tuple[str, tuple[float, float]]:
            events.append("supervisor_after_prediction")
            start_ns = perf_counter_ns()
            context = context_box["value"]
            predictive_value = min(prediction.rho_cert_H, upstream.rho_upstream_H)
            decision = self.evaluate_supervisor(
                context.certificate.rho,
                predictive_value,
                context.instantaneous.feasible,
                context.hard_polytope_empty,
            )
            decision_box["value"] = decision
            if decision.mode is SupervisorMode.NORMAL:
                selected = tuple(map(float, provisional))
            elif decision.mode is SupervisorMode.ANTICIPATORY:
                target = (float(provisional[0]), max(0.0, context.limits["U_a"]))
                try:
                    selected = project_weighted_2d(target, context.rows)
                except ValueError:
                    selected = tuple(map(float, provisional))
            elif decision.mode is SupervisorMode.CERTIFIED_BACKUP:
                selected = certified_backup(context.collision, context.limits)
            else:
                selected = (
                    max(0.0, min(context.limits["U_f"], context.limits["L_f"])),
                    max(0.0, context.limits["U_a"]),
                )
            self.previous_mode = decision.mode
            self.recovery_counter = decision.recovery_counter
            timings_ns["supervisor"] += perf_counter_ns() - start_ns
            return decision.mode.value, selected

        def reverify(_state: object, action: Sequence[float]) -> PredictiveReserveResult:
            events.append("reverify_changed_final_action")
            start_ns = perf_counter_ns()
            try:
                return self._predict(cycle, context_box["value"], action)
            finally:
                timings_ns["reverification"] += perf_counter_ns() - start_ns

        def validate(_state: object, action: Sequence[float]) -> None:
            events.append("validate_final_hard_rows")
            context = context_box["value"]
            context.final_residuals = all_residuals(context.rows, action)
            context.final_residuals_valid = all(value >= -1e-6 for value in context.final_residuals.values())
            mode = decision_box["value"].mode
            if not context.final_residuals_valid and mode is not SupervisorMode.MINIMAL_RISK:
                raise ValueError("final command failed a hard-row residual outside minimal-risk mode")

        def apply(_action: Sequence[float]) -> None:
            if not events or events[-1] != "validate_final_hard_rows":
                raise RuntimeError("unverified command cannot be queued for plant integration")
            events.append("queue_verified_action")

        callbacks = TwoPassCallbacks(
            measure_and_align=measure,
            build_margins_and_intervals=build,
            sample_nominal_actor=nominal,
            solve_normal_qp=solve,
            predict_from_provisional=predict,
            aggregate_bottleneck=aggregate,
            select_final_command=select,
            fast_verify_final_prediction=reverify,
            validate_hard_rows=validate,
            apply_command=apply,
            log_cycle=lambda record: events.append("log_controller_cycle"),
        )
        two_pass = execute_two_pass_control_cycle(callbacks)
        timings_ns["total_controller"] = perf_counter_ns() - total_start_ns
        context = context_box["value"]
        return IntegratedCycleResult(
            two_pass=two_pass,
            instantaneous_certificate=context.certificate,
            instantaneous_result=context.instantaneous,
            upstream_message=aggregate_box["value"],
            hard_rows=tuple(context.rows),
            hard_residuals=context.final_residuals,
            residuals_valid=context.final_residuals_valid,
            qp_status=context.qp_status,
            callback_order=tuple(events),
            supervisor_decision=decision_box["value"],
            barriers={
                "collision_h": context.collision["h"],
                "collision_psi1": context.collision["h_dot"],
                "temperature_h": (
                    context.thermal["h"] if context.thermal is not None else context.low_speed["g_T0_K"]
                ),
                "temperature_psi1": context.thermal["h_dot"] if context.thermal is not None else context.low_speed["g_T0_K"],
                "fade_h": context.fade["h"],
                "auxiliary_h": context.auxiliary["h"],
                **context.margins,
            },
            timings_s={name: value * 1e-9 for name, value in timings_ns.items()},
        )

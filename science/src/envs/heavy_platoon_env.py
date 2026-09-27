"""Synchronous heterogeneous Phase 2B platoon environment."""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

from controllers.debug_nominal import (
    AuxiliaryFirstDescentController,
    NominalDebugController,
    SpeedTrackingDebugController,
    ZeroCommandController,
)
from controllers.integrated_safety_controller import (
    ControllerCycleInput,
    DebugSafetyControllerConfig,
    IntegratedCycleResult,
    IntegratedSafetyController,
)
from controllers.predecessor_following import PredecessorFollowingDebugController
from envs.leader_profile import LeaderState
from envs.observations import (
    CentralizedTrainingState,
    SafetyControllerInformation,
    build_augmented_state,
    conservative_predecessor_replacement,
    expand_predecessor_packet,
    local_actor_observation,
    spacing_m,
)
from envs.scenario import PlatoonScenario
from network.message_buffer import DebugConsistencyTolerances, LocalMotionSensing, MessageBuffer
from network.packet import V2VPacket
from network.v2v_channel import V2VChannel
from safety.fleet_reserve import UpstreamSafetyMessage, compute_fleet_reserve
from simulator.gearbox import GearState, advance_dwell_timer, GearboxParameters
from simulator.integrators import integrate_zero_order_hold
from simulator.truck_dynamics import (
    EnvironmentInput,
    VehicleCommand,
    VehicleState,
    derivative_to_vector,
    nonbraking_force_N,
    state_from_vector,
    state_to_vector,
    vehicle_rhs,
    auxiliary_force_envelope_N,
)


@dataclass(frozen=True)
class Phase2BLogRecord:
    time_s: float
    vehicle_id: int
    physical_state: VehicleState
    augmented_state: object
    nominal_action: tuple[float, float]
    provisional_action: tuple[float, float]
    final_action: tuple[float, float]
    barriers: dict[str, float]
    hard_row_residuals: dict[str, float]
    delta_f_N: float
    delta_a_N: float
    M_mps2: float
    rho: float
    g_T0_K: float | None
    rho_H_cert: float
    centralized_fleet_rho_H_cert: float
    local_recursive_rho_up: float
    critical_vehicle_id: int
    critical_prediction_step: int
    critical_component: str
    message_age_s: float
    message_valid: bool
    supervisor_mode: str
    reverification_performed: bool
    qp_status: str
    final_residuals_valid: bool
    centralized_equals_local_asserted: bool
    data_provenance: str = "synthetic_debug"
    paper_eligible: bool = False

    def __post_init__(self) -> None:
        if self.data_provenance == "synthetic_debug" and self.paper_eligible:
            raise ValueError("synthetic-debug logs cannot be paper eligible")
        if self.paper_eligible and self.data_provenance != "literature_calibrated":
            raise ValueError("paper-eligible logs require literature calibration")


@dataclass(frozen=True)
class PlatoonStepResult:
    time_s: float
    states: tuple[VehicleState, ...]
    controller_results: tuple[IntegratedCycleResult, ...]
    logs: tuple[Phase2BLogRecord, ...]
    centralized_fleet_rho_H_cert: float
    head_local_rho_up: float


class HeavyPlatoonEnv:
    """Vehicle 0 is the leader; controlled heavy trucks are ids 1..N."""

    def __init__(
        self,
        scenario: PlatoonScenario,
        nominal_controllers: Sequence[NominalDebugController] | None = None,
        safety_configs: Sequence[DebugSafetyControllerConfig] | None = None,
        safety_controllers: Sequence[IntegratedSafetyController] | None = None,
    ) -> None:
        self.scenario = scenario
        self.controlled_truck_count = scenario.controlled_truck_count
        self.total_vehicle_count = self.controlled_truck_count + 1
        self.time_s = 0.0
        self.states = list(scenario.initial_states)
        self.previous_commands = [(0.0, 0.0) for _ in self.states]
        self.gear_states = [GearState(2, 10.0) for _ in self.states]
        self.gearbox_parameters = GearboxParameters(
            gears=(1, 2, 3),
            permitted_shifts=frozenset({(1, 2), (2, 1), (2, 3), (3, 2)}),
            minimum_dwell_s=1.0,
        )
        self.channel = V2VChannel(scenario.channel_config)
        tolerances = DebugConsistencyTolerances(
            20.0, 8.0, scenario.data_provenance, scenario.paper_eligible
        )
        self.buffers = {
            vehicle_id: MessageBuffer(0.3, tolerances)
            for vehicle_id in range(1, self.controlled_truck_count + 1)
        }
        if safety_controllers is not None and safety_configs is not None:
            raise ValueError("provide safety controllers or safety configurations, not both")
        if safety_controllers is not None:
            self.controllers = list(safety_controllers)
        else:
            configs = list(safety_configs or (
                DebugSafetyControllerConfig.from_vehicle(parameters, scenario.control_step_s)
                for parameters in scenario.parameters
            ))
            self.controllers = [IntegratedSafetyController(config) for config in configs]
        if len(self.controllers) != self.controlled_truck_count:
            raise ValueError("exactly N safety controllers are required")
        if any(controller.config.data_provenance != scenario.data_provenance or controller.config.paper_eligible != scenario.paper_eligible for controller in self.controllers):
            raise ValueError("scenario and safety-controller provenance must match")
        self.nominal_controllers = list(nominal_controllers or self._default_nominal_controllers())
        if len(self.nominal_controllers) != self.controlled_truck_count:
            raise ValueError("exactly N nominal controllers are required")
        self._sequence: dict[tuple[int, int, str], int] = {}
        self.logs: list[Phase2BLogRecord] = []
        self.execution_trace: list[str] = []
        self.command_snapshot_time_by_vehicle: dict[int, float] = {}
        self.last_message_validity: tuple[bool, ...] = tuple(False for _ in self.states)

    def _default_nominal_controllers(self) -> tuple[NominalDebugController, ...]:
        if self.scenario.name == "flat_steady":
            controller: NominalDebugController = ZeroCommandController()
        elif self.scenario.name == "leader_braking":
            controller = PredecessorFollowingDebugController(5.0, 1.2, 1_500.0, 4_000.0)
        elif self.scenario.name in {"constant_descent", "communication_delay_loss"}:
            controller = SpeedTrackingDebugController(17.0, 8_000.0, 0.6)
        else:
            controller = AuxiliaryFirstDescentController(17.0, 8_000.0, 35_000.0)
        return tuple(controller for _ in range(self.controlled_truck_count))

    def _next_sequence(self, source: int, target: int, kind: str) -> int:
        key = (source, target, kind)
        value = self._sequence.get(key, 0)
        self._sequence[key] = value + 1
        return value

    def _leader_and_snapshot(self) -> tuple[LeaderState, tuple[VehicleState, ...]]:
        return self.scenario.leader.state_at(self.time_s), tuple(self.states)

    def _predecessor_state(
        self, vehicle_id: int, leader: LeaderState, snapshot: tuple[VehicleState, ...]
    ) -> tuple[float, float, float, float]:
        if vehicle_id == 1:
            return (
                leader.position_m,
                leader.speed_mps,
                leader.acceleration_mps2,
                self.scenario.leader_length_m,
            )
        predecessor = snapshot[vehicle_id - 2]
        parameters = self.scenario.parameters[vehicle_id - 2]
        acceleration = self._physical_acceleration(vehicle_id - 1, predecessor)
        return predecessor.position_m, predecessor.speed_mps, acceleration, parameters.length_m

    def _physical_acceleration(self, vehicle_id: int, state: VehicleState) -> float:
        parameters = self.scenario.parameters[vehicle_id - 1]
        environment = EnvironmentInput(
            self.scenario.road.grade_rad(state.position_m),
            self.scenario.ambient_temperature_K,
            self.scenario.residual_heat_W,
        )
        command = VehicleCommand(self.scenario.drive_force_N[vehicle_id - 1], 0.0, 0.0)
        force = nonbraking_force_N(state, command, environment, parameters)
        return (force - state.friction_force_N - state.auxiliary_force_N) / parameters.mass_kg

    def _dispatch_due_packets(
        self, leader: LeaderState, snapshot: tuple[VehicleState, ...]
    ) -> None:
        for packet in self.channel.poll(self.time_s):
            if packet.target_vehicle_id not in self.buffers:
                continue
            sensing = None
            if packet.packet_kind == "motion":
                position, speed, _acceleration, _length = self._predecessor_state(
                    packet.target_vehicle_id, leader, snapshot
                )
                sensing = LocalMotionSensing(position, speed)
            self.buffers[packet.target_vehicle_id].ingest(packet, self.time_s, sensing)

    def _broadcast_motion(
        self, leader: LeaderState, snapshot: tuple[VehicleState, ...]
    ) -> None:
        for vehicle_id in range(1, self.controlled_truck_count + 1):
            position, speed, acceleration, _length = self._predecessor_state(vehicle_id, leader, snapshot)
            source = vehicle_id - 1
            jerk = self.scenario.leader.jerk_bound_mps3 if source == 0 else 3.0
            packet = V2VPacket(
                source_vehicle_id=source,
                target_vehicle_id=vehicle_id,
                generation_time_s=self.time_s,
                sequence_number=self._next_sequence(source, vehicle_id, "motion"),
                packet_kind="motion",
                position_m=position,
                speed_mps=speed,
                acceleration_mps2=acceleration,
                declared_jerk_bound_mps3=jerk,
                data_provenance=self.scenario.data_provenance,
                paper_eligible=self.scenario.paper_eligible,
            )
            self.channel.send(packet, self.time_s)

    def _information_for(
        self, vehicle_id: int, leader: LeaderState, snapshot: tuple[VehicleState, ...]
    ) -> tuple[SafetyControllerInformation, object]:
        follower = snapshot[vehicle_id - 1]
        pred_position, pred_speed, pred_acceleration, pred_length = self._predecessor_state(
            vehicle_id, leader, snapshot
        )
        source = vehicle_id - 1
        packet, reason, age = self.buffers[vehicle_id].latest_valid(source, "motion", self.time_s)
        if packet is not None:
            reachable = expand_predecessor_packet(packet, self.time_s, 0.5)
            message_valid = True
            message_age = packet.age_s(self.time_s)
            pred_speed_for_chi = 0.5 * (reachable.speed_mps.low + reachable.speed_mps.high)
            pred_accel_for_chi = 0.5 * (
                reachable.acceleration_mps2.low + reachable.acceleration_mps2.high
            )
        else:
            reachable = conservative_predecessor_replacement(
                pred_position, pred_speed, self.time_s, 4.0, reason
            )
            message_valid = False
            message_age = self.controllers[vehicle_id - 1].config.maximum_message_age_s
            pred_speed_for_chi = pred_speed
            pred_accel_for_chi = pred_acceleration
        gap = spacing_m(pred_position, follower.position_m, pred_length)
        chi = build_augmented_state(
            follower,
            self.previous_commands[vehicle_id - 1],
            self.gear_states[vehicle_id - 1],
            message_age,
            gap,
            pred_speed_for_chi,
            pred_accel_for_chi,
        )
        information = SafetyControllerInformation(chi, reachable, message_valid, reason)
        return information, local_actor_observation(chi)

    def _received_successor(self, vehicle_id: int) -> UpstreamSafetyMessage | None:
        if vehicle_id >= self.controlled_truck_count:
            return None
        packet, _reason, _age = self.buffers[vehicle_id].latest_valid(
            vehicle_id + 1, "bottleneck", self.time_s
        )
        if packet is None or packet.rho_H_cert is None:
            return None
        return UpstreamSafetyMessage(
            packet.rho_H_cert,
            packet.critical_vehicle_id if packet.critical_vehicle_id is not None else packet.source_vehicle_id,
            packet.critical_prediction_step or 0,
            packet.critical_component or "unspecified",
            packet.generation_time_s,
            False,
        )

    def _send_bottleneck(
        self,
        vehicle_id: int,
        result: IntegratedCycleResult,
        snapshot: tuple[VehicleState, ...],
    ) -> None:
        if vehicle_id <= 1:
            return
        state = snapshot[vehicle_id - 1]
        upstream = result.upstream_message
        packet = V2VPacket(
            source_vehicle_id=vehicle_id,
            target_vehicle_id=vehicle_id - 1,
            generation_time_s=self.time_s,
            sequence_number=self._next_sequence(vehicle_id, vehicle_id - 1, "bottleneck"),
            packet_kind="bottleneck",
            position_m=state.position_m,
            speed_mps=state.speed_mps,
            acceleration_mps2=self._physical_acceleration(vehicle_id, state),
            declared_jerk_bound_mps3=3.0,
            rho_H_cert=upstream.rho_upstream_H,
            critical_vehicle_id=upstream.critical_vehicle_id,
            critical_prediction_step=upstream.critical_future_step,
            critical_component=upstream.limiting_physical_constraint,
            data_provenance=self.scenario.data_provenance,
            paper_eligible=self.scenario.paper_eligible,
        )
        self.channel.send(packet, self.time_s)

    def centralized_training_state(self) -> CentralizedTrainingState:
        leader = self.scenario.leader.state_at(self.time_s)
        diagnostic = self.logs[-1].centralized_fleet_rho_H_cert if self.logs else None
        return CentralizedTrainingState(
            (leader.position_m, leader.speed_mps, leader.acceleration_mps2),
            tuple(self.states),
            self.last_message_validity,
            diagnostic,
        )

    def current_local_observations(self) -> tuple[object, ...]:
        """Return only causally available local actor observations.

        This read-only interface does not send or poll packets.  The safety
        controller retains its separate, richer information path inside
        :meth:`step`.
        """
        leader, snapshot = self._leader_and_snapshot()
        return tuple(
            self._information_for(vehicle_id, leader, snapshot)[1]
            for vehicle_id in range(1, self.controlled_truck_count + 1)
        )

    def step(self) -> PlatoonStepResult:
        leader, snapshot = self._leader_and_snapshot()
        self.execution_trace = ["snapshot_all_vehicles"]
        self.command_snapshot_time_by_vehicle = {}
        self._dispatch_due_packets(leader, snapshot)
        self._broadcast_motion(leader, snapshot)
        self._dispatch_due_packets(leader, snapshot)
        information: dict[int, tuple[SafetyControllerInformation, object]] = {
            vehicle_id: self._information_for(vehicle_id, leader, snapshot)
            for vehicle_id in range(1, self.controlled_truck_count + 1)
        }
        self.last_message_validity = tuple(
            information[vehicle_id][0].message_valid
            for vehicle_id in range(1, self.controlled_truck_count + 1)
        )
        results_by_id: dict[int, IntegratedCycleResult] = {}
        for vehicle_id in range(self.controlled_truck_count, 0, -1):
            info, observation = information[vehicle_id]
            state = snapshot[vehicle_id - 1]
            parameters = self.scenario.parameters[vehicle_id - 1]
            grade = self.scenario.road.grade_rad(state.position_m)
            environment = EnvironmentInput(
                grade,
                self.scenario.ambient_temperature_K,
                self.scenario.residual_heat_W,
            )
            drive_only = VehicleCommand(self.scenario.drive_force_N[vehicle_id - 1], 0.0, 0.0)
            nonbraking = nonbraking_force_N(state, drive_only, environment, parameters)
            auxiliary_limit = auxiliary_force_envelope_N(parameters, state.speed_mps)
            if self.scenario.name == "auxiliary_limitation":
                auxiliary_limit *= 0.25
            cycle = ControllerCycleInput(
                vehicle_id=vehicle_id,
                time_s=self.time_s,
                information=info,
                observation=observation,
                nominal_controller=self.nominal_controllers[vehicle_id - 1],
                nonbraking_force_N=nonbraking,
                ambient_temperature_K=self.scenario.ambient_temperature_K,
                residual_heat_W=self.scenario.residual_heat_W,
                grade_rad=grade,
                auxiliary_limit_N=auxiliary_limit,
                predecessor_jerk_bound_mps3=(
                    self.scenario.leader.jerk_bound_mps3 if vehicle_id == 1 else 3.0
                ),
                received_successor=self._received_successor(vehicle_id),
                is_tail_vehicle=vehicle_id == self.controlled_truck_count,
            )
            self.command_snapshot_time_by_vehicle[vehicle_id] = self.time_s
            self.execution_trace.append(f"compute_verified_command:{vehicle_id}")
            result = self.controllers[vehicle_id - 1].run_cycle(cycle)
            results_by_id[vehicle_id] = result
            self._send_bottleneck(vehicle_id, result, snapshot)
            self._dispatch_due_packets(leader, snapshot)

        results = tuple(results_by_id[index] for index in range(1, self.controlled_truck_count + 1))
        fleet = compute_fleet_reserve(tuple(result.two_pass.final_prediction for result in results))
        centralized = fleet.fleet_reserve
        head_local = results[0].upstream_message.rho_upstream_H
        local_records: list[Phase2BLogRecord] = []
        for vehicle_id, result in enumerate(results, start=1):
            info, _observation = information[vehicle_id]
            certificate = result.instantaneous_certificate
            prediction = result.two_pass.final_prediction
            record = Phase2BLogRecord(
                time_s=self.time_s,
                vehicle_id=vehicle_id,
                physical_state=snapshot[vehicle_id - 1],
                augmented_state=info.chi,
                nominal_action=result.two_pass.nominal_command,
                provisional_action=result.two_pass.provisional_command,
                final_action=result.two_pass.final_command,
                barriers=result.barriers,
                hard_row_residuals=result.hard_residuals,
                delta_f_N=certificate.delta_f_N,
                delta_a_N=certificate.delta_a_N,
                M_mps2=certificate.M_mps2,
                rho=certificate.rho,
                g_T0_K=certificate.g_T0_K,
                rho_H_cert=prediction.rho_cert_H,
                centralized_fleet_rho_H_cert=centralized,
                local_recursive_rho_up=result.upstream_message.rho_upstream_H,
                critical_vehicle_id=result.upstream_message.critical_vehicle_id,
                critical_prediction_step=result.upstream_message.critical_future_step,
                critical_component=result.upstream_message.limiting_physical_constraint,
                message_age_s=info.chi.message_age_s,
                message_valid=info.message_valid,
                supervisor_mode=result.two_pass.mode,
                reverification_performed=result.two_pass.prediction_reverified,
                qp_status=result.qp_status,
                final_residuals_valid=result.residuals_valid,
                centralized_equals_local_asserted=(
                    self.scenario.channel_config.delay_model == "zero"
                    and self.scenario.channel_config.packet_loss_probability == 0.0
                    and not self.scenario.channel_config.forced_loss_sequences
                    and not self.scenario.channel_config.forced_burst_starts
                ),
                data_provenance=self.scenario.data_provenance,
                paper_eligible=self.scenario.paper_eligible,
            )
            local_records.append(record)
            self.logs.append(record)

        self.execution_trace.append("all_commands_verified")
        next_states: list[VehicleState] = []
        for vehicle_id, result in enumerate(results, start=1):
            state = snapshot[vehicle_id - 1]
            parameters = self.scenario.parameters[vehicle_id - 1]
            final = result.two_pass.final_command
            auxiliary_limit = auxiliary_force_envelope_N(parameters, state.speed_mps)
            if self.scenario.name == "auxiliary_limitation":
                auxiliary_limit *= 0.25
            command = VehicleCommand(
                self.scenario.drive_force_N[vehicle_id - 1], final[0], final[1]
            )

            def rhs(_time: float, vector: tuple[float, ...], held: VehicleCommand) -> tuple[float, ...]:
                current = state_from_vector(vector)
                environment = EnvironmentInput(
                    self.scenario.road.grade_rad(current.position_m),
                    self.scenario.ambient_temperature_K,
                    self.scenario.residual_heat_W,
                )
                return derivative_to_vector(
                    vehicle_rhs(current, held, environment, parameters, auxiliary_limit)
                )

            self.execution_trace.append(f"integrate_vehicle:{vehicle_id}")
            trajectory = integrate_zero_order_hold(
                rhs,
                state_to_vector(state),
                (command,),
                self.scenario.control_step_s,
                self.scenario.simulation_step_s,
                data_provenance=self.scenario.data_provenance,
                paper_eligible=self.scenario.paper_eligible,
            )
            next_states.append(state_from_vector(trajectory.samples[-1].state))
        self.states = next_states
        self.previous_commands = [result.two_pass.final_command for result in results]
        self.gear_states = [
            advance_dwell_timer(state, self.scenario.control_step_s, self.gearbox_parameters)
            for state in self.gear_states
        ]
        self.time_s += self.scenario.control_step_s
        self.execution_trace.append("advance_shared_time")
        return PlatoonStepResult(
            self.time_s,
            tuple(self.states),
            results,
            tuple(local_records),
            centralized,
            head_local,
        )

    def run(self, steps: int) -> tuple[PlatoonStepResult, ...]:
        if steps < 1:
            raise ValueError("steps must be positive")
        return tuple(self.step() for _ in range(steps))

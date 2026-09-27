# Simulation readiness report

**READINESS MODE: `SCAFFOLD_ONLY_DATA_REQUIRED`**

This is a strict evidence gate. It does not infer defaults, fill parameters, or accept synthetic-debug data as production evidence.

## Manifest

- Expected production manifest: `data/production/physical_manifest.json`
- Manifest found: `false`
- Required entries: `40`
- Fully validated entries: `0`

## Repository capability finding

Reference implementations exist for the safety core, instantaneous/predictive certificates, interval reachability, set-valued affine backup, distributed bottleneck aggregation, two-pass callback order, supervisor logic, actor parameterization and scalar learning losses.

The Phase 2A continuous simulator, Phase 2B heterogeneous platoon/V2V/safety integration, and Phase 2C-1 NumPy MAPPO core exist and pass synthetic-debug software tests, but are not physically or network calibrated and are not paper eligible. Differentiable-QP backward propagation, matched baselines, immutable experiment logger, production experiment manager, statistics pipeline, generated-table pipeline, paper-eligible checkpoints and claim-bearing logs remain absent. The reference safety core includes the explicit low-speed state-only `g_T0` gate. See `docs/repository_inventory.md`.

## Missing requirements

- `vehicle.mass` — production manifest is absent
- `vehicle.length` — production manifest is absent
- `vehicle.payload_configuration` — production manifest is absent
- `vehicle.rolling_resistance` — production manifest is absent
- `vehicle.CdA` — production manifest is absent
- `friction_brake.maximum_force` — production manifest is absent
- `friction_brake.actuator_lag` — production manifest is absent
- `auxiliary_brake.actuator_lag` — production manifest is absent
- `auxiliary_brake.force_torque_map` — production manifest is absent
- `auxiliary_brake.power_map` — production manifest is absent
- `auxiliary_brake.speed_dependence` — production manifest is absent
- `auxiliary_brake.gear_dependence` — production manifest is absent
- `auxiliary_brake.gear_dwell_shift_logic` — production manifest is absent
- `thermal.capacity` — production manifest is absent
- `thermal.cooling_coefficient` — production manifest is absent
- `thermal.eta` — production manifest is absent
- `thermal.ambient_range` — production manifest is absent
- `thermal.critical_temperature` — production manifest is absent
- `thermal.residual_heat_model_bounds` — production manifest is absent
- `fade.phi_T` — production manifest is absent
- `fade.calibration_range` — production manifest is absent
- `fade.uncertainty_bound` — production manifest is absent
- `fade.validation_split` — production manifest is absent
- `road.longitudinal_position` — production manifest is absent
- `road.grade_theta` — production manifest is absent
- `road.coordinate_convention` — production manifest is absent
- `road.grade_rate_bound` — production manifest is absent
- `communication.delay_distribution_or_trace` — production manifest is absent
- `communication.packet_loss_model_or_trace` — production manifest is absent
- `communication.message_age_limit` — production manifest is absent
- `communication.stale_message_policy` — production manifest is absent
- `predecessor.emergency_deceleration_bound` — production manifest is absent
- `predecessor.acceleration_bound` — production manifest is absent
- `predecessor.jerk_bound` — production manifest is absent
- `uncertainty.robust_margin_intervals` — production manifest is absent
- `uncertainty.predictive_reachability_intervals` — production manifest is absent
- `uncertainty.integration_error_bound` — production manifest is absent
- `computation.controller_sampling_time` — production manifest is absent
- `computation.simulator_internal_step` — production manifest is absent
- `computation.target_hardware` — production manifest is absent

## Invalid requirements

None.

## Decision

Production simulation/training, publication figures, numerical result tables, and manuscript result updates are prohibited unless this gate returns `READY_FOR_FULL_SIMULATION`. Explicitly marked synthetic DEBUG software-validation smoke runs remain non-paper-eligible.

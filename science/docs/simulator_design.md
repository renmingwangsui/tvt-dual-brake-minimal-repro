# Phase 2A simulator design

## Scope and evidence status

This phase implements and validates the continuous vehicle simulator architecture only. It does not implement MAPPO, differentiable-QP training, baselines, platoon experiments or paper-result generation. The only executable configuration is `configs/debug/synthetic_debug.yaml`; its values are deliberately nonphysical software-test values carrying `data_provenance: synthetic_debug` and `paper_eligible: false`.

Production construction remains gated on the validated physical manifest listed in `docs/DATA_REQUIRED.md`. Synthetic-debug execution is not evidence for a vehicle-performance or safety-performance claim.

## State and commands

The continuous state is

\[
x=[p,v,b,r,T]^\top,
\]

where `p` is forward longitudinal position in m, `v` is nonnegative forward speed in m/s, `b` is realized friction-brake force in N, `r` is realized auxiliary-brake force in N and `T` is brake temperature in K.

The held command is `[F_d,u_f,u_a]`: non-braking drive force, friction-brake command and auxiliary-brake command, all in N. Commands `u_f` and `u_a` are not substituted for realized forces; the two actuator states evolve through their own lags.

## Continuous dynamics and sign convention

`src/simulator/truck_dynamics.py::vehicle_rhs` is the single simulator RHS:

\[
\dot p=v,
\qquad
m\dot v=F_r-b-r,
\]

\[
\tau_f\dot b=-b+u_f,
\qquad
\tau_a\dot r=-r+u_a,
\]

\[
C\dot T=\eta b v-k(T-T_a)+q_w,
\]

with

\[
F_r=F_d-mg\sin\theta-mgc_r\cos\theta-
\tfrac12\rho_{air}C_dA v^2+w.
\]

Position increases in the forward travel direction. Positive `theta` is uphill, so gravity subtracts forward force. Negative `theta` is downhill, so `-m g sin(theta)` contributes positive forward force. Both realized braking forces are nonnegative magnitudes that subtract from forward force. Positive residual heat `q_w` increases temperature.

The safety core imports `longitudinal_acceleration_mps2`, `first_order_lag_rate` and `thermal_rate_Kps` from the simulator package. It therefore evaluates the same scalar continuous equations. The interval predictor in `src/safety/reachable_set.py` remains a conservative sampled-data enclosure of that plant, not an alternative physical model.

The continuous thermal plant is intentionally distinct from the paper's low-speed frozen/ZOH safety row. The latter is a conservative one-step constraint and does not replace continuous thermal integration.

## Typed parameters

`VehicleParameters` contains:

| field | unit | meaning |
|---|---:|---|
| `mass_kg` | kg | vehicle mass |
| `length_m` | m | vehicle length |
| `rolling_resistance_coefficient` | 1 | rolling coefficient |
| `drag_area_m2` | m2 | `CdA` |
| `tau_f_s`, `tau_a_s` | s | friction and auxiliary actuator time constants |
| `thermal_capacity_J_per_K` | J/K | thermal capacity `C` |
| `cooling_W_per_K` | W/K | cooling coefficient `k` |
| `heat_fraction` | 1 | heat fraction `eta` |
| `friction_force_limit_N` | N | friction-command magnitude interface |
| `auxiliary_force_limit_N` | N | top-level auxiliary magnitude interface |
| `critical_temperature_K` | K | registered critical temperature |
| `air_density_kg_per_m3` | kg/m3 | air-density input |
| `gravity_mps2` | m/s2 | gravitational acceleration |

Production instances must be built from a validated manifest. The debug loader accepts only the explicit synthetic-debug configuration and refuses a debug configuration marked paper eligible.

## Road, fade, auxiliary and gear interfaces

- `src/envs/road_profile.py` provides constant and piecewise-linear `theta(p)` plus a named sampled-profile interpolator. Boundary behavior is explicit (`clamp` or `error`).
- `src/simulator/fade.py` provides a monotone table interface with value and derivative. Its default policy raises outside the registered temperature domain. The only permissive policy returns the minimum tabulated factor outside the domain and is explicitly named `conservative_clip`.
- `src/simulator/auxiliary_brake.py` separates the auxiliary limit from the actuator lag. Its interface accepts speed and gear and may impose a power limit. The included envelope is labeled synthetic-debug and contains no manufacturer map.
- `src/simulator/gearbox.py` tracks discrete gear and dwell time, validates permitted transitions and resets the dwell timer after a shift.

## Numerical integration and command timing

`rk4_step` is fixed-step classical RK4. `integrate_reference` is an adaptive Dormand--Prince 5(4) method with local error control and is used only for numerical validation. `integrate_zero_order_hold` splits every controller interval into simulator steps, shortens the last internal step to land exactly on the controller boundary and supplies one unchanged command throughout each open controller interval. At a discontinuity, numerical RHS calls may observe the old right limit and the new left limit at the same timestamp; no command is used outside its adjacent interval.

## Module dependency diagram

```text
configs/debug/synthetic_debug.yaml
             |
             v
simulator.truck_dynamics -----> simulator.thermal
       |       |                         ^
       |       +---- shared primitives -+---- safety_core
       |
       +---- envs.road_profile
       +---- simulator.auxiliary_brake <---- simulator.gearbox
       +---- simulator.fade
       +---- simulator.integrators (RK4 / reference / ZOH)
```

No simulator module imports RL, training, baseline or paper-result code.

## Relationship to manuscript equations

The five-state RHS implements the manuscript sampled plant before discretization. `nonbraking_force_N` implements the stated `F_r` decomposition. The actuator and thermal functions are imported by `safety_core.state_derivatives`; `safety_core.continuous_jerk` is computed from the same actuator rates. Low-speed ZOH, HOCBF, fade-CBF and predictive interval certificates remain safety calculations layered on this shared plant.

## DEBUG versus production data

Every `SimulationResult` contains `data_provenance` and `paper_eligible`. A result with `data_provenance == synthetic_debug` raises immediately if `paper_eligible` is true. The synthetic fade and auxiliary interfaces enforce the same rule. No conversion path from debug output to a paper-eligible result is provided.

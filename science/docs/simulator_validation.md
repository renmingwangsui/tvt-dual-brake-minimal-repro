# Phase 2A simulator validation

## Validation status

All five Phase 2A simulator scripts pass. All twelve pre-existing theory/reference scripts also pass after the shared-dynamics refactor. The production-result provenance gate continues to fail closed because `results/manifest.json` is absent.

This is numerical and software validation with synthetic-debug inputs. It is **not physical model validation**, calibration evidence, vehicle-test evidence or paper-eligible experimental evidence.

## Tests executed

| script | principal coverage |
|---|---|
| `test_simulator_dynamics.py` | stationary sanity, flat/downhill/uphill signs, friction and auxiliary braking effects, exact force identity, shared safety/simulator primitives |
| `test_simulator_actuators.py` | analytical friction and auxiliary lags, speed/gear/power envelope, dwell transitions, fade domain/value/derivative policy |
| `test_simulator_thermal.py` | heating, cooling, zero-speed friction power, residual heat sign and analytical cooling |
| `test_simulator_road.py` | constant grade, piecewise interpolation, transitions and boundary policy |
| `test_simulator_integration.py` | RK4 one-step identity/order, adaptive reference, exact ZOH boundaries, vehicle timestep convergence and debug provenance |

## Analytical comparisons

- Friction and auxiliary states were integrated for constant commands and compared with `u+(x0-u) exp(-t/tau)`. The acceptance criteria are relative error `2e-10` and absolute error `1e-7 N`.
- Cooling-only integration was compared with `Ta+(T0-Ta) exp(-k t/C)`. The acceptance criteria are relative error `2e-11` and absolute error `2e-9 K`.
- RK4 was checked against its exact one-step fourth-order polynomial for `y'=y`; the adaptive reference result at 1 s is within `1e-10` of `e`.

## Vehicle timestep convergence

Three 3 s synthetic-debug trajectories used different initial positions, speeds, brake temperatures and drive/friction/auxiliary command allocations over a position-dependent grade. Commands were zero-order held at `0.1 s`. Fixed-step results were compared with the adaptive Dormand--Prince reference using identical command boundaries. The table reports the worst absolute component error over the three trajectories.

Absolute final-state errors `[position m, speed m/s, friction N, auxiliary N, temperature K]` were:

| simulator step | position | speed | friction force | auxiliary force | temperature |
|---:|---:|---:|---:|---:|---:|
| `dt = 0.04 s` | 2.4355e-6 | 9.2525e-7 | 1.5221e-3 | 1.8064e-4 | 1.2559e-7 |
| `dt/2 = 0.02 s` | 8.4356e-7 | 3.3235e-7 | 1.1237e-4 | 1.3520e-5 | 2.7322e-8 |
| `dt/4 = 0.01 s` | 8.0404e-7 | 3.1281e-7 | 6.8761e-6 | 8.3331e-7 | 2.8831e-8 |

Position, speed and both force-state errors decrease under both halvings. The worst temperature error decreases from `1.2559e-7 K` to `2.7322e-8 K`, then changes to `2.8831e-8 K`; this `1.51e-9 K` increase is treated as the adaptive-reference/roundoff floor, not advertised as monotone temperature convergence. The normalized worst error across all five registered component tolerances decreases `1.256 -> 0.422 -> 0.402`. The `dt/4` acceptance tolerances are respectively `2e-6 m`, `2e-6 m/s`, `2e-2 N`, `2e-2 N` and `1e-7 K`. All `dt/4` components pass. These are software-regression thresholds and do not certify production integration error.

## ZOH validation

The non-dividing internal step `0.03 s` was tested against a `0.1 s` controller interval. The integrator shortened the fourth step to land exactly at each boundary. The integrated scalar command area was exact to `1e-12`, and every RHS observation used the command for the current interval or one of the two adjacent one-sided values exactly at a discontinuity.

## Failures and limitations

Final implementation failures: none in the five simulator tests. During test development, one interpolation assertion incorrectly required bitwise decimal equality and one ZOH assertion did not admit the two valid one-sided evaluations at a discontinuity; both test-harness conditions were corrected without changing equations or loosening numerical convergence tolerances.

Remaining limitations:

- no validated production physical manifest;
- no calibrated road, fade, retarder, gear or residual-heat data;
- no identified uncertainty or production integration-error certificate;
- no vehicle, HIL, field or real-time validation;
- no platoon environment, V2V channel, controller training, baselines or paper experiments;
- debug parameter magnitudes and debug trajectories must not be cited as physical performance.

## Provenance conclusion

`DEBUG DATA USED = YES`. All debug outputs preserve `data_provenance: synthetic_debug` and `paper_eligible: false`; attempts to construct a paper-eligible debug result raise an exception. `PAPER-ELIGIBLE RESULTS GENERATED = NO`.

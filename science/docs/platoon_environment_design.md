# Phase 2B heterogeneous platoon environment

## Scope and indexing

Vehicle `0` is the externally configured leader. Controlled heterogeneous heavy trucks use ids `1..N`; the total vehicle count is therefore `N+1`. `PlatoonScenario` validates that exactly `N` parameter sets, physical states and drive-force entries exist. Construction and execution tests cover `N = 1, 3, 5, 10, 20, 40`.

Each controlled plant retains the Phase 2A state `[p,v,b,r,T]`. Realized brake forces `b` and `r` remain continuous actuator states and are never replaced with commands. The environment calls the validated Phase 2A `vehicle_rhs` and `integrate_zero_order_hold`; it contains no duplicate plant equations.

## Geometry and canonical controller state

Forward position increases along travel. For follower `i`, bumper spacing is

`d_i = p_(i-1) - p_i - L_(i-1)`.

Positive `d_i` is separation, zero is bumper contact and negative is overlap. The single constructor `envs.observations.build_augmented_state` maps physical state, previous commands, gear/dwell state, message age, spacing and predecessor motion into the existing `AugmentedHybridState`, which is the executable manuscript state `chi_i,k`. No safety module reconstructs `chi` independently.

## Observation separation

Three typed interfaces remain distinct:

- `LocalActorObservation`: own realized state, gap, relative speed, previous local commands, gear, message age and received safety intent. It contains no future state, realized uncertainty, global fleet state or centralized minimum.
- `SafetyControllerInformation`: canonical `chi`, causal predecessor reachable intervals and packet validity/replacement reason.
- `CentralizedTrainingState`: leader and all controlled states, communication validity and optional centralized diagnostic. It is reserved for future centralized training/evaluation and is not passed to a decentralized nominal controller.

MAPPO is not implemented in Phase 2B.

## Leader profiles

`leader_profile.py` provides constant speed, smooth acceleration, smooth deceleration and a smooth emergency braking pulse. Each exposes position, speed and acceleration from a common time argument plus a declared jerk-bound interface. Followers receive current local sensing and timestamped packets; they never receive a future leader trajectory.

## Synchronous sampled-data semantics

At control time `k`, `HeavyPlatoonEnv.step`:

1. freezes leader and all `N` controlled physical states;
2. processes due packets and creates all controller observations from that same snapshot;
3. computes and validates commands, tail-to-head only to permit causal zero-delay bottleneck packet propagation;
4. computes the centralized minimum for diagnostics only;
5. after every vehicle has a verified command, integrates all plants over the same ZOH interval;
6. advances the shared clock once.

No vehicle is integrated to `k+1` while another controller is still reading state `k`. The execution trace and per-vehicle snapshot times are regression tested.

## Heterogeneity and DEBUG scenarios

Phase 2B varies mass, actuator lags and thermal capacity across vehicles using explicitly synthetic-debug perturbations. Six short `N=3` scenarios exercise flat/steady motion, constant descent, leader braking, hot initial brakes, auxiliary limitation, and delay/loss. `N=5,10,20,40` receive short execution smoke tests.

These runs carry `data_provenance = synthetic_debug` and `paper_eligible = false`. They are software integration validation, not physical vehicle validation or paper-performance evidence.

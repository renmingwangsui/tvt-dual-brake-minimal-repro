# Pre-revision repository audit

Audit scope: `manuscript/main.tex`, every section/table/figure source, bibliography, executable safety/RL reference code, QP rows, YAML plans, algebra scripts, generated normalized CSV, tests, build log, and evidence documents. The vendored `IEEEtran.cls` is treated as an external template. No vehicle logs, trained checkpoints, physical parameter files, or target-hardware traces are present.

## 1. Claimed novelty

The defensible claim is the collision--thermal feasibility conflict: the collision HOCBF imposes a lower bound on combined braking, while thermal/fade rows cap friction braking and mode/rate/envelope rows cap auxiliary braking. Under the declared interval-plus-one-coupled-row structure, the exact signed reserve is `M=A_f U_f+A_a U_a-D_c`. Conflict-aware learning uses predicted next-state reserve to anticipate loss of feasibility. Generic MAPPO, CBF/HOCBF filtering, differentiable QP, cooperative platoon safety, heavy trucks, auxiliary braking, and Chebyshev centers are not novel.

## 2. Mathematical gaps

- Core collision, thermal, fade, auxiliary, jerk, ZOH, reserve, and sampled-data equations are now implemented and regression-tested.
- The uncertainty vector and its interval components are not yet enumerated one-by-one in the paper; the source of each bound and its deterministic/probabilistic status must be explicit.
- The exact reserve theorem is valid only while every noncollision hard row reduces to an individual command interval and the collision row is the sole coupled hard row. Any future power-sharing, tire-friction-ellipse, or fleet-coupled row requires a support-function/LP-dual certificate.
- The backup claim is conditional, not global recursive feasibility; a nontrivial invariant backup domain has not been established from data.
- Physical parameter values, residual-rate bounds, and model-error bounds remain `[DATA REQUIRED]`.

## 3. Notation gaps

- A Markdown symbol table exists, but the required machine-readable `docs/symbol_audit.csv` does not.
- First-use locations and hard-guarantee status need a single authoritative CSV.
- `H` has already been separated into `H_QP`, `H_pred`, and `N_rec`; this separation must be retained.
- Units of every row residual must remain attached to logs; normalized diagnostics must not be confused with the physical reserve `M`.

## 4. Equation/code mismatches

- The manuscript, `src/safety_core.py`, and randomized tests agree on the principal affine rows.
- `src/actor_parameterization.py` is a framework-independent reference, not a complete MAPPO trainer; no optimizer, replay/rollout buffer, GAE implementation, or automatic-differentiation QP backend is present.
- `project_weighted_2d` is an exact two-input projection reference, not a batched differentiable training solver.
- The old `experiments/verify_constraint_algebra.py` contains 40,000 legacy checks; the authoritative suite is `tests/test_equation_regression.py` with 50,000 identities plus boundary cases.
- Figure generation currently validates provenance/schema and then stops; it deliberately does not plot absent results.

## 5. Missing experiments

- DiffQP versus stop-gradient with an identical forward QP.
- Friction-only, fixed auxiliary-first, learned dual allocation, and centralized allocator/MPC.
- Collision-only, collision+thermal, and collision+thermal+fade barrier ablations.
- Four robust-margin variants, four cooperation variants, the complete OOD matrix, and scalability at `N={3,5,10,20,40}`.
- All eight required real-log figures, matched scenarios, at least ten independent training seeds (unless prospectively changed), bootstrap intervals, effect sizes, Holm correction, and named-hardware latency.

## 6. Unsupported claims

- No trained-policy superiority, vehicle-level safety rate, string stability, operational benefit, HIL/field validity, target-hardware real-time capability, or global recursive feasibility is supported.
- The 25,921-point normalized sweep is only an algebraic monotonicity test and cannot support a vehicle or learning claim.
- Zero observed failures, if later obtained, will remain finite-suite evidence rather than an unconditional guarantee.
- Every claim requiring missing logs/data must remain `[EXPERIMENT REQUIRED]` or `[DATA REQUIRED]`.

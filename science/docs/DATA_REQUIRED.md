# DATA_REQUIRED manifest

Repository search found no provenance-backed files sufficient for system-level simulation. Required inputs are:

- Per-truck mass, geometry, rolling/aerodynamic resistance, payload and fleet-order configurations.
- Thermal capacity `C`, cooling coefficient `k`, heat fraction `eta`, ambient range, residual heat and derivative bounds.
- Calibrated fade map `phi(T)`, critical temperature and validation split.
- Friction force limits and actuator-lag identification; no independent physical command-slew constraint is assumed.
- Retarder force/torque/power maps by speed and gear, gear dwell/shift logic, and auxiliary lag.
- Surveyed long-descent grade profiles with coordinate convention and grade-rate bounds.
- Communication delay/loss traces or a validated stochastic model, message-age limits and predecessor motion bounds.
- Continuous-time integration-error and componentwise deterministic uncertainty bounds.
- Paper-eligible MAPPO training/checkpoints, the Phase 2C-2 identical-forward stop-gradient variant and immutable production configuration hashes (the existing Phase 2C-1 DEBUG checkpoint is ineligible).
- Target hardware identity and timing instrumentation.

Until these inputs are supplied, the mandatory A--I matched experiments and P1--P8 real-log figures cannot be executed without fabricating evidence.

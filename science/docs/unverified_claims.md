# Unverified claims and missing evidence

- No validated heavy-truck parameter file, fade/retarder/gear map, road profile, uncertainty calibration, or continuous-time integration-error certificate was supplied.
- No paper-eligible trained MAPPO checkpoint or production autodiff-QP trainer was supplied. The Phase 2C-1 checkpoint is synthetic DEBUG software validation only.
- No provenance-verified per-step/per-episode logs exist for the mandatory matched ablations, OOD matrix, fleet-size scaling, early-warning, critical-vehicle/limit attribution, or runtime percentiles.
- Therefore no performance improvement, violation reduction, warning-time gain, real-time deadline, string-stability benefit, or deployment guarantee is claimed.
- The included normalized sweep is only an algebraic feasibility-envelope test. Monte Carlo tests are empirical containment checks, not certificates.
- Interval dependency may cause false alarms; P7 must determine whether affine arithmetic or zonotopes are warranted.
- The ZOH and digital formulas are executable, but their numeric guarantee requires validated component bounds.
- No global recursive-feasibility or time-to-conflict theorem is claimed.
- A negative `rho_cert_H` is not evidence of a physical infeasibility realization; it remains an inconclusive interval warning unless `rho_true_H` is solved exactly.
- The affine-plus-saturation reference backup has a sound set extension, but any production backup with different branches must supply and test its own interval extension.
- Zero-delay recursive min aggregation equals the centralized minimum only after complete chain propagation. Delayed/lost-message deployment does not provide every truck the instantaneous global minimum.
- Normalization sign invariance is tested, but trigger-time and learning conclusions across physical scale triples require real logs.

# Remaining gaps before an IEEE T-ITS submission

## Predictive-reserve extension

- Identify every interval in ReachabilityUncertainty, including a continuous-to-sampled integration-error allowance; unit-test coverage alone does not validate these bounds.
- Integrate the Phase 2C-1 MAPPO core with the future production/autodiff safety-learning path. The current repository contains only a synthetic DEBUG smoke checkpoint, not a trained paper-eligible controller.
- Register $H_{\rm pred}$, $H_L$, trigger/recovery thresholds, soft-min temperature, loss weights, truncated-BPTT length, active-set tolerance, and gradient clip before training.
- Produce Predictive P1--P4 only from hash-verified per-step/per-episode logs. The supplied generator fails closed while logs are absent.
- Run the matched instantaneous-only, predictive-vehicle, and predictive+fleet ablation with the same seeds, transitions, network capacity, reward terms, and journeys.

## Calibration limitations retained after Phase 2M-LIT-3

- No required literature-calibrated parameter remains unresolved. The fade curve is explicitly representative rather than material-identical, and the retarder installation is an explicit cross-source heavy-duty model mapping.
- Obtain material-matched dynamometer fade data and vehicle-specific retarder transient/installation data before making identity-specific vehicle claims.
- Complete calibrated component-wise uncertainty intervals and held-out evidence for every deterministic bound. Existing literature ranges and sensitivity-only intervals are not confidence intervals.
- Replace the literature-defined long-descent benchmark with a surveyed road profile if a real-route claim is intended, and obtain target heavy-truck communication traces and target hardware evidence.

## Implementation required

- A validated vehicle/platoon simulator connected to `src/safety_core.py`.
- Implement Phase 2C-2 differentiable QP and the matched identical-forward stop-gradient variant; then register production configurations before any evidence-bearing training.
- Centralized robust MPC/dual allocator plus rule-based, independent, and predecessor-following baselines.
- Full supervisor integration around `certified_backup`, including message-age handling and post-certificate-loss minimal-risk mode.
- Real-log plotting and statistical analysis after provenance verification.

## Experiments required

- All registered allocation, barrier, margin, cooperation, OOD, scalability, emergency, Pareto, and latency studies.
- At least ten independent training seeds unless a prospective paired power analysis changes the count.
- Mean/std, paired bootstrap 95% CI, paired standardized effect size, Holm-corrected tests, minimum barriers/reserve, certificate-loss episodes, and the worst emergency trace.
- Required Figs. 1--8 generated only from immutable hash-verified logs.

## Claims that remain prohibited

- Unconditional safety, global recursive feasibility, field/HIL validity, trained-policy superiority, string stability, operational benefit, or real-time deployment readiness.
- Treating a zero failure count as a theorem or the normalized envelope sweep as vehicle evidence.

Current submission status: theory/reference implementation and literature-parameter gate are ready for paper-candidate model simulation; system validation remains `[EXPERIMENT REQUIRED]`, and no result claim has yet been generated.

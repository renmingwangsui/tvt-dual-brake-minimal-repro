# Zhou et al. (IEEE T-ITS 2025) reproduction and heavy-duty adaptation

## Source and scope

J. Zhou, L. Yan, J. Liang, and K. Yang, “Enforcing Cooperative Safety for Reinforcement Learning-Based Mixed-Autonomy Platoon Control,” *IEEE Transactions on Intelligent Transportation Systems*, 2025, DOI: [10.1109/TITS.2025.3627592](https://doi.org/10.1109/TITS.2025.3627592), open preprint [arXiv:2411.10031](https://arxiv.org/abs/2411.10031).

The campaign identifier is `PHASE_2M_EXT_ZHOU`. The native protocol SHA-256 is `eee753d06894fb4bfd74455c5d5f85edd3a7c147668933d2e17f89937d49b470`; the adapted protocol SHA-256 is `6ad915c2b18c512a678f2dd93f3d565a4e750477fa09e97675aa7c481332ab1e`. The native and matched-comparison manifests are immutable under `results/external_baselines/zhou_tits_2025/native_v1/` and `results/external_baselines/zhou_tits_2025/formal_v1/`, respectively.

The heavy-duty comparison is an adapted implementation of Zhou et al., not an exact reproduction of their original mixed-autonomy physical environment.

## Equations and reported parameters implemented

For vehicle `i`, the native reproduction implements

`dot(s_i)=v_(i-1)-v_i`,

`dot(v_i)=u_i` for a connected automated vehicle, and

`dot(v_i)=alpha[V(s_i)-v_i]+beta[v_(i-1)-v_i]` for a human-driven vehicle.

The Full Velocity Difference desired-speed law is the reported piecewise cosine law. The native parameters are `alpha=0.6`, `beta=0.9`, `s_st=5 m`, `s_go=35 m`, equilibrium spacing `20 m`, and equilibrium speed `15 m/s`; the latter fixes `v_max=30 m/s` algebraically. Vehicles 2 and 4 are controlled; vehicles 1, 3, 5, 6, and 7 are human-driven.

The global/local reward follows the paper's velocity-disturbance, efficiency, and time-to-collision terms with weights 0.1 and 0.9, headway threshold 2.5 s, and time-to-collision threshold 4 s. The cooperative barrier is `h_i=s_i-0.3v_i`, the cooperative coefficient is `k=0.4`, and the acceleration range is `[-5,5] m/s^2`. The native controller applies a minimum-change safe correction to the nominal shared-actor action. Conformal prediction uses failure probability 0.01, disjoint training/calibration/test sets, per-sample maximum absolute acceleration error, and quantile index `ceil((n_cal+1)(1-epsilon))` with infinity appended.

The native training uses the reported 450 episodes, 1000 steps per episode, 0.1 s step, actor and critic learning rates `3e-4` with linear decay, batch size 2048, generalized-advantage parameter 0.95, and clipping parameter 0.2. It executed 450,000 environment steps and 439 optimizer updates.

## Unreported implementation choices

The source did not report all neural widths, discount factor, optimizer-epoch interpretation, entropy coefficient, gradient clipping, class-K gain, relaxation weight, communication set, conformal dataset sizes, or sine frequency. These were frozen before evaluation as follows: 64-by-64 hyperbolic-tangent actor and critic layers, discount 0.99, ten optimization epochs, entropy coefficient 0.01, gradient clip 0.5, class-K gain 1.0, relaxation weight 1000, all preceding/following communication, conformal splits 4096/2048/2048, predictor width 64-by-64, and sine angular frequency 0.2 rad/s. Neural/PPO defaults were informed by the authors' public predecessor repository, which is not claimed to be companion code for this paper. Full-batch PPO and the deterministic half-space projection solver are disclosed implementation choices.

The displayed sign in the open preprint's cooperative row conflicts with the derivative of its reduced candidate and with the accompanying physical explanation. The implementation uses the derivative-consistent positive preceding-CAV action coefficient; this choice was recorded before evaluation.

## Native reproduction result

The native gate passed without result-driven retuning. All three cases were collision-free and maintained nonnegative barriers to numerical tolerance.

| Native case | Average CAV headway (s) | AAVE (m/s) | Minimum spacing (m) | Minimum CBF | Collisions |
|---|---:|---:|---:|---:|---:|
| Scenario 1: leader braking/recovery | 1.385 | 1.620 | 2.079 | `6.08e-10` | 0 |
| Scenario 2: HDV 5 acceleration | 1.149 | 1.362 | 4.500 | `8.32e-10` | 0 |
| Sine disturbance | 1.144 | 4.699 | 4.489 | `1.00e-10` | 0 |

The paper's M5 values (2.10 s and 3.83 m/s) were sanity targets only. The sine-case absolute differences, 0.956 s and 0.869 m/s, were inside the pre-registered bands of 1.0 s and 2.0 m/s. No target value was inserted into training or control.

## Heavy-duty action mapping

The native shared actor is transferred without heavy-plant retraining. Heavy trucks map to native identities `[2,4,2]`; unobserved native-state entries remain at equilibrium. The observed heavy gap is mapped as `clip(20 gap/65,5,35)`, and speeds are clipped to `[0,30] m/s` before insertion into the native observation.

For each safe acceleration command, the fixed adapter computes

`F_required=max(0,F_nonbraking-m a_safe)`.

It allocates `min(F_required,F_aux_available)` to auxiliary braking, then allocates the remainder up to the cold friction-command limit. Auxiliary availability is the minimum registered speed/power envelope over the full one-sample speed interval, including tabulated breakpoints. The unchanged heavy-vehicle right-hand side and RK4 step are used, with the physically required projection `v>=0` after each RK4 substep.

The adapter never reads `rho_i`, `rho_H_cert`, upstream reserve, a predictive tube, a certificate supervisor, low-speed certificate logic, physical argmin attribution, a thermal higher-order barrier, or a fade barrier. The accepted certificate stack is evaluated only after the Zhou command is fixed and is labelled “post-hoc common feasibility evaluator.” Static and runtime leakage checks passed.

## Matched comparison

Both methods used the same three-truck plant, order, initial state, road, ambient temperature, leader trajectory, communication case, integration step, evaluation duration, and seed. Six deterministic cases were executed: nominal 10% descent; three exact hot starts at 413.7081614076934 K, 453.7081614076934 K, and 461.7081614076934 K; frozen emergency leader braking during descent; and frozen nominal communication delay.

| Scenario | Method | Coll. | Peak T (K) | Fade violations | Min post-hoc `rho_H` | Friction / auxiliary energy (MJ) | Speed / spacing RMSE | Jerk RMS | Runtime P99 (ms) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Nominal | Accepted | 0 | 458.83 | 0 | -17.99 | 12.81 / 25.26 | 2.56 / 6.74 | 0.252 | 1.198 |
| Nominal | Adapted Zhou | 0 | 570.98 | 157 | -134514.52 | 13.37 / 5.23 | 12.19 / 110.57 | 0.907 | 0.268 |
| Hot 413.708 K | Accepted | 0 | 458.89 | 0 | -5.30 | 7.15 / 14.94 | 2.49 / 5.64 | 0.239 | 1.173 |
| Hot 413.708 K | Adapted Zhou | 0 | 621.03 | 257 | -201097.59 | 13.37 / 5.23 | 10.82 / 54.64 | 1.172 | 0.248 |
| Hot 453.708 K | Accepted | 0 | 458.77 | 0 | -28.54 | 4.91 / 15.91 | 3.71 / 12.23 | 0.146 | 1.522 |
| Hot 453.708 K | Adapted Zhou | 0 | 652.59 | 313 | -242902.64 | 13.37 / 5.23 | 10.82 / 54.64 | 1.172 | 0.232 |
| Hot 461.708 K | Accepted | 0 | 461.71 | 0 | -33.74 | 4.45 / 16.16 | 3.98 / 13.89 | 0.122 | 1.037 |
| Hot 461.708 K | Adapted Zhou | 0 | 658.92 | 324 | -251263.65 | 13.37 / 5.23 | 10.82 / 54.64 | 1.172 | 0.269 |
| Emergency | Accepted | 0 | 458.68 | 0 | 0.034 | 7.85 / 8.74 | 0.89 / 6.06 | 0.397 | 1.363 |
| Emergency | Adapted Zhou | 0 | 570.98 | 88 | -40880.22 | 13.28 / 5.23 | 8.88 / 24.60 | 1.438 | 0.328 |
| Delay | Accepted | 0 | 362.43 | 0 | 0.033 | 0.80 / 3.53 | 1.64 / 1.47 | 0.576 | 1.519 |
| Delay | Adapted Zhou | 0 | 479.24 | 0 | -13675.42 | 5.88 / 5.17 | 8.94 / 24.79 | 0.921 | 0.238 |

All cases completed and neither method collided. The adapted baseline was faster computationally, but had worse tracking, spacing, comfort, and thermal/fade outcomes in these cases. The accepted method also lost the conservative certificate in four descent/hot cases, so the evidence does not support an unconditional feasibility or superiority claim. Exact per-case differences are in `paired_statistics.json`; no confidence interval is reported because evaluation is deterministic with one paired seed and heterogeneous scenarios are not pseudo-replicates.

## Fairness, differences, and limitations

- The heavy-duty plant was not replaced or simplified, and neither method was retuned by scenario.
- Zhou's original mixed-autonomy fleet has two connected automated vehicles and five human-driven vehicles; the matched heavy fleet has three controlled trucks and no human-driven vehicle. Cooperative HDV rows therefore do not apply in the adapted evaluation.
- The transferred actor was not retrained for heavy dynamics. The observation map, identity reuse, scalar-to-dual-brake adapter, and standstill projection can dominate the result.
- The adapted comparison consequently remains `INCONCLUSIVE / ADAPTER-SENSITIVE` for general algorithmic superiority, even though the exact matched outcomes are valid for this frozen adapter.
- Runtime values are descriptive single-platform measurements. Training randomness was not re-sampled, so no inferential claim or confidence interval is made.
- The common collision barrier can be negative without geometric collision; collision count and minimum spacing are therefore reported separately.
- Software-defect corrections v1.0.1 through v1.0.5 were versioned before the successful result manifest. Rejected runs produced no manifest and are excluded from all claims.

The final matched-comparison manifest SHA-256 is `5609dd929e100711de562f0e1b1b9c4de013f9bae12c6f4b97fd41dd3a35c357`. Existing Phase 2M, Phase 2M-FOLLOWUP, Phase 2C-2, and Phase 2C-3 evidence was not modified.

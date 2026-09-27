# Final revision summary

## Evidence lock

- Phase 2M, follow-up, and formal-learning integrity tests passed before manuscript editing.
- Frozen configurations, raw logs, manifests, claims, physical parameters, and learning results were not modified.
- Final figures/tables are generated into `generated/final/` from immutable analysis artifacts, with their own SHA-256 manifest.

## Title candidates

1. **Predictive Feasibility Certification for Collision–Thermal Conflict and Dual-Brake Allocation in Heavy-Duty Platoons** — selected; closest to the strongest theory and model evidence.
2. Collision–Thermal Feasibility Certificates and Dual-Brake Control for Heterogeneous Heavy-Duty Platoons.
3. Distributed Predictive Bottleneck Certification for Thermally Constrained Heavy-Duty Platoons on Long Descents.

The selected title is not DiffQP-centric.

## Scientific changes in presentation

- Reordered the contributions around the physics, instantaneous certificate, predictive/distributed certificate, dual-brake allocation, and conditional KKT extension.
- Rewrote the Abstract, Introduction, validation protocol, Results/Discussion, limitations, and Conclusion from verified evidence only.
- Corrected the canonical QP everywhere to the accepted two-variable `(u_f,u_a)` formulation and removed the nonexistent jerk slack/rows.
- Added the structural LICQ explanation: four active inequality gradients cannot be independent in a two-dimensional decision space.
- Integrated M1–M10, targeted follow-up, Phase 2C-2 implementation validation, and the ten-paired-seed Phase 2C-3 negative result.
- Removed DiffQP learning-improvement, universal robustness, string-stability, global differentiability, global recursive-feasibility, and low-speed-continuity claims.
- Added source-level citations for every physical calibration family used in the manuscript.

## Administrative state

The paper remains anonymized. Author names, affiliations, funding/acknowledgment text, conflict-of-interest declarations, data/code availability wording, and the target submission system's current metadata must be supplied or confirmed by the authors before submission.

## Final pre-submission minor cleanup

- Preserved the accepted title, two-variable QP, all equations, all frozen numbers, the ten-paired-seed result, and the negative DiffQP interpretation.
- Expanded `P99` and removed standalone `QP`, `LICQ`, and `KKT` abbreviations from the Abstract without changing its scientific meaning; necessary method names remain. Final Abstract length is 220 words.
- Set the Index Terms exactly to the required six-term method/application metadata string.
- Retained the anonymous administrative placeholder because no verified author metadata exists in the repository.
- Regenerated `main.bbl`, rebuilt and visually audited the 13-page PDF, and rebuilt the source archive with `main.bbl` and a verified 40-file SHA-256 manifest.
- Both pre- and post-revision complete regression suites pass 33/33.

## Final theory and reproducibility micro-revision

- Clarified `\rho_{\rm air}`, the driveline/residual-force terms, estimated grade, gear dwell, message age, auxiliary-envelope derivative, and the non-safety role of `\epsilon_g`.
- Stated the barrier-function class and the strong-convexity chain `W_i\succeq0`, `\epsilon>0`, and `H_{{\rm QP},i}=W_i+2\epsilon I\succ0`; connected the generic KKT notation directly to the accepted two-command projection.
- Completed the concise conditional proof for Proposition 6 without strengthening the sampled-data guarantee.
- Made the registered predictive backup implementation reproducible as a componentwise outward-rounded interval enclosure; no midpoint backup is certified or used, the provisional QP action remains the first predictive input, and final-action changes remain subject to reverification.
- Refined the equation-to-loop trace and added the Discussion opening sentence that centers feasibility certification and physical conflict attribution.
- Preserved every frozen result and limitation, including the M1 feasible-count result, all M2 certificate losses, targeted-only predictive support, M6/M7/M8/M9/M10 limitations, and the formal DiffQP zero-valid-gradient result.
- No scientific experiment was rerun, no parameter was retuned, and no generated evidence table or figure changed. The final 14-page PDF, 33/33 regression suite, 40-file manifest, and isolated archive build all pass.

## Final notation and Abstract micro-patch

- Defined `\Delta_T` as the registered low-speed thermal safety buffer without adding or changing a numerical value.
- Defined `\beta` and `\lambda_I` as the entropy and registered intervention-penalty weights without changing the actor objective or learning protocol.
- Defined `C_{\min}`, `\bar e_{\rm int}`, and the local registered-bound meaning of overbars in the sampled-data/low-speed error bounds while preserving established physical envelope notation.
- Removed the `DiffQP` and `StopGradient` method labels from the Abstract only; the body terminology and the full negative-result interpretation remain intact. The final Abstract is 222 words.
- No scientific result, frozen evidence, experiment, parameter, figure, generated result table, title, contribution hierarchy, or bibliography content changed. The 14-page build, 33/33 regression, 40-file manifest, isolated archive build, and Desktop hash synchronization pass.

## External T-ITS baseline manuscript integration

- Verified and integrated the frozen Zhou et al. campaign manifest `5609dd929e100711de562f0e1b1b9c4de013f9bae12c6f4b97fd41dd3a35c357` without rerunning, retraining, retuning, or modifying either method.
- Added a native reimplementation sanity-check statement and a six-case heavy-duty transfer protocol with post-hoc-only certificate semantics.
- Added one automatically generated two-column comparison table sourced directly from immutable JSON. EXT-1 was retained outside the main paper to preserve layout and readable sizing.
- Integrated collision, thermal, fade, post-hoc reserve, tracking, energy, runtime, and adapter-sensitivity outcomes without a general superiority, safety-failure, or universal-feasibility claim.
- Preserved the Abstract and all existing theory, negative results, limitations, M1--M10 evidence, Phase 2M-FOLLOWUP, Phase 2C-2, and formal learning evidence.
- The final 15-page PDF, visual audit, 33/33 regression suite, 41-file source manifest, and isolated source-archive build pass. The source ZIP SHA-256 is `5fa756fbb410b81c282ec92c4f3b0cb97a837d23a7d173a5ce23640c6ce3dac2`.

## Final external-baseline wording micro-patch

- Replaced the generic native-reimplementation "passed" wording with the supported claim that collision-free behavior and nonnegative cooperative-CBF values were reproduced in the three source-style disturbance families; numerical reproduction of every published value remains explicitly unclaimed.
- Narrowed the heavy-duty transfer interpretation to the frozen scalar-to-dual-brake adapter and no-heavy-duty-retraining conditions, while expressly excluding a general Zhou-method safety judgment or native-domain algorithm ranking.
- No scientific experiment was rerun, no training or retuning was performed, no frozen result or Table XV value changed, and no theorem, equation, Abstract, or Conclusion changed.
- Both complete regression gates pass 33/33. The final 15-page PDF, all-page visual audit, 41-file SHA-256 source manifest, and isolated source-archive build pass. The rebuilt source ZIP SHA-256 is `c8792e76985dd662df9bcfaff24ca29be835a95e1a612aaf7ee8b040b56fd707`.

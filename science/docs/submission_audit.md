# Submission audit

## Scientific integrity

- Frozen physical configuration SHA-256: `d10d13eb82490bcfb2151f7db2114e38d8896810255b3138d17540cddabd5104`.
- No physical parameter, frozen M1--M10 result, formal-learning record, or Phase 2M follow-up log was intentionally changed by this revision. New analyses write to a separate `results/submission_revision/` tree.
- The boundary result is one-step command-domain evidence. The same-domain ablation does not establish a safety-performance gain. Positive predictive stored-sample checks are contemporaneous consistency checks, not exhaustive future-rollout validation.
- Theoretical Proposition 2 assumes sound interval and reserve enclosure. Source inspection found no directed/outward endpoint rounding, no complete gear/low-speed branch hull, and no proof that the executed width-offset reserve callback is a global lower bound. The current floating-point predictor is diagnostic, not a formally assured finite-horizon certificate. Details: `docs/interval_arithmetic_soundness.md` and `docs/predictive_interval_implementation_audit.md`. This is an unresolved scientific implementation-assurance issue, not a change to the theorem or frozen numbers.
- The 300-point stratified design is not a physical probability distribution. Thermal capacity and auxiliary lag stay fixed because their registered uncertainty intervals are degenerate.

## Manuscript / package

- IEEEtran LaTeX source and compiled PDF are under `manuscript/`. Under Branch B the title distinguishes exact instantaneous feasibility certification from numerical predictive monitoring; Proposition 2 remains conditional.
- Main figures are editable TikZ or dependency-free SVG/PDF outputs; figure generators and input CSV/JSON are included.
- Anonymous author metadata remains a submission blocker until verified author, affiliation, funding, and acknowledgment details are provided.
- No real-vehicle or hardware-in-the-loop validation is claimed. Finite-horizon certification is conditional on sound implementation as well as the declared model and uncertainty bounds.

## Reference coverage and positioning

The 28 cited items were reviewed by topic, with no numerical target for bibliography length. Heavy-duty platooning/CACC: adequately covered (`naus2010string`, `ploeg2011cacc`, `devika2022fade`). Heavy braking/fade: adequately covered (`devika2022fade`, `nhtsa2010scam`, `ntsb2002simulation`, `yan2018thermal`). Retarder/multi-actuator braking: adequately covered (`li2025retarder`, `li2026retarder`, `donkers2020powertrain`). CBF/HOCBF: adequately covered (`ames2017cbf`, `xiao2022hocbf`). Sampled-data/robust CBF: thin (`emam2022robust`, `hu2025event` address adjacent uncertainty but not machine-sound interval evaluation). Predictive/reachable-set safety: adequately covered for positioning (`ohnemus2026dpcbf`, `guo2025uncertainty`). Delayed connected-vehicle communication: adequately covered for the model domain (`hu2021delay`, `gao2016dsrc`), though not a proof of the recursive algorithm. Exact safety-filter feasibility: adequately covered (`sah2026exact`). Recent T-ITS connected-vehicle safety: adequately covered (`li2025connectedcbf`, `zhou2026cooperative`). Differentiable optimization/safe learning: adequately covered (`amos2017optnet`, `ma2022differentiable`, `xiao2023barriernet`, `yang2023diffcbf`, `yu2022mappo`).

No citation was added merely to increase the count: the sampled-data interval-implementation gap is a code-assurance gap, not something a new citation can resolve. The manuscript distinguishes the physics-specific geometry from generic CBF filtering, predictive/distributed CBF, exact feasibility, differentiable QP, and MAPPO. Its closest-work discussion retains Zhou, Ohnemus, Sah and Keshavan, Guo, Devika, and Li.

## Verification record

- Abstract: 206 words by the local word-token audit; no unnecessary QP/LICQ/KKT/DiffQP abbreviation in the Abstract. The six Index Terms retain the specified IEEE T-ITS ordering.
- `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex`: PASS; 15 pages after first-use acronym expansion and traceability clarification; no undefined citation/reference or overfull-box diagnostic in the final log.
- `python scripts/run_regression_suite.py`: PASS, 33/33 scripts (12 reference, 5 Phase 2A, 5 Phase 2B, 5 Phase 2C-1, 2 Phase 2C-2, 1 Phase 2C-3, 1 Phase 2M, 1 Phase 2M-LIT, 1 Phase 2M-FOLLOWUP).
- The literature-calibration regression invokes a legacy finalizer that rewrites repository-source hashes in the frozen config. The test now restores the byte-identical pre-test config at process exit; the final config SHA-256 remains `d10d13eb82490bcfb2151f7db2114e38d8896810255b3138d17540cddabd5104`.
- All 15 rendered pages were inspected. Table II and Table III are first referenced on page 2 before appearing on page 3; all five figures and nine tables have prior meaningful references. There is no clipping, overlap, malformed math, or broken caption. Six final references occupy a separate, mostly blank page 15 after the acronym expansions; IEEE template geometry was not changed. Scientific results in previous campaigns were not regenerated.

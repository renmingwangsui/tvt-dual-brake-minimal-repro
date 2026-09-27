# Submission-oriented revision changelog

- Repositioned the manuscript around collision--thermal--actuation command geometry, exact instantaneous feasibility, one-sided predictive certification, age-aware recursive bottlenecks, and dual-brake supervision. The Introduction now states three contributions.
- Clarified the low-speed proof: for `0 < v <= v_epsilon`, the zero-order-hold thermal row is absorbed into the friction upper bound; at `v = 0`, its zero command coefficient leaves the state-only condition `g_T0 >= 0`.
- Added a 1,275-state initial-command feasibility frontier without changing the calibrated plant or frozen M1--M10 evidence. It identifies 43 dual-only feasible grid points under the specified 15 m/s, 10% downgrade, temperature, and gap design.
- Added a same-plant instantaneous-only supervisor ablation. The three tested pairs show no reduction in hard-certificate-loss samples; the manuscript states this explicitly.
- Added fixed-delay fleet-awareness comparisons, a 300-point boundary-state parameter design, and normalizer attribution analysis. All are conditional model-based analyses, not physical probability or field evidence.
- Reframed predictive validation as one-sided: certified-positive same-sample consistency plus interval-enclosure regression, and explicit negative-side warning classifications. Future executed traces are not treated as a theorem test when they depart the certified backup.
- Created a new vector framework figure and vector frontier, predictive, and communication figures. Removed the uninformative learning diagnostic and large runtime/robustness tables from the main narrative; their source artifacts remain in the repository.
- Kept the frozen model configuration and prior paper-candidate/follow-up results unchanged. New CSV/JSON outputs are under `results/submission_revision/`, with generators under `scripts/`.
- Retained anonymous metadata; author identities, affiliations, funding, and acknowledgments require actual author input before submission.
- Updated the manuscript-integrity test to look for required technical wording anywhere in the manuscript rather than only in the Abstract, without relaxing its Abstract abbreviation or Index Terms checks. Made the literature-calibration regression restore the frozen config after its legacy finalizer runs.

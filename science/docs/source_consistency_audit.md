# Phase 2M-LIT-3 source-consistency audit

## Outcome

**PASS WITH EXPLICIT CROSS-SOURCE MODEL-MAPPING LIMITATIONS.** All 19 retained physical requirements have a source or reproducible derivation, units are consistent, and no retained record is `ASSUMED` or unresolved. The `Reference_Truck` is a literature-composed heavy-duty model, not an experimentally identified single vehicle.

## Consistency matrix

| Check | Status | Reason |
|---|---|---|
| Mass vs friction capacity | PASS for model-equivalent bound | same NHTSA vehicle/mass/stopping family |
| Vehicle size vs class | PASS with scope limitation | compatible AASHTO design vehicle, not the same measured truck |
| Thermal capacity/cooling vs brake type | PASS for reduced model | same NHTSA S-cam family and documented ten-wheel-end aggregation |
| Heat conversion | PASS as model mapping | `eta=1` only at the matched mechanical-power/aggregate-thermal boundary |
| Ambient | PASS as nominal benchmark | Yan–Xu 25 °C baseline; not a population distribution |
| Fade and `T_crit` | PASS as representative mapping | NTSB published heavy-truck S-cam curve; conservative monotone inferior-lining normalization; not material-identical |
| Thermal residual | PASS as deterministic discrepancy envelope | 40-point NHTSA/Yan–Xu comparison grid; not an experimental confidence interval |
| Auxiliary map vs drivetrain | PASS as explicit cross-source mapping | measured Li et al. retarder curve plus documented heavy-duty final drive, wheel radius, and efficiency |
| Auxiliary lag | PASS as derived dynamic mapping | published 0.25-s first-order approximation; total settling time is not equated silently to `tau_a` |
| Command-slew partition | PASS | unsupported independent physical rows removed; both realized-force lags retained |
| Predecessor vehicle class | PASS with derivation caveat | heavy-commercial evidence; jerk remains derived |
| Communication model | LIMITED but traceable | bounded observation/design envelope, not a fitted universal IID distribution |
| Uncertainty semantics | PASS for readiness | source ranges, digitization bounds, model discrepancy, and sensitivity ranges remain distinctly labeled |

## Audit boundary

The source chain is dimensionally and physically consistent for a representative heavy-duty paper-candidate model. It does not support claims of a material-matched brake lining, a vehicle-specific retarder installation, a statistical ambient/residual confidence interval, or field validation. Those limitations remain mandatory in any later results discussion.

The strict gate returns `READY_FOR_PAPER_CANDIDATE_MODEL_SIMULATION`. No final M1–M10 run was performed here.

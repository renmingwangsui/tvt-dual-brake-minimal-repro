# Phase 2M Physical Data Required

Every item below needs a source appropriate to the selected heavy-vehicle class, an exact source location, operating range, units, uncertainty, and accepted calibration evidence before `LITERATURE_CALIBRATED` execution.

| Dependency | Required evidence |
|---|---|
| Vehicle mass `m` and length `L` | vehicle specification and loading distribution/range |
| Rolling resistance `c_r` and drag area `C_d A` | test standard or measured/coast-down fit with conditions |
| Friction brake force limit `u_f,max` | axle/tire/brake-system envelope including road friction assumptions |
| Friction lag `tau_f` | transient brake-system measurements; ZOH command slew is not a separate hard physical parameter |
| Auxiliary lag `tau_a` | retarder transient measurements or an explicit first-order model mapping |
| Auxiliary envelope `r_bar(v,q,P)` | gear-, speed-, thermal-, and power-dependent manufacturer/test map |
| Thermal capacity `C` | effective drum/disc model identification and modeled mass boundary |
| Cooling coefficient `k` | speed/ambient-dependent thermal identification or justified model |
| Heat fraction `eta` | energy-partition measurement/model with uncertainty |
| Critical temperature `T_crit` | material/system safety criterion linked to the fade model |
| Fade curve `phi(T)` | dynamometer or road-test curve with hysteresis/conditioning notes |
| Ambient temperature `T_a` | scenario distribution or measured trace |
| Residual heat `q_w` | identified non-friction heat source model |
| Road grade `theta(p)` | measured or authoritative road profile with spatial resolution |
| Predecessor acceleration/jerk bounds | data-derived maneuver envelope and sampling method |
| Communication delay `d_comm` | target V2V stack/network measurements and tail statistics |
| Packet/burst loss `p_loss` | target channel measurements, correlation, and burst model fit |

The DEBUG ±10% perturbations and independent uniform joint samples are software stimuli, not calibrated uncertainty distributions. They must not be reused as physical robustness evidence.

## Phase 2M-LIT-3 audit status — 18 September 2026

The final literature audit retains 19 required dependencies: seven `DIRECT`, ten `DERIVED`, and two `DIGITIZED`. No retained dependency is `ASSUMED` or unresolved. The two command-slew items were removed from the physical hard-constraint set because ZOH commands are constant between samples and the realized forces are already governed by first-order lags; no independent measured command-rate limit was found.

See `docs/model_parameter_provenance.md`, `docs/source_consistency_audit.md`, and `data/model_parameters/parameter_provenance.yaml`. The strict gate returns `READY_FOR_PAPER_CANDIDATE_MODEL_SIMULATION` and the frozen `literature_calibrated_v1.yaml` exists. No final M1--M10 run has been performed.

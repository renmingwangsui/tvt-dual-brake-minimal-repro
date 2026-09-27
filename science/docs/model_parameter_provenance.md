# Phase 2M-LIT-3 model-parameter provenance

## Decision

The final literature gate is **READY_FOR_PAPER_CANDIDATE_MODEL_SIMULATION**. The machine-readable authority is `data/model_parameters/parameter_provenance.yaml`: 19 retained requirements, comprising seven `DIRECT`, ten `DERIVED`, and two `DIGITIZED` records, with zero `ASSUMED` and zero unresolved records. This decision authorizes a later paper-candidate simulation; it is not an experiment result.

The two former command-slew requirements were retired. Under zero-order hold each command is constant on the open sampling interval and may change at a sample; realized friction and auxiliary forces remain continuous through their retained first-order actuator lags. No measured independent command-rate limit is claimed.

## Final closure of the former nine gaps

| Item | Decision | Evidence and scope |
|---|---|---|
| Friction command slew | removed from physical hard set | no defensible independent physical limit; `tau_f` retained |
| Auxiliary lag `tau_a` | 0.25 s, `DERIVED` | Li et al. (2026) published simplified first-order hydraulic-retarder link; 1.0 s is only the stated four-time-constant 2% convention |
| Auxiliary command slew | removed from physical hard set | no independent measurement; `tau_a` retained |
| Auxiliary envelope | `DIGITIZED` | Li et al. Fig. 5, 300-kPa measured torque-speed curve; 300 kW continuous and 450 kW peak; mapped by `omega_ds=v*i0/r_w` and `F=T*i0*eta_fd/r_w` with `i0=5.7`, `r_w=0.492 m`, `eta_fd=0.97` |
| Heat conversion `eta` | 1.0, `DERIVED MODEL-MAPPING` | NHTSA Appendix C injects brake mechanical power directly; valid only for the same aggregate brake-system boundary |
| Critical temperature | 463.708161 K, `DERIVED` | pre-registered first crossing of conservative `phi(T)=0.80`; not a failure temperature |
| Fade curve | `DIGITIZED_REPRESENTATIVE_HEAVY_TRUCK_S_CAM_FADE` | NTSB HWY-01-M-H-25, PDF p. 7 / printed p. 4, Fig. 4, inferior-lining curve; representative, not material-identical |
| Ambient | 298.15 K, `DIRECT` | Yan and Xu (2018), Table 1 baseline; not a statistical interval |
| Residual heat | 0 W nominal with ±92,551.573166 W bound, `DERIVED` | maximum absolute NHTSA-versus-Yan–Xu cooling discrepancy on the registered 50–70 km/h and 25 °C-to-`T_crit` grid; not an experimental confidence interval |

## Reproducible artifacts

- `scripts/calibration/finalize_phase2m_lit3.py` reproduces both digitizations, all model mappings, the discrepancy grid, registry closure, and frozen config.
- `data/model_parameters/source_points/phase2m_lit3_digitization.json` stores raw pixel coordinates, axis calibration, source locations, derived SI points, and image hashes.
- `data/model_parameters/derived/thermal_model_discrepancy.json` stores the comparison domain, equations, 40 residual samples, maximum absolute residual, script, and reproducible source record.
- `data/model_parameters/derived/phase2m_lit3_parameters.json` records the final mapping decisions.
- `configs/model_simulation/literature_calibrated_v1.yaml` freezes registry/source hashes, per-parameter hashes, repository file hashes, and the hard-constraint partition.

The Reference_Truck remains a documented cross-source heavy-duty composite, not an experimentally identified individual truck. No final M1–M10 execution or paper-eligible result was produced in this phase.

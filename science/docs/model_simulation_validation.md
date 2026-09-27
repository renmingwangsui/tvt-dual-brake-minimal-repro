# Phase 2M Validation Protocol

## Required checks

1. Compile the controller, pipeline, analysis, and scripts.
2. Run `tests/test_model_simulation_pipeline.py` using a reduced deterministic DEBUG configuration.
3. Verify that all M1--M10 record files exist and contain immutable DEBUG provenance fields.
4. Verify zero-speed/hot-brake `g_T0` hard-infeasibility propagation and finite ZOH diagnostics.
5. Verify runtime component keys and nonnegative timing values.
6. Verify the publication gate rejects DEBUG configuration, registry, manifest, and default figure generation.
7. Generate explicitly allowed DEBUG figures and verify their manifest and visible watermark.
8. Run `scripts/run_regression_suite.py`, which fails on the first regression.

## Acceptance logic

Passing this protocol establishes executable software plumbing, deterministic record production, analysis completeness, and fail-closed provenance behavior. It does not validate physical parameters, experimental realism, vehicle calibration, generalization, formal string stability, real-time guarantees, or manuscript claims.

The full DEBUG run must report all ten experiments, preserve `paper_eligible=false`, and generate no paper-eligible result. Literature-calibrated execution remains unavailable until `docs/model_simulation_data_required.md` is resolved and independently reviewed.

## Evidence-state distinctions

| State | Meaning in this repository | Current Phase 2M status |
|---|---|---|
| Software-verified | Code paths, schemas, deterministic execution and gates pass automated tests | yes, for DEBUG M1--M10 |
| Mathematically verified | Implemented identities, signs, units and interval/certificate semantics pass the existing analytical and randomized tests | yes, within the documented model assumptions |
| Literature-calibrated | Every run dependency is tied to an accepted source and calibrated operating range | no |
| Physically validated | Outputs are compared against appropriate vehicle/network measurements with an accepted validation protocol | no |
| Paper-eligible | Frozen configuration, complete provenance, validated result manifest and all publication checks pass | no |

These states are not interchangeable: software or mathematical verification cannot substitute for literature calibration or physical validation.

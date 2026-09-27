# Phase 2M Model-Simulation Design

## Scope and status

Phase 2M exercises the existing nonlinear plant, hard safety QP, predictive interval certificate, distributed bottleneck protocol, and supervisor without training or DiffQP. It does not alter MAPPO semantics. The currently executable suite is `DEBUG_SYNTHETIC`; its outputs validate software paths only and are not physical evidence or paper results.

Two modes are defined:

- `DEBUG_SYNTHETIC`: executable with repository debug values, deterministically seeded, always `paper_eligible=false`.
- `LITERATURE_CALIBRATED`: specified as a fail-closed template. It remains disabled until every physical parameter has an accepted source, location, uncertainty bound, and calibration record and the frozen configuration/result hashes pass provenance validation.

## Experiments

| ID | Purpose | Principal outputs | Interpretation limit |
|---|---|---|---|
| M1 | Feasibility geometry across speed, temperature, grade, mass, auxiliary availability | command widths, demand, friction-only and dual-actuator margins, rho | Debug parameter geometry only |
| M2 | Long descent with friction-only, friction-first, auxiliary-first, and nominal dual-brake/hard-QP allocation strategies | temperature, actions, reserves, barriers | The same hard safety QP remains active for every strategy |
| M3 | Hot-brake initial-condition sweep | temperature, fade, reserve, supervisor state | No physical temperature claim |
| M4 | Predictive warning sequence | instantaneous rho, horizon rho, trigger and boundary times | Negative horizon reserve is a conservative warning, not proof of infeasibility |
| M5 | Horizon sensitivity | tube width, horizon reserve, warning count, runtime | Numerical sensitivity only |
| M6 | Robustness | independent perturbations and seeded joint samples | DEBUG uniform distributions are not empirical uncertainty models |
| M7 | Communication | centralized/local bottlenecks, identity/component agreement, trigger rate | Synthetic channel models |
| M8 | Low-speed ZOH | `g_T0`, hard feasibility, ZOH versus continuous plant update | Covers zero, near-zero, epsilon, above-epsilon and hot/cold cases |
| M9 | Platoon scaling | empirical G2/Ginf ratios and peak errors | Numerical study only; no string-stability theorem |
| M10 | Runtime scaling | raw component timing samples and percentiles | Machine/run dependent; no hardware guarantee |

## Execution and artifacts

Run `python scripts/run_model_simulation_suite.py`. The runner writes one JSONL file per experiment, `manifest.json`, and `analysis_summary.json` under `artifacts/model_based/debug`. All records carry the run ID, experiment ID, configuration hash, provenance hash, seed, provenance class, and paper-eligibility flag.

Diagnostic SVGs require the explicit command `python scripts/generate_model_simulation_figures.py --allow-debug`. Every DEBUG SVG is visibly watermarked. Omitting `--allow-debug` rejects the ineligible input with a nonzero exit status.

## Scientific boundaries

The mathematical equations and controller algorithms are those already validated in Phases 2A/2B/2C-1. Phase 2M adds orchestration, logging, timing observation, analysis, and diagnostic rendering. Runtime timing does not enter control decisions. No manuscript result table is changed. Literature calibration and paper-result generation are separate future gates.

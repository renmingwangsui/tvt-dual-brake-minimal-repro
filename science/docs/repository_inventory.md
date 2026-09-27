# Repository readiness inventory

Audit date: 17 September 2026. Scope includes the complete project tree and the separately delivered `Desktop/2` package. The delivery folder contains only the packaged manuscript/PDF and audit documents; it contains no additional physical data, logs, paper-eligible checkpoints, or production configuration.

| component | file | implemented? | tested? | production-ready? | paper equation | missing dependency |
|---|---|---:|---:|---:|---|---|
| IEEE manuscript source | `manuscript/main.tex`, `manuscript/sections/`, `manuscript/references.bib` | yes | compile/visual QA | n/a | all | author metadata and system evidence |
| Shared continuous dynamics primitives | `src/simulator/truck_dynamics.py`, `src/simulator/thermal.py`, `src/safety_core.py` | Phase 2A implementation | analytical/sign/regression tests | no | `eq:model`, `eq:jerk`, `eq:zohthermal` | calibrated production parameters and physical validation |
| Instantaneous certificate $\rho$ | `src/safety/conflict_reserve.py` | yes | yes | no | `eq:reserve`, `eq:rho`, `eq:instantfeas` | physical limits and uncertainty calibration |
| Low-speed thermal row and $g_{T0}$ gate | `src/safety_core.py`; `src/safety/conflict_reserve.py` | reference implementation | zero/near-zero/threshold hot/cold tests | no | `eq:zohthermal`, `eq:gT0`, `eq:rho` | calibrated thermal data and simulator integration |
| Predictive reserve | `src/safety/predictive_reserve.py` | reference implementation | yes | no | `eq:predlower`, `eq:rhocert` | calibrated uncertainty and production simulator callback |
| Interval reachability | `src/safety/reachable_set.py` | reference implementation | yes | no | `eq:tube` | identified interval bounds and integration-error bound |
| Set-valued backup extension | `src/safety/reachable_set.py` | affine+saturation reference | yes | no | `eq:tube` | production backup policy and sound extension proof/test |
| Distributed bottleneck | `src/safety/fleet_reserve.py`, `src/network/`, `src/envs/heavy_platoon_env.py` | Phase 2B event-driven integration | zero-delay equivalence and delayed/lost non-equivalence | no | `eq:fleetreserve`, `eq:minconsensus` | validated delay/loss data |
| Two-pass controller order | `src/safety/control_loop.py`, `src/controllers/integrated_safety_controller.py` | Phase 2B closed-loop integration | exact order, changed-action re-verification, residual gate | no | Sec. V-F | calibrated uncertainty/margins and timing validation |
| Four-mode supervisor | `src/safety/supervisor.py` | decision logic | unit tested indirectly | no | `eq:backupdomain` | full controller integration and calibrated thresholds |
| Actor parameterization | `src/actor_parameterization.py`, `src/rl/mappo/actor.py` | scalar reference plus shared NumPy DEBUG actor | exact density/shape/bounds/leakage tests | no | Sec. V-C | production framework selection and physical training evidence |
| Predictive learning losses | `src/training_objective.py` | scalar reference | yes | no | `eq:actorloss` | autodiff tensor implementation and trainer |
| Differentiable QP/KKT | manuscript equation and tests only | no production layer | algebra only | no | `eq:kktdiff` | QP solver, implicit backward layer, condition monitoring |
| MAPPO actor/critic/buffer/GAE/trainer | `src/rl/mappo/` | Phase 2C-1 NumPy DEBUG implementation | 5 dedicated scripts plus full regression | no | Sec. V-C | production framework/configuration, calibration and evidence |
| Production truck simulator | `src/simulator/` | Phase 2A continuous architecture | synthetic-debug unit/convergence tests | no | `eq:model` | validated physical manifest and physical validation |
| Numerical integrators/convergence | `src/simulator/integrators.py` | RK4, adaptive reference, ZOH | dt/dt2/dt4 and analytical tests | no | `eq:model` | production integration-error certificate |
| Platoon environment | `src/envs/heavy_platoon_env.py`, `observations.py`, `leader_profile.py`, `scenario.py` | heterogeneous synchronous Phase 2B environment | N=1/3/5/10/20/40 and six DEBUG scenarios | no | `eq:chi` | physical calibration and optional future Gym wrapper |
| Road-profile module | `src/envs/road_profile.py` | constant/piecewise/interpolated | boundary and sign tests | no | `eq:model` | surveyed road files and provenance |
| V2V channel simulator | `src/network/` | event buffer, delay/loss/burst/stale/order/consistency | Phase 2B network tests | no | `eq:minconsensus` | validated trace/distribution and sensing calibration |
| Baselines | absent (`baselines/`) | no | no | no | experiment protocol | all registered matched baselines |
| Structured immutable logger | schemas, in-memory `Phase2BLogRecord`, `experiments/model_based/pipeline.py` | deterministic JSONL DEBUG writer and run manifest | M1--M10 fields, hashes and provenance flags | no | experiment protocol | production append protection, Parquet/HDF5 and calibrated manifest |
| Result provenance gate | `experiments/check_result_provenance.py` | partial | fail-closed run | no | experiment protocol | valid result manifest and real logs |
| Experiment manager | gate only: `experiments/run_registered_suite.py` | no runner | fail-closed run | no | experiment protocol | simulator/trainer/config system |
| Registered experiment intent | `experiments/configs/registered_suite.yaml`, `experiments/experiment_plan.yaml` | placeholders only | syntax read | no | Sec. VII | typed schemas and resolved production values |
| Statistics pipeline | `analysis/model_based/analyze.py` | DEBUG M1--M10 summaries and runtime percentiles | reduced and full DEBUG suites | no | Sec. VII | literature-calibrated logs, inferential protocol and review |
| Figure pipeline | `analysis/model_based/figures.py`, `scripts/generate_model_simulation_figures.py` | eight watermarked DEBUG SVG diagnostics with fail-closed default | explicit DEBUG override plus rejection path | no | future figures | paper-eligible real logs and validated production renderer |
| Table pipeline | absent | no | no | no | Table VII | verified metrics and generated-table scripts |
| Checkpoints | `artifacts/debug/phase2c1_mappo_smoke.pkl` | DEBUG smoke only | save/load and RNG equivalence | no | n/a | paper-eligible trained models, immutable run manifest and hashes |
| Claim-bearing logs | DEBUG logs only under `artifacts/model_based/debug/` | software-validation records | provenance and publication gates block paper use | no | Sec. VIII | literature-calibrated immutable logs |
| Theory/reference, simulator, platoon, MAPPO and model-simulation regression tests | `tests/` | yes | 28 scripts pass after Phase 2M | no physical system evidence | multiple | production physical/network/system validation |
| Strict readiness validator | `src/data_validation.py`, `scripts/check_simulation_readiness.py` | yes | executed in this audit | audit-ready | n/a | production manifest and all required entries |

Overall repository status is **not production-ready**. Phase 2B supplies an executable heterogeneous closed loop, event-driven V2V channel and integrated audited safety stack; Phase 2C-1 adds a complete DEBUG MAPPO forward-training/checkpoint/evaluation pipeline; Phase 2M adds non-learning M1--M10 DEBUG simulation, logging, analysis and watermarked diagnostic figures. All use explicitly synthetic debug data. These artifacts do not constitute a calibrated controller, convergence result, physical/network validation, or paper-eligible system experiment.

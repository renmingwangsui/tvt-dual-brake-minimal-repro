# Final validation report — 17 September 2026

## Scope

This report covers the interval-certificate, causal-control, distributed-bottleneck, set-valued-backup, backup-domain and normalization revision. It does not claim system-level performance.

## Automated validation

- All 29 current Python test/audit scripts passed in the primary workspace: 12 pre-existing theory/reference scripts, 5 Phase 2A simulator scripts, 5 Phase 2B platoon/network/safety-integration scripts, 5 Phase 2C-1 MAPPO scripts, 1 Phase 2M model-simulation integration script and 1 Phase 2M-LIT calibration/provenance script. The 12 legacy scripts also retain the prior independent NTFS case-sensitive pass.
- The traceability audit enforces the unique, exact-case canonical path `docs/unverified_claims.md`, including on case-insensitive development hosts.
- All Python files under `src/`, `tests/`, `experiments/`, `analysis/` and `scripts/` passed byte-code compilation after Phase 2M integration.
- Randomized coverage includes 50,000 equation identities, 40,000 constraint-algebra checks, 10,000 interval-box LP comparisons, 5,000 tube-containment samples and 5,000 one-sided interval-lower-bound samples.
- Regression coverage includes exact causal callback order, conditional final-command re-verification, nested closed-loop boxes, distributed delay/loss/stale/multihop aggregation, normalization sign/magnitude/attribution behavior, the complete backup-domain guard, and zero/near-zero/threshold hot/cold $g_{T0}$ cases.
- Phase 2A coverage includes the shared continuous plant, road interpolation, both actuator lags, thermal analytical cases, fade-domain handling, auxiliary/gear interfaces, exact ZOH command boundaries, RK4, adaptive-reference comparison and three-scenario timestep convergence. See `docs/simulator_validation.md`.
- Phase 2B coverage includes heterogeneous `N=1/3/5/10/20/40` construction, synchronous integration, separated observations, causal leader profiles, event-driven V2V delay/loss/burst/stale/order/consistency behavior, predecessor interval expansion, communication-based bottleneck propagation, integrated hard QP/certificates/set-valued backup/four-mode supervisor, changed-action re-verification, final residual validation, low-speed closed-loop `g_T0`, six short `N=3` DEBUG episodes and scalability execution through `N=40`. See `docs/safety_integration_validation.md`.
- Phase 2C-1 coverage includes the shared local actor, fixed-`N` centralized critic, exact affine tanh-squashed density, three-action semantics, full Phase 2B safety-forward rollout, terminal/truncation-aware GAE, clipped PPO, finite/clipped gradients, checkpoint/RNG restoration, normalization-frozen deterministic evaluation and an `N=3` DEBUG smoke/sanity job. See `docs/mappo_validation.md`.
- Phase 2M coverage includes all M1--M10 model-based DEBUG experiments, per-record configuration/provenance hashes, the low-speed hot/cold ZOH branch, deterministic robustness/channel cases, empirical platoon-response metrics, raw runtime samples, summary analysis, eight visibly watermarked SVG diagnostics and a fail-closed publication gate. See `docs/model_simulation_validation.md`.
- Phase 2M-LIT-3 coverage closes the 19 retained requirements as 7 `DIRECT`, 10 `DERIVED`, and 2 `DIGITIZED`, with zero `ASSUMED` and zero unresolved records. It regression-checks the NTSB fade digitization and pre-registered 80% `T_crit`, NHTSA `eta=1` boundary mapping, 25 °C baseline, 40-point independent thermal discrepancy envelope, measured hydraulic-retarder map/power/drivetrain conversion, 0.25-s first-order lag mapping, and removal of unsupported physical command-slew rows while retaining actuator lags. The frozen literature config exists and the gate returns `READY_FOR_PAPER_CANDIDATE_MODEL_SIMULATION`; all 29 scripts pass. No final paper-candidate M1--M10 run was performed.
- Portability preparation adds repository-relative readiness paths, exact-case checks, LF declarations, a Linux dependency/workflow specification, a real PyTorch-autograd preflight, and one platform-neutral fail-fast command that reran all 27 accepted scripts successfully. PyTorch remains intentionally unverified on this Windows host and must be validated in the approved Linux/WSL/VM target before migration.

## LaTeX and PDF validation

- Tectonic compiled the IEEEtran source successfully into an 11-page PDF.
- All 17 cited bibliography keys resolve; no undefined references/citations or overfull boxes remain.
- Every page was rendered at 120 dpi and visually inspected. No clipping, overlap, formula corruption or unreadable page was found.

## Experiments and evidence limits

The 25,921-point normalized feasibility-envelope sweep was rerun as an algebraic unit illustration only. The provenance gate correctly stopped on the missing `results/manifest.json`, and the registered system suite correctly stopped on `AUTHOR_DECISION` and `DATA_REQUIRED`. One short synthetic DEBUG MAPPO smoke checkpoint and the Phase 2M M1--M10 DEBUG logs were generated solely for software validation; all are explicitly `paper_eligible=false`. Eight Phase 2M SVGs were generated only through the explicit DEBUG override and carry a visible non-paper watermark; the default figure command rejects these inputs. No literature-calibrated, physical-vehicle, paper-eligible trained-policy, HIL, field, communication-network or real-time experiment was run. Manuscript result placeholders and performance claims remain unchanged.

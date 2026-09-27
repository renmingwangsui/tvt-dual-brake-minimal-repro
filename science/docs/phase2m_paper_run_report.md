# Phase 2M-PAPER Run Report

Status: **PASS WITH RECORDED SCIENTIFIC LIMITATIONS — frozen model-based campaign completed and integrity-verified**.

## Frozen inputs and execution

- Campaign: `phase2m-paper-d10d13eb-6634280c`
- Runs in corrected campaign: 92 passed, 0 failed, 0 invalidated
- Superseded prior M6 runs: 38 (the original campaign is preserved byte-for-byte; these runs are excluded from final evidence)
- Protocol correction: M6 now emits explicit grade_lower and grade_upper endpoint runs and includes grade_rad in every seeded joint design-sensitivity sample. The registered grade endpoints are equal; no new perturbation range or physical value is introduced.
- Configuration SHA-256: `d10d13eb82490bcfb2151f7db2114e38d8896810255b3138d17540cddabd5104`
- Parameter-provenance SHA-256: `a66873453502b8883586c80575fd43cc8033fb4cbb129cadec2419da67c8094f`
- Digitized-source SHA-256: `e00df4e778cd2ffbf696f401fa2af60e8fb408e808a96ed8fb73646d25115daf`
- Result-manifest SHA-256: `c7c3859a3f40ee32d4d59fbb8623ce6ed92aa07ccbac440682d0eaa3790b092c`
- Code identity: `UNVERSIONED:66047dcf639e8155eba2f23b2292353a4debe1cc270ffbaac4878180713f8d5f`; the directory was not Git-versioned, so the manifest also records an immutable Python source snapshot hash `9860ffe8e1cb2a58a0b388e7e1c67c6199d43ba0142b903f9ee1dec307356755`.
- Learning execution: none. These results do not validate a DiffQP learning claim.

## Pre-run sanity audit

The loaded critical temperature was 463.708161408 K. The registered residual/model-discrepancy heat bound was ±92551.573166207 W. Units were K and W, all thermal bounds were finite, and the nominal initial state retained a positive instantaneous and predictive reserve. The discrepancy bound was retained without retuning.

## Experiment outcomes

- M1: 270/288 dual-brake feasible points versus 270/288 friction-only points. There were 192 matched points with a positive dual-brake reserve gain, but 0 newly feasible points.
- M2: all four matched allocation methods completed. Exact metrics are in `generated/tables/table_m_b_controller_comparison.tex`.
- M3: all three frozen hot starts completed; physical critical components and certificate losses, if any, are retained in `analysis_summary.json`.
- M4: predictive trigger=-- s, observed instantaneous boundary=-- s, warning=-- s. A negative predictive reserve is interpreted only as a conservative warning/inconclusive condition.
- M5: H=1,3,5,10 completed and prefix monotonicity was verified for every record.
- M6: 20 registered endpoint cases and 20 seeded joint design-sensitivity samples completed. These samples are not a physical probability distribution.
- M7: all six frozen communication cases completed; delayed `rho_up` is not equated to the instantaneous centralized minimum.
- M8: all 8 low-speed cases were finite, but the hot-brake pair changed feasibility sign across 0.5 to 0.500001 m/s (rho jump 406.098). Status: **COMPLETED_WITH_THRESHOLD_DISCONTINUITY**. This unfavorable branch-discontinuity result is retained and is not papered over.
- M9: 20 size/disturbance combinations completed and are described only as numerical/empirical evidence.
- M10: runtime percentiles were measured on the recorded Windows/Python platform; no target-hardware real-time claim is made.

## Generated artifacts

- Figures: 8 SVG files plus a provenance manifest.
- Tables: 6 LaTeX tables plus a provenance manifest.
- Raw logs remain under `results/paper_candidate/phase2m-paper-d10d13eb-6634280c/raw/` and were not rewritten during analysis.

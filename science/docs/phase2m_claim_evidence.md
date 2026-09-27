# Phase 2M Claim–Evidence Classification

Classifications follow the pre-registered decision rules; unfavorable outcomes are retained.

## C1: PARTIALLY_SUPPORTED

Dual reserve was nondecreasing and improved at 192/288 points, but both methods were feasible at 270 points; strict feasible-count enlargement was not observed.

## C2: SUPPORTED

Allocation changed=True; peak-temperature spread=94.4686 K.

## C3: INCONCLUSIVE

Predictive trigger=None; instantaneous crossing=None. Full support requires positive warning before an observed crossing.

## C4: PARTIALLY_SUPPORTED

Useful agreement was observed, but the maximum missed-trigger count was 20; full support requires zero missed triggers in every frozen channel case.

## C5: NOT_SUPPORTED

All registered cases avoid hard certificate loss=False; deterministic endpoint and seeded joint design-sensitivity results are both included.

## C6: NOT_SUPPORTED

Observed maxima were G2=4.21649 and Ginf=5.16703, exceeding the <=1 criterion; no theorem is asserted.

## C7: SUPPORTED

Worst tested P99 total controller time was 1.29069 ms versus the 100 ms sampling period, on the reported test platform only.

## Scope limitation

C1–C7 concern only the frozen non-learning/model-based campaign. Evidence that DiffQP improves MAPPO learning remains pending Phase 2C-2 and later matched learning experiments.

## Material limitations retained

- M1 produced 0 newly feasible matched grid points, so C1 is not fully supported despite positive reserve gains.
- M4 had no observed predictive trigger or instantaneous crossing, so C3 remains inconclusive.
- M6 contains hard certificate loss in registered cases, so C5 is not supported.
- M8 hot-brake feasibility changes sign across the low-speed branch threshold; this is a model/controller branch-continuity limitation.
- M9 exceeds unity in the grade-transition case, so C6 is not supported.

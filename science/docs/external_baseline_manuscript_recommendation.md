# External-baseline manuscript recommendation

## Decision

Do not modify the manuscript until the authors review this recommendation. The comparison supports a narrow, qualified discussion of what collision-focused safe multi-agent reinforcement learning does and does not cover after transfer to the frozen heavy-duty plant. It does not support a blanket superiority claim.

| Intended claim | Status | Evidence and recommended wording |
|---|---|---|
| Q1: adapted Zhou maintains geometric collision safety | **SUPPORTED** | Both methods completed all six cases with zero geometric collisions. Say only that the frozen adapted implementation remained collision-free in the registered cases. |
| Q2: collision-only cooperative CBF prevents thermal/fade feasibility loss | **NOT SUPPORTED** | The adapted baseline produced 88–324 fade-envelope violations in five descent/hot cases, with peak temperature 570.98–658.92 K, despite zero collisions. |
| Q3: the proposed framework identifies physical feasibility loss earlier or more explicitly | **INCONCLUSIVE** | The post-hoc evaluator identifies collision-demand-dominated negative reserve for the adapted trajectories, but timing comparisons are entangled with severe observation/action transfer mismatch. The accepted method also has negative reserve in four cases. |
| Q4: braking-energy allocation differs | **DESCRIPTIVE** | The accepted method generally uses substantially more auxiliary energy and less friction energy in descent/hot cases; the delay case differs. Report exact values, not a general optimality claim. |
| Q5: tracking, spacing, and comfort cost | **DESCRIPTIVE** | The accepted method has lower speed RMSE, spacing RMSE, and jerk RMS in all six frozen cases. No stochastic CI exists. |
| Q6: runtime | **DESCRIPTIVE** | The adapted Zhou controller has lower P99 runtime in all six cases because its scalar collision projection and fixed allocator are much simpler. This is a genuine disadvantage of the proposed stack and should be stated. |
| Overall algorithmic superiority | **NOT SUPPORTED / NOT MADE** | The heavy-duty transfer is adapter-sensitive, uses no retraining, and changes fleet composition. |

## Suggested manuscript placement after approval

Add one compact paragraph to the experiments section, one scenario-grouped table, and at most the two frozen figures. Keep the full native-reproduction discussion in supplementary material or the reproducibility package. A suitable qualified paragraph is:

> We additionally evaluated an independently reproduced cooperative safe-MAPPO method from Zhou et al. after a frozen scalar-acceleration-to-dual-brake adaptation. Both controllers remained geometrically collision-free in six matched heavy-duty cases. The adapted baseline was computationally faster, but its collision-only safety layer did not prevent thermal/fade violations under five descent or hot-brake cases. Because the original mixed-autonomy plant, fleet composition, and observation/action spaces differ materially, these results characterize this fixed adaptation and do not establish general algorithmic superiority.

The following sentence must accompany any table or figure:

> The heavy-duty comparison is an adapted implementation of Zhou et al., not an exact reproduction of their original mixed-autonomy physical environment.

## Required caveats

- Cite DOI `10.1109/TITS.2025.3627592` and arXiv `2411.10031`.
- State that the native reproduction passed a pre-registered sanity gate but did not numerically duplicate every published metric.
- State that the heavy policy was transferred without retraining, the heavy fleet has no human-driven vehicle, and the identity/observation/action mapping is fixed.
- Label reserve quantities for the adapted baseline as “post-hoc common feasibility evaluator.”
- Report the adapted baseline's faster runtime and any metric on which it is better.
- Do not pool the six deterministic cases into a confidence interval or treat them as independent stochastic replicates.
- Do not describe negative conservative collision-barrier values as geometric collisions.

## Frozen artifacts to cite

- Native manifest: `results/external_baselines/zhou_tits_2025/native_v1/native_result_manifest.json`, SHA-256 `a8b50a0a79d0c4a04f097373dbea299915c77dacbb7b04868384db7cfa242a69`.
- Matched comparison manifest: `results/external_baselines/zhou_tits_2025/formal_v1/result_manifest.json`, SHA-256 `5609dd929e100711de562f0e1b1b9c4de013f9bae12c6f4b97fd41dd3a35c357`.
- Exact deterministic differences: `results/external_baselines/zhou_tits_2025/formal_v1/paired_statistics.json`.
- Candidate table and figures: `table_external_baseline_comparison.tex`, `figure_EXT1_emergency_trajectory.svg`, and `figure_EXT2_brake_allocation.svg` in the frozen formal directory.

Recommended manuscript integration status: **PARTIALLY_SUPPORTED, PENDING AUTHOR REVIEW**.

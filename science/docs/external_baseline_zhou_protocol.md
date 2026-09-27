# PHASE_2M_EXT_ZHOU pre-registered protocol

This campaign independently implements the native mixed-autonomy environment and cooperative safe-MAPPO method described by Zhou, Yan, Liang, and Yang, “Enforcing Cooperative Safety for Reinforcement Learning-Based Mixed-Autonomy Platoon Control,” *IEEE Transactions on Intelligent Transportation Systems*, DOI `10.1109/TITS.2025.3627592`, open preprint `arXiv:2411.10031`.

The machine-readable authority is `configs/external_baselines/zhou_tits_2025_protocol_v1.yaml`. It was written before executing the native reproduction. Existing Phase 2M, Phase 2M-FOLLOWUP, Phase 2C-2, and Phase 2C-3 hashes and the 33/33 pre-change regression result are embedded in that file. No existing result manifest may be rewritten by this campaign.

## Native model and source mapping

The native environment implements paper equations (5), (6), (39), and (40): `dot(s_i)=v_(i-1)-v_i`; CAV acceleration equals its safe action; and HDV acceleration is `0.6[V(s_i)-v_i]+0.9[v_(i-1)-v_i]`. The desired-velocity curve is the paper's piecewise cosine function with `s_st=5 m`, `s_go=35 m`, and `v_max=30 m/s`. The last value follows algebraically from the reported equilibrium `(s,v)=(20 m,15 m/s)` because 20 m is the midpoint of the cosine branch. CAVs are vehicles 2 and 4; vehicles 1, 3, 5, 6, and 7 are HDVs.

The reward is paper equations (8)-(13): a 0.1-weighted global velocity-disturbance penalty and a 0.9-weighted sum of local efficiency and time-to-collision terms. The native controller does not use this project's heavy-duty reward.

The safety layer implements `h_i=s_i-tau*v_i`, `tau=0.3 s`, the CAV barrier rows, and the reduced cooperative candidate `h_i,suf=h_i-sum(k*h_j)` with `k=0.4`. The action is the nominal shared-actor output plus a minimum-change safe correction under `[-5,5] m/s^2`. HDV rows have nonnegative residual relaxation. The implementation follows the algebraic derivative of the reduced candidate, whose CAV action coefficient is positive; this sign also matches the paper's explanation that preceding CAV acceleration opens space for a following HDV. This choice is disclosed because the displayed sign in the open preprint's equation (21) conflicts with that derivative and narrative.

The acceleration predictor uses strictly disjoint train/calibration/test arrays and train-set input/target standardization. Its nonconformity score is the per-sample maximum absolute error, and its finite-sample conformal index is `ceil((n_cal+1)*(1-epsilon))`, after appending infinity, with `epsilon=0.01`.

## Unreported choices frozen before evaluation

The paper does not report the actor/critic layer widths, PPO discount, PPO epoch/minibatch details, entropy coefficient, barrier class-K gain, QP relaxation penalty, exact communication sets, predictor dataset sizes, or sine frequency. These are fixed in the machine-readable protocol. Neural and PPO choices use the authors' official public predecessor implementation as an auxiliary provenance source, but that repository is not claimed to be companion code for this multi-CAV paper. Full-batch PPO was selected as a declared computational choice; it is not claimed to be reported.

The native gate is evaluated once using the registered seed and tolerances. Published M5 values (2.10 s average CAV headway and 3.83 m/s AAVE) are sanity targets only. They are neither inserted into the controller nor used for tuning.

## Gate and transfer rule

The adapted heavy-duty comparison is prohibited until the native gate passes. A material collision/CBF mismatch or a sine metric outside the pre-registered comparability bands stops the campaign. If the gate passes, the next stage must keep the existing heavy-duty plant unchanged, use a fixed auxiliary-first scalar-acceleration adapter, and deny the Zhou controller access to every proposed certificate, predictive tube, backup set, physical argmin attribution, and certificate supervisor. Those quantities may only be computed post hoc.

The heavy-duty comparison is an adapted implementation of Zhou et al., not an exact reproduction of their original mixed-autonomy physical environment.

After the one-shot native gate passed, the heavy-duty transfer choices were separately frozen in `configs/external_baselines/zhou_tits_2025_adapted_protocol_v1.yaml`. This preserves the native protocol hash and makes the transfer mapping auditable before any matched heavy-duty result is viewed.

## Execution record

The native reproduction passed its registered gate. Its manifest SHA-256 is `a8b50a0a79d0c4a04f097373dbea299915c77dacbb7b04868384db7cfa242a69`.

During adapted integration, rejected attempts exposed implementation defects in the auxiliary-envelope hold and the nonnegative-speed boundary. Each correction was versioned in the adapted protocol, no rejected attempt produced a manifest, and no comparison result was used to tune a controller or threshold. The successful frozen implementation is `zhou-adapted-heavy-v1.0.5`, bound to adapted protocol SHA-256 `6ad915c2b18c512a678f2dd93f3d565a4e750477fa09e97675aa7c481332ab1e`.

Six deterministic matched cases completed with seed `20260918`. The canonical formal directory is `results/external_baselines/zhou_tits_2025/formal_v1/`; its result-manifest SHA-256 is `5609dd929e100711de562f0e1b1b9c4de013f9bae12c6f4b97fd41dd3a35c357`. The result declares `result_driven_retuning_performed=false`, `existing_frozen_evidence_modified=false`, and `novel_certificate_leakage_check=PASS`.

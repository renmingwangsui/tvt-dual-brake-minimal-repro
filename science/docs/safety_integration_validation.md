# Phase 2B safety integration validation

## Reused audited implementations

`IntegratedSafetyController` is an adapter around the existing implementations of hard rows, exact two-dimensional hard projection, complete instantaneous certificate, interval predictive reserve, set-valued backup extension, recursive fleet bottleneck, four-mode supervisor and two-pass causal loop. It does not implement a parallel simplified certificate or QP.

The controller builds explicit audit rows for collision HOCBF, thermal HOCBF or mutually exclusive low-speed ZOH/`g_T0`, fade CBF, friction and auxiliary envelopes, and the realized auxiliary-force barrier. Commands are held between samples and realized forces remain continuous through their first-order lags; no separate physical command-slew row is asserted. Jerk remains soft. Directional and digital margins are constructed through the existing audited margin functions and logged with barrier diagnostics.

## Non-reorderable callback order

For each vehicle and sample the logged order is:

1. measure physical state;
2. process aligned sensing/messages;
3. construct canonical `chi`;
4. construct uncertainty;
5. construct directional/digital margins;
6. construct hard rows and intervals;
7. compute complete instantaneous `rho`;
8. obtain the nominal DEBUG action;
9. solve the normal hard QP for `u_star_0`;
10. predict with `u_star_0` on the first transition and the existing set-valued backup extension afterward;
11. compute predictive `rho_H_cert` and critical metadata;
12. aggregate the received distributed bottleneck;
13. evaluate the existing supervisor;
14. select normal, anticipatory, certified-backup or minimal-risk action;
15. re-run prediction whenever final and provisional actions differ;
16. evaluate every final hard-row residual;
17. queue the command only after validation;
18. after all vehicles finish, integrate every plant synchronously;
19. log provisional/final actions and all certificates.

In certificate-loss minimal-risk mode the residual validator still executes and records any unavoidable negative residual; other modes reject a final action with a negative hard residual.

## Predictive and supervisor validation

Tests confirm that the predictive object's first control equals the hard-QP provisional action, while later steps have nonzero set-valued backup widths. An anticipatory threshold case changes the action and triggers the fast re-verifier before residual validation. Deterministic inputs exercise NORMAL, ANTICIPATORY, CERTIFIED_BACKUP and CERTIFICATE_LOSS_MINIMAL_RISK plus two-sample recovery hysteresis.

The low-speed closed-loop case produces non-null `g_T0` and a `thermal_zoh`/`thermal_zoh_state` hard row. Existing standalone zero/near-zero/hot/cold `g_T0` tests remain unchanged.

## Logging

Every per-vehicle step record includes time/id, physical `x`, augmented `chi`, nominal/provisional/final actions, barriers and margins, every hard-row residual, `Delta_f`, `Delta_a`, `M`, `rho`, active `g_T0`, predictive reserve, centralized diagnostic minimum, locally available recursive minimum, critical metadata, message age/validity, supervisor mode, re-verification flag, QP status and residual-validation status.

All logs enforce `data_provenance = synthetic_debug` and `paper_eligible = false`.

## Test summary and limitations

Five Phase 2B scripts cover the 27 requested areas: supported sizes and indexing, spacing, synchronous updates, observation privilege separation, all leader modes, delay/loss/burst/stale/out-of-order/inconsistency behavior, predecessor interval expansion, centralized/local minimum semantics, QP and certificates, two-pass order, backup widths, re-verification, all supervisor modes, hard residuals, low-speed `g_T0`, six `N=3` episodes and scalability execution through `N=40`.

Limitations remain: physical parameters, network distributions, sensing uncertainty, safety margins and integration-error bounds are synthetic; the environment is not calibrated; no HIL/vehicle/field/real-time validation was performed; and no MAPPO, differentiable PPO training, validated baselines or paper experiments were run. Phase 2B DEBUG closed-loop runs are software validation, not physical vehicle validation.

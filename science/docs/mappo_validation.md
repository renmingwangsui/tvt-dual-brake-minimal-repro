# Phase 2C-1 MAPPO validation

Validation date: 17 September 2026. This is DEBUG software validation, not evidence of controller performance or convergence.

## Automated coverage

Five Phase 2C-1 scripts cover all 24 requested categories:

1. `test_mappo_distribution_actor.py` checks actor shape and bounds, `log_std`, affine mapping, exact independently calculated tanh-squashed log probability, local-information exclusion, allowed centralized state, and fixed-`N` rejection of silent padding.
2. `test_mappo_buffer_gae.py` checks the three action fields, insertion, batching, a hand calculation, true-terminal bootstrap suppression, and pure-truncation bootstrap with stopped cross-boundary recursion.
3. `test_mappo_trainer.py` checks nominal-action probability ratios, positive/negative-advantage clipping, finite actor/critic gradients, updates to both networks, and additive contributions from multiple trucks to one shared actor optimizer.
4. `test_mappo_checkpoint_eval.py` checks actor, critic, optimizer, normalization, step, configuration and provenance payloads; Python/NumPy/generator RNG restoration; deterministic repeatability; and frozen evaluation statistics.
5. `test_mappo_smoke.py` checks an actual `N=3` rollout through the Phase 2B safety controller, distinct nominal/executed commands, analytical first-order actuator agreement with the executed rather than nominal command, finite two-update training, both parameter checksum changes, checkpoint reload, and deterministic evaluation.

## DEBUG smoke record

The generated record is `artifacts/debug/phase2c1_mappo_smoke.json`; its binary checkpoint is `artifacts/debug/phase2c1_mappo_smoke.pkl`. The record contains initial/final actor and critic SHA-256 parameter checksums and explicitly declares:

- `data_provenance = synthetic_debug`;
- `paper_eligible = false`;
- `training_mode = debug`;
- `diffqp_backward_active = false`.

The smoke job executes two short updates only. Finite losses and returns, changed checksums, and repeatability establish software plumbing; they do not establish learning convergence, safety performance, robustness, or publication evidence.

## Regression and evidence limits

The complete 27-script suite comprises 12 reference/theory/audit scripts, 5 Phase 2A scripts, 5 Phase 2B scripts, and 5 new Phase 2C-1 scripts. The production provenance and registered-suite gates remain fail-closed. No production physical parameter, experiment claim, manuscript result, figure, or table is created or changed by Phase 2C-1.

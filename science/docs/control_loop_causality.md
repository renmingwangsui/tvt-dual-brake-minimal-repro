# Two-pass control-loop causality

The executable order is fixed by `execute_two_pass_control_cycle`:

1. Measure/time-align state and messages.
2. Build margins and hard intervals.
3. Sample the nominal actor command.
4. Solve the normal hard QP for the provisional command.
5. Predict from that provisional command, then use set-valued backup commands.
6. Aggregate the bottleneck and choose supervisor mode.
7. Keep the provisional command or compute anticipatory/backup command.
8. Re-evaluate/fast-verify prediction if the final command differs.
9. Validate final hard rows and apply.
10. Log nominal, provisional, final, and provisional/final certificates.

The predictor rejects a missing provisional command. The causal-order test records every callback and fails if prediction precedes the normal QP or if a changed final command is not re-verified.

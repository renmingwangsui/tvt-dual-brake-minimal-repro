# Final Figure--Manuscript Consistency Audit

## Scope and controls

This audit checks Figs. 1--7 against the final manuscript equations, definitions, algorithms, captions, and frozen evidence. Figures 2, 4, 5, and 6 were eligible for a consistency-only patch. Figures 1, 3, and 7 were audited read-only. No controller parameter was retuned, no scientific experiment was rerun, and no raw scientific record was edited.

Frozen-evidence integrity was checked against the recorded SHA-256 manifests before the data-driven figures were regenerated. The M1 raw record hash remained `b30d6c435d2c9307e960d7a89be52154f07d6ab3b905d129d597ccc3f0df3d63`. The Phase 2M, formal-learning, and external-baseline result manifests remained unchanged.

## Fig. 1: Vehicle, road, and communication context

- **Purpose:** Define the platoon, predecessor-follower communication, road grade, ambient temperature, and the two braking channels.
- **Matching manuscript content:** System model and notation in Sec. III, especially the longitudinal states and friction/auxiliary braking channels.
- **Scientific match:** PASS. The figure represents the same vehicle ordering, communication direction, and actuation classes as the manuscript.
- **Visual semantics:** The environmental block is present, but its arrow visually terminates at the lead vehicle rather than explicitly broadcasting that grade and ambient conditions enter every vehicle model.
- **Changes:** None; this figure was read-only.
- **Remaining limitation:** Environment-arrow scope is visually ambiguous. This is a presentation limitation, not a scientific contradiction.

## Fig. 2: Coupled longitudinal, actuator, and thermal dynamics

- **Purpose:** Show how admissible friction and auxiliary commands propagate through actuator dynamics, vehicle motion, brake heating/cooling, and temperature-dependent fade.
- **Matching manuscript content:** The coupled plant in Sec. III, including the thermal balance in (8c) and the temperature/fade-derived friction bounds in (19)--(20).
- **Scientific match:** PASS.
- **Visual semantics:** The thermal node now includes `q_i^w`; the fade path constrains the admissible friction-command set and does not imply that fade alters the actuator-lag state equation.
- **Changes:** Added the wheel-end heat term to the displayed thermal equation, renamed the command node as admissible commands, and rerouted the fade arrow to that command constraint with the label “friction bound.”
- **Remaining limitation:** The diagram is a compact causal summary rather than a complete state-space expansion.

## Fig. 3: Runtime architecture and supervision

- **Purpose:** Summarize decentralized observations, centralized training information, the safety filter, distributed certificate exchange, supervision, and executed actions.
- **Matching manuscript content:** The runtime/training architecture and centralized-training/decentralized-execution description in Secs. IV--V.
- **Scientific match:** PASS.
- **Visual semantics:** Learning is subordinate to the physics-specific feasibility and supervisory chain; no universal closed-loop guarantee is implied.
- **Changes:** None; this figure was read-only.
- **Remaining limitation:** The architecture intentionally omits row-level algebra, which is provided in the equations and algorithms.

## Fig. 4: Conditional differentiable-training path

- **Purpose:** Distinguish the always-used forward safety projection from the conditional, training-only quadratic-program gradient path.
- **Matching manuscript content:** Sec. V-C through V-F, including the provisional hard projection, executed supervised action, regularity test, implicit Karush--Kuhn--Tucker differentiation, and predefined zero-gradient fallback.
- **Scientific match:** PASS.
- **Visual semantics:** Actor parameters produce the nominal command; the provisional two-dimensional hard quadratic program returns `u_i^{*,0}`; the forward supervisor produces the executed `u_i^*`, which alone enters the environment transition. The training-only intervention path is separate. It reaches actor parameters only when linear independence constraint qualification and strict complementarity hold; otherwise the quadratic-program contribution is zero.
- **Changes:** Rebuilt the diagram around the provisional/executed action distinction, added the explicit regularity gate and fallback branch, and removed any visual implication of an unconditional full return through the quadratic program.
- **Remaining limitation:** The compact figure names, rather than expands, the implicit linear system.

## Fig. 5: Frozen M1 physical-feasibility evidence

- **Purpose:** Compare friction-only and dual-brake feasibility membership and show dual-brake reserve improvement on the common feasible set.
- **Matching manuscript content:** The frozen M1 results in Sec. VIII and Table VIII.
- **Scientific match:** PASS.
- **Visual semantics:** The left panel uses the neutral label “feasible fraction,” because the two policies have identical feasible membership. The right panel reports reserve improvement without implying that dual braking creates new feasible grid points.
- **Changes:** Replaced the policy-specific y-axis wording with “feasible fraction.” No plotted scientific value changed.
- **Frozen membership audit:** 288 total points; BOTH feasible = 270; DUAL ONLY = 0; FRICTION ONLY ONLY = 0; NEITHER = 18. Dual feasible total = 270 and friction-only feasible total = 270. Membership is IDENTICAL.
- **Plotted-data audit:** PASS. The temperature/speed grouped feasible fractions and mean reserve gains match the immutable M1 records used by the figure generator.
- **Claim status:** The manuscript statement that dual braking adds no feasible grid point in M1 is supported.
- **Remaining limitation:** This is a finite registered grid, not a universal feasible-set theorem.

## Fig. 6: Communication and distributed-bottleneck diagnostics

- **Purpose:** Show agreement, false/missed-trigger counts, and message age for the registered M7 channel cases.
- **Matching manuscript content:** Sec. VIII, the distributed-bottleneck definition, and Table XII.
- **Scientific match:** PASS.
- **Visual semantics:** Each x-axis now states “channel-case index (see Table XII),” preventing the numeric positions from being mistaken for a continuous physical variable. The trigger-count axis starts at zero. The figure does not equate the delayed recursive bottleneck with instantaneous centralized truth.
- **Changes:** Added the Table XII cross-reference to all x-axis labels and fixed the trigger-count lower bound at zero.
- **Remaining limitation:** Case names remain in Table XII to preserve figure readability at IEEE column scale.

## Fig. 7: Formal-learning regularity/fallback diagnostic

- **Purpose:** Report whether the implicit quadratic-program backward path was valid in the frozen paired learning campaign.
- **Matching manuscript content:** Sec. VIII-C, Table XIV, and the discussion of persistent linear-independence-constraint-qualification failure.
- **Scientific match:** PASS.
- **Visual semantics:** Valid fraction = 0 and fallback fraction = 1. The figure supports a negative diagnostic result, not differentiable-learning superiority.
- **Changes:** None; this figure was read-only.
- **Remaining limitation:** The result characterizes the registered campaign and is not a universal statement about differentiable quadratic programming.

## Final disposition

All seven figures are consistent with the manuscript’s stated scientific scope after the permitted patches. The sole documented presentation ambiguity is the scope of the Fig. 1 environment arrow. No remaining figure constitutes a scientific blocker. Final visual inspection found no clipping, overlap, missing panel, or illegible plot element in the compiled manuscript.

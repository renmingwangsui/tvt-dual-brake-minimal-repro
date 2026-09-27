# Revision audit — complete predictive certificate

## Closed in this revision

- Replaced the logically incomplete use of $M_i$ as the sole certificate with $\rho_i=\min\{\Delta_f/s_f,\Delta_a/s_a,M_i/s_M,q_i^{T0}\}$, where $q_i^{T0}=g_i^{T0}/s_{T0}$ in the low-speed branch and $+\infty$ otherwise. `M_i` remains a physical collision-demand reserve even if an individual interval or the state-only thermal gate is infeasible.
- Separated the generally intractable true horizon infimum from the computable outward-rounded interval lower bound $\underline\rho^{IA}$ and certificate $\rho_i^{H,cert}$; a negative computed bound is explicitly inconclusive.
- Added a causal two-pass implementation: the normal hard-QP output is the provisional first command, and any changed supervisor command triggers fast re-verification before application.
- Added a set-valued affine-plus-saturation backup extension, logged backup-command widths, and restricted uncertainty monotonicity to inclusion-monotone closed-loop set maps and antitone evaluators.
- Distinguished the centralized fleet minimum used in training/evaluation from the decentralized successor-to-predecessor recursive minimum with timestamp, argmin metadata, and conservative stale/loss fallback.
- Expanded the backup-feasible domain to include $h^c,h^T,h^F,h^A,\psi_1^c,\psi_1^T,\rho,v$, gear/dwell admissibility, and communication-bound validity.
- Added horizon, conditional closed-loop uncertainty, and centralized/recursive fleet propositions with explicit finite-horizon scope.
- Added executable low-speed ZOH model-error and four-row digital rate-bound functions plus regression tests.
- Made the thermal HOCBF and low-speed ZOH partitions mutually exclusive, retained the undivided $c_Tu_f\le g_{T0}$ row, and added zero/near-zero/threshold hot/cold tests proving that $g_{T0}<0$ is hard infeasibility when $c_T=0$.
- Registered tube-width diagnostics, all required ablations/metrics, and eight real-log predictive figures. Generation remains fail-closed because no qualifying logs are present.
- Updated novelty positioning through 17 September 2026 and removed generic first/novel claims for MARL+CBF+DiffQP, predictive CBF, exact CBF feasibility reserve, and generic agent minima.

## Audit decisions

- Baseline interval propagation is retained. No affine-arithmetic/zonotope claim is made because no physical experiment yet demonstrates unacceptable dependency inflation; P7 is registered to decide whether a tighter set representation is needed.
- No time-to-conflict theorem is included; a global certified Dini/Clarke rate bound is unavailable.
- Temperature sensitivity is conditional on every active temperature-dependent friction upper bound being nonincreasing. No unconditional claim is made across gear/mode changes.
- A negative interval lower certificate is treated as a conservative warning, never as proof of infeasibility, unless a separately verified exact infimum is available.
- Normalizers are fixed physical scales shared by all methods; positive changes preserve sign but can change magnitude, trigger timing, learning loss and critical-row attribution, so registered sensitivity analysis is mandatory.

## Evidence status

Algebra, reference code, symbolic/finite-difference checks, box-LP equivalence, sampled one-sided-bound checks, interval-containment sanity checks, causal-order checks, closed-loop nesting checks, distributed-message regressions and normalization tests are executable. Physical parameters, trained checkpoints, validated uncertainty bounds, system logs, named hardware and real predictive figures are missing; related claims remain unverified.

# Symbol first-use audit

This audit covers symbols that carry manuscript-specific semantics. Standard mathematical operators, indices used locally in an immediately stated range, and universally conventional set/real-number notation are excluded.

| Symbol | First location | First explanation | Status | Fix applied |
|---|---|---|---|---|
| $\vartheta_i=(m_i,L_i,c_{r,i},C_{d,i}A_i,\tau_i^f,\tau_i^a,C_i,k_i,\eta_i,\bar b_i)$ | (1), Problem Formulation | Vehicle/class parameter vector and every component | PASS | Added component-by-component prose |
| $x_i=[p_i,v_i,b_i,r_i,T_i]^\top$ | (2), Problem Formulation | Physical state and every component | PASS | Added local definitions |
| $u_i=[u_i^f,u_i^a]^\top$ | (2), Problem Formulation | Exactly two commanded forces: friction and auxiliary | PASS | Added local definitions and two-coordinate statement |
| $\chi_{i,k}$ | (3), Problem Formulation | Augmented hybrid/message state | PASS | Added gear, dwell, age, gap, and predecessor definitions |
| $o_i$, $d_i$, $\Delta v_i$, $\mathcal M_i$ | (4), Problem Formulation | Local observation and time-stamped communication content | PASS | Clarified locally |
| $J_{\rm op}$ and component metrics | (5), Problem Formulation | Energy, error, jerk, and stop metrics | PASS | RMS expansion added |
| $G_{2,i}$, $G_{\infty,i}$, $\epsilon_g$ | (6), Problem Formulation | Empirical gain metrics and denominator regularizer | PASS | Existing explanation retained |
| $F_i^r$, $F_i^d$, $q_i^w$ | (7)–(8), Dynamics | Residual force, driveline force, and heat residual | PASS | Figure/source consistency retained |
| $a_i:=\dot v_i$ | Dynamics, before (10) | Longitudinal acceleration | PASS | Added explicit definition |
| $t_{h,i}$, $d_{0,i}$ | Collision HOCBF, before (11) | Registered time headway and registered standstill spacing | PASS | Added explicit first-use definitions |
| $h_i^c,h_i^T,h_i^F,h_i^A$ | (11), (16), (19), auxiliary-envelope text | Collision, thermal, fade, and auxiliary barriers | PASS | Existing/local definitions retained |
| $v_\epsilon$ | Before (21) | Registered low-speed switching threshold | PASS | Added explicit first-use definition |
| $I_{0i},I_{1i},T_{i,k+1}^{\rm base},g_{i,k}^{T0}$ | (21)–(24) | ZOH kernels, base prediction, undivided reserve | PASS | Existing definitions retained |
| $\zeta_i$, $\mathcal B_i(\hat\zeta_i)$, $\gamma_i^k$ | Before (25) | Uncertain data, declared box, directional margin | PASS | Existing definitions retained; IA expanded |
| $L_i^f,U_i^f,L_i^a,U_i^a$ | Method, before (26) | Effective friction/auxiliary command intervals | PASS | Existing definitions retained |
| $u_i^{\rm RL}$, $W_i$, $\epsilon$ | Immediately after (26) | Nominal actor command, projection weighting matrix, and positive regularization coefficient | PASS | Added explicit first-use definitions |
| $M_i$ | (28) | Physical collision supply minus demand | PASS | Existing interpretation retained |
| $q_{i,k}^{T0}$, $s_f,s_a,s_M,s_{T0}$ | (29)–(30) | Low-speed reserve and fixed positive scales | PASS | Existing definitions retained |
| $\rho_i$ | (30)–(31) | Complete four-component instantaneous certificate | PASS | Existing definition retained |
| $H_{\rm pred}$, $\Omega_\ell$, $\mathcal X^B$, $\mathbb U_i^B$, $\mathcal F_i^{\rm IA}$ | Before (34) | Horizon, uncertainty box, tube, backup set, interval extension | PASS | All defined before equation |
| $\kappa_i^B$, $\mathbb U_i^B(\mathcal X)$ | Before (34) | Pointwise state-feedback backup law and sound set-valued command enclosure | PASS | Added explicit first-use definitions |
| $\underline\rho^{\rm IA}$, $\rho_i^{H,{\rm cert}}$ | (36)–(37) | One-sided lower bound and horizon minimum | PASS | Existing definitions retained |
| $\rho_{{\rm fleet}}^{H,{\rm cert}}$, $\rho_i^{H,{\rm up}}$ | (38)–(39) | Centralized versus delayed recursive quantity | PASS | Distinction strengthened |
| $d,i_{\rm crit},\ell_{\rm crit},q_{\rm crit}$ | After (40) | Message delay and attaining vehicle/step/physical row | PASS | Added explicit metadata definitions |
| $H_L$, $\tau_s$ | Table IV / actor-objective subsection | Learning horizon and soft-min temperature | PASS | Table IV is referenced before float and precedes use |
| $G,q,\mathcal A,H_{\rm QP},B$ | Before (42) | Constraint matrix/bound, active set, Hessian, RHS | PASS | Added explicit generic-QP definitions |
| $u_{i,k}^{*,0}$, $u_i^*$ | Predictive/execution text | Provisional hard-QP output versus final supervised action | PASS | Distinction retained in text and Fig. 4 |
| $\mathcal X_i^B$ | (43) | Backup-feasible domain with all state/Boolean conditions | PASS | Existing definition retained |
| $L_i^q$, $\bar e_T$ | (44)–(45) | Digital row-rate and low-speed sampled-model error bounds | PASS | Existing definitions retained |

Audit result: **PASS — no manuscript-specific symbol is used materially before definition, and the two QP command coordinates remain unambiguous.**

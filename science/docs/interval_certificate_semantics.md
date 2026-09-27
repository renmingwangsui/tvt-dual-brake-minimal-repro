# Interval certificate semantics

The theoretical reserve is

`rho_true_H = min_l inf_{chi in X^B_l} rho(chi)`.

The executable interval monitor returns `rho_IA_lower_l`, satisfying

`rho_IA_lower_l <= inf_{chi in X^B_l} rho(chi)`,

and reports `rho_cert_H = min_l rho_IA_lower_l`. In the active low-speed branch, the evaluated certificate also includes the outward-rounded lower endpoint of `g_T0/s_T0`; at higher speed that component is inactive (`+infinity`).

- `rho_cert_H >= 0` certifies nonempty hard command sets for every state enclosed over the declared horizon.
- `rho_cert_H < 0` is an inconclusive conservative warning. It is not evidence that an admissible physical trajectory reaches infeasibility unless an independent exact optimization proves equality.
- Interval dependency may widen the gap between `rho_cert_H` and `rho_true_H`; the registered horizon/uncertainty study must quantify this.
- Code entry points are `compute_interval_certificate_lower_bound` and `compute_predictive_reserve`. The latter logs every IA lower bound and never labels it an exact infimum.

The first predicted transition uses the already computed provisional hard-QP command. Later transitions use a sound set extension of the state-feedback backup; its command widths are logged.

"""Symbolic sign/control-coefficient checks for the paper's safety rows."""
import sympy as sp

t = sp.symbols("t", real=True)
m, tau_f, tau_a, C, eta, k = sp.symbols("m tau_f tau_a C eta k", positive=True)
a_s, a_e, q = sp.symbols("a_s a_e q", positive=True)
u_f, u_a, Fr_dot = sp.symbols("u_f u_a Fr_dot", real=True)

v = sp.Function("v")(t)
vp = sp.Function("vp")(t)
d = sp.Function("d")(t)
b = sp.Function("b")(t)
r = sp.Function("r")(t)
a = sp.Function("a")(t)
ap = sp.Function("ap")(t)
jp = sp.symbols("j_p", real=True)

h_c = d - q * v - v**2 / (2 * a_s) + vp**2 / (2 * a_e)
subs1 = {sp.diff(d, t): vp - v, sp.diff(v, t): a, sp.diff(vp, t): ap}
h_c_dot = sp.simplify(sp.diff(h_c, t).subs(subs1))
expected_dot = vp - v - (q + v / a_s) * a + vp * ap / a_e
assert sp.simplify(h_c_dot - expected_dot) == 0

j = Fr_dot / m + (b - u_f) / (m * tau_f) + (r - u_a) / (m * tau_a)
subs2 = {
    sp.diff(d, t): vp - v,
    sp.diff(v, t): a,
    sp.diff(vp, t): ap,
    sp.diff(a, t): j,
    sp.diff(ap, t): jp,
}
h_c_ddot = sp.simplify(sp.diff(expected_dot, t).subs(subs2))
c = q + v / a_s
assert sp.simplify(sp.diff(h_c_ddot, u_f) - c / (m * tau_f)) == 0
assert sp.simplify(sp.diff(h_c_ddot, u_a) - c / (m * tau_a)) == 0

T = sp.Function("T")(t)
Ta = sp.Function("Ta")(t)
qw = sp.Function("qw")(t)
Tcrit = sp.symbols("Tcrit", real=True)
T_dot = (eta * b * v - k * (T - Ta) + qw) / C
h_T_dot = -T_dot
b_dot = (-b + u_f) / tau_f
thermal_subs = {sp.diff(b, t): b_dot, sp.diff(v, t): a}
h_T_ddot = sp.diff(h_T_dot, t).subs(thermal_subs)
assert sp.simplify(sp.diff(h_T_ddot, u_f) + eta * v / (C * tau_f)) == 0

bar_b, phi_T, phi_prime, alpha_F, gamma_F, mu_F = sp.symbols(
    "bar_b phi_T phi_prime alpha_F gamma_F mu_F", real=True
)
h_F = bar_b * phi_T - b
fade_lhs = bar_b * phi_prime * T_dot + (b - u_f) / tau_f + alpha_F * h_F - gamma_F - mu_F
fade_upper = b + tau_f * (bar_b * phi_prime * T_dot + alpha_F * h_F - gamma_F - mu_F)
assert sp.simplify(tau_f * fade_lhs - (fade_upper - u_f)) == 0

rbar_v, alpha_A, gamma_A, mu_A, h_A = sp.symbols("rbar_v alpha_A gamma_A mu_A h_A", real=True)
aux_lhs = rbar_v * a + (r - u_a) / tau_a + alpha_A * h_A - gamma_A - mu_A
aux_upper = r + tau_a * (rbar_v * a + alpha_A * h_A - gamma_A - mu_A)
assert sp.simplify(tau_a * aux_lhs - (aux_upper - u_a)) == 0

print("PASS: symbolic collision, thermal, fade, auxiliary, jerk, and control-coefficient identities")


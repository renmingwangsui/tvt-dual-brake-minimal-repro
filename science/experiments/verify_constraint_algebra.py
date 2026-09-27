"""Numerical sign/unit checks for manuscript constraint algebra (stdlib only)."""
from math import isclose
import random

rng = random.Random(20260916)

# Directional robust margin: Xi_hat - gamma <= Xi whenever gamma >= Xi_hat-Xi.
for _ in range(10000):
    xi_true = rng.uniform(-50.0, 50.0)
    xi_hat = xi_true + rng.uniform(-5.0, 5.0)
    gamma = max(0.0, xi_hat - xi_true) + rng.uniform(0.0, 1.0)
    assert xi_hat - gamma <= xi_true + 1e-12

# Fade-CBF rearrangement: hdot+alpha(h)>=0 iff uf is below the stated bound.
for _ in range(10000):
    tau = rng.uniform(0.05, 1.5)
    b = rng.uniform(0.0, 200.0)
    thermal_term = rng.uniform(-30.0, 10.0)  # bar_b*phi'(T)*Tdot
    alpha_h = rng.uniform(0.0, 30.0)
    upper = b + tau * (thermal_term + alpha_h)
    uf = upper - rng.uniform(0.0, 20.0)
    hdot_plus_alpha = thermal_term + (b - uf) / tau + alpha_h
    assert hdot_plus_alpha >= -1e-10

# Auxiliary command lowers the collision-required friction command.
for _ in range(10000):
    af = rng.uniform(0.01, 5.0)
    aa = rng.uniform(0.01, 5.0)
    q = rng.uniform(-20.0, 200.0)
    ua0 = rng.uniform(0.0, 100.0)
    ua1 = ua0 + rng.uniform(0.0, 100.0)
    ell0 = (q - aa * ua0) / af
    ell1 = (q - aa * ua1) / af
    assert ell1 <= ell0 + 1e-12
    assert isclose((ell1 - ell0) / (ua1 - ua0), -aa / af, rel_tol=1e-10)

# Unbounded nonnegative jerk slack can satisfy either affine jerk value.
for _ in range(10000):
    jerk = rng.uniform(-500.0, 500.0)
    jmax = rng.uniform(0.1, 10.0)
    delta = max(0.0, abs(jerk) - jmax)
    assert -jmax - delta - 1e-12 <= jerk <= jmax + delta + 1e-12

print("PASS: 40,000 directional-margin, fade-CBF, auxiliary-monotonicity, and jerk-slack checks")

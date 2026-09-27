"""Reference tests for nominal likelihood and next-state reserve loss semantics."""
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from actor_parameterization import one_step_reserve_loss, squashed_gaussian_sample  # noqa: E402

sample = squashed_gaussian_sample(
    mean=(0.1, -0.2),
    log_std=(-0.5, -0.8),
    noise=(0.4, -1.1),
    lower=(0.0, 0.0),
    upper=(180_000.0, 120_000.0),
)
assert 0.0 <= sample["command"][0] <= 180_000.0
assert 0.0 <= sample["command"][1] <= 120_000.0
assert math.isfinite(sample["nominal_log_prob"])


def step(state, command):
    return (state[0] + 1e-5 * command[0], state[1] + 2e-5 * command[1])


def reserve(state):
    return state[0] + state[1]


low = one_step_reserve_loss((0.0, 0.0), (10_000.0, 10_000.0), step, reserve, 2.0)
high = one_step_reserve_loss((0.0, 0.0), (20_000.0, 20_000.0), step, reserve, 2.0)
assert high["predicted_reserve"] > low["predicted_reserve"]
assert high["loss"] < low["loss"]
print("PASS: nominal squashed-Gaussian likelihood and action-dependent next-state reserve loss")

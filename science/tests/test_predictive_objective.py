"""Predictive actor objective and gradient-policy semantics."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from training_objective import (  # noqa: E402
    GradientPathPolicy,
    ReserveLossWeights,
    actor_objective,
    predictive_reserve_losses,
)


def main() -> None:
    losses = predictive_reserve_losses(
        [[3.0, 2.0, 1.0], [4.0, 3.0, 2.0]],
        [2.0, 3.0],
        instant_reference=1.0,
        predictive_reference=1.5,
        fleet_reference=1.2,
        soft_temperature=0.1,
    )
    assert losses["L_predictive_reserve"] > 0.0
    assert losses["L_fleet_reserve"] > 0.0
    policy = GradientPathPolicy(3, True, 3, True, 1e-6, 1.0)
    policy.validate()
    weights = ReserveLossWeights(0.01, 1.0, 0.5, 0.7, 0.8, 0.3, 0.2)
    objective = actor_objective(1.0, 0.5, 0.1, 0.2, 0.3, 0.4, 0.1, 0.1, weights)
    assert isinstance(objective, float)
    print("PASS: predictive vehicle/fleet losses and explicit gradient-path policy")


if __name__ == "__main__":
    main()

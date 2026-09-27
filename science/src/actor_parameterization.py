"""Reference nominal-action parameterization and anticipatory reserve loss.

This module fixes the probability semantics used by the paper.  It is a small
NumPy-free reference, not a trained MAPPO implementation.  A future PyTorch/JAX
implementation must match these equations and retain the nominal log-probability.
"""
from __future__ import annotations

from math import exp, log, pi, tanh
from typing import Callable, Dict, Sequence, Tuple


def squashed_gaussian_sample(
    mean: Sequence[float],
    log_std: Sequence[float],
    noise: Sequence[float],
    lower: Sequence[float],
    upper: Sequence[float],
) -> Dict[str, Tuple[float, float] | float]:
    if not all(len(x) == 2 for x in (mean, log_std, noise, lower, upper)):
        raise ValueError("dual-brake actor expects two coordinates")
    raw = tuple(mean[i] + exp(log_std[i]) * noise[i] for i in range(2))
    squashed = tuple(tanh(value) for value in raw)
    scale = tuple((upper[i] - lower[i]) / 2.0 for i in range(2))
    shift = tuple((upper[i] + lower[i]) / 2.0 for i in range(2))
    command = tuple(shift[i] + scale[i] * squashed[i] for i in range(2))
    normal_log_prob = sum(
        -0.5 * (((raw[i] - mean[i]) / exp(log_std[i])) ** 2 + 2.0 * log_std[i] + log(2.0 * pi))
        for i in range(2)
    )
    log_abs_det = sum(log(max(1e-12, scale[i] * (1.0 - squashed[i] ** 2))) for i in range(2))
    return {"raw": raw, "command": command, "nominal_log_prob": normal_log_prob - log_abs_det}


def softplus(value: float) -> float:
    if value > 30.0:
        return value
    if value < -30.0:
        return exp(value)
    return log(1.0 + exp(value))


def one_step_reserve_loss(
    state: Sequence[float],
    executed_command: Sequence[float],
    differentiable_step: Callable[[Sequence[float], Sequence[float]], Sequence[float]],
    reserve_at_state: Callable[[Sequence[float]], float],
    reserve_reference: float,
) -> Dict[str, float | Sequence[float]]:
    """Loss uses M(x_{k+1}^{pred}), so the executed command has a pathwise route."""
    predicted_state = differentiable_step(state, executed_command)
    predicted_reserve = reserve_at_state(predicted_state)
    return {
        "predicted_state": predicted_state,
        "predicted_reserve": predicted_reserve,
        "loss": softplus(reserve_reference - predicted_reserve) ** 2,
    }

"""Registered predictive-reserve actor objective components.

The weights are configuration inputs, never inferred or presented as tuned
values.  Positive rewards use the maximization sign convention in the paper.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import exp, log
from typing import Mapping, Sequence

from safety.fleet_reserve import softmin


def softplus(value: float) -> float:
    if value > 40.0:
        return value
    if value < -40.0:
        return exp(value)
    return log(1.0 + exp(value))


@dataclass(frozen=True)
class ReserveLossWeights:
    entropy: float
    intervention: float
    instant_reserve: float
    predictive_reserve: float
    fleet_reserve: float
    thermal: float
    fade: float


@dataclass(frozen=True)
class GradientPathPolicy:
    rollout_horizon: int
    differentiate_each_qp: bool
    truncated_bptt_steps: int
    detach_certified_interval_monitor: bool
    active_set_tolerance: float
    gradient_clip_norm: float

    def validate(self) -> None:
        if self.rollout_horizon < 1 or self.truncated_bptt_steps < 1:
            raise ValueError("positive rollout and truncated-BPTT horizons required")
        if not self.detach_certified_interval_monitor:
            raise ValueError("certified interval extrema must be detached from actor gradients")
        if self.active_set_tolerance <= 0.0 or self.gradient_clip_norm <= 0.0:
            raise ValueError("positive active-set tolerance and clip norm required")


def predictive_reserve_losses(
    vehicle_reserve_sequences: Sequence[Sequence[float]],
    instant_reserves: Sequence[float],
    instant_reference: float,
    predictive_reference: float,
    fleet_reference: float,
    soft_temperature: float,
) -> Mapping[str, object]:
    if not vehicle_reserve_sequences:
        raise ValueError("at least one vehicle sequence is required")
    vehicle_soft = tuple(softmin(sequence, soft_temperature) for sequence in vehicle_reserve_sequences)
    fleet_soft = softmin(vehicle_soft, soft_temperature)
    instant_loss = sum(softplus(instant_reference - value) ** 2 for value in instant_reserves) / len(instant_reserves)
    predictive_loss = sum(softplus(predictive_reference - value) ** 2 for value in vehicle_soft) / len(vehicle_soft)
    fleet_loss = softplus(fleet_reference - fleet_soft) ** 2
    return {
        "vehicle_predictive_reserve": vehicle_soft,
        "fleet_predictive_reserve": fleet_soft,
        "L_instant_reserve": instant_loss,
        "L_predictive_reserve": predictive_loss,
        "L_fleet_reserve": fleet_loss,
    }


def actor_objective(
    L_clip: float,
    entropy: float,
    L_intervention: float,
    L_instant_reserve: float,
    L_predictive_reserve: float,
    L_fleet_reserve: float,
    L_thermal: float,
    L_fade: float,
    weights: ReserveLossWeights,
) -> float:
    return (
        L_clip
        + weights.entropy * entropy
        - weights.intervention * L_intervention
        - weights.instant_reserve * L_instant_reserve
        - weights.predictive_reserve * L_predictive_reserve
        - weights.fleet_reserve * L_fleet_reserve
        - weights.thermal * L_thermal
        - weights.fade * L_fade
    )

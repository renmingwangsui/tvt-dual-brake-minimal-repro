"""Modular synthetic-debug reward terms; production weights remain CONFIG_REQUIRED."""
from __future__ import annotations

from dataclasses import dataclass

from envs.heavy_platoon_env import Phase2BLogRecord


@dataclass(frozen=True)
class DebugRewardWeights:
    speed_tracking: float
    spacing_tracking: float
    control_effort: float
    friction_usage: float
    auxiliary_usage: float
    intervention: float
    safety_penalty: float
    data_provenance: str = "synthetic_debug"
    paper_eligible: bool = False

    def __post_init__(self) -> None:
        if self.data_provenance != "synthetic_debug" or self.paper_eligible:
            raise ValueError("DEBUG reward weights cannot be paper eligible")


@dataclass(frozen=True)
class RewardBreakdown:
    total: float
    speed_tracking: float
    spacing_tracking: float
    control_effort: float
    friction_usage: float
    auxiliary_usage: float
    intervention: float
    safety_penalty: float


class DebugReward:
    def __init__(self, weights: DebugRewardWeights, target_speed_mps: float, target_gap_m: float) -> None:
        self.weights = weights
        self.target_speed_mps = target_speed_mps
        self.target_gap_m = target_gap_m

    def __call__(self, record: Phase2BLogRecord) -> RewardBreakdown:
        speed = -(record.physical_state.speed_mps - self.target_speed_mps) ** 2
        gap = -(record.augmented_state.gap_m - self.target_gap_m) ** 2
        effort = -(record.final_action[0] ** 2 + record.final_action[1] ** 2) / 1e10
        friction = -record.final_action[0] / 100_000.0
        auxiliary = -record.final_action[1] / 100_000.0
        intervention = -sum((a - b) ** 2 for a, b in zip(record.final_action, record.nominal_action)) / 1e10
        safety = min(0.0, record.rho_H_cert)
        terms = (
            self.weights.speed_tracking * speed,
            self.weights.spacing_tracking * gap,
            self.weights.control_effort * effort,
            self.weights.friction_usage * friction,
            self.weights.auxiliary_usage * auxiliary,
            self.weights.intervention * intervention,
            self.weights.safety_penalty * safety,
        )
        return RewardBreakdown(sum(terms), *terms)

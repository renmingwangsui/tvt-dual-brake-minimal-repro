"""Causal predecessor-following nominal DEBUG controller."""
from __future__ import annotations

from dataclasses import dataclass

from envs.observations import LocalActorObservation


@dataclass(frozen=True)
class PredecessorFollowingDebugController:
    standstill_gap_m: float
    time_headway_s: float
    gap_gain_N_per_m: float
    closing_gain_N_per_mps: float
    auxiliary_fraction: float = 0.6
    data_provenance: str = "synthetic_debug"
    paper_eligible: bool = False

    def command(self, observation: LocalActorObservation) -> tuple[float, float]:
        desired_gap = self.standstill_gap_m + self.time_headway_s * observation.own_speed_mps
        shortage = max(0.0, desired_gap - observation.gap_m)
        closing_speed = max(0.0, -observation.relative_speed_mps)
        demand = self.gap_gain_N_per_m * shortage + self.closing_gain_N_per_mps * closing_speed
        auxiliary = self.auxiliary_fraction * demand
        return demand - auxiliary, auxiliary

"""Synthetic-debug nominal brake policies; these are not paper baselines."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from envs.observations import LocalActorObservation


class NominalDebugController(Protocol):
    data_provenance: str
    paper_eligible: bool
    def command(self, observation: LocalActorObservation) -> tuple[float, float]: ...


@dataclass(frozen=True)
class ZeroCommandController:
    data_provenance: str = "synthetic_debug"
    paper_eligible: bool = False

    def command(self, observation: LocalActorObservation) -> tuple[float, float]:
        return 0.0, 0.0


@dataclass(frozen=True)
class ConstantCommandController:
    friction_command_N: float
    auxiliary_command_N: float
    data_provenance: str = "synthetic_debug"
    paper_eligible: bool = False

    def command(self, observation: LocalActorObservation) -> tuple[float, float]:
        return max(0.0, self.friction_command_N), max(0.0, self.auxiliary_command_N)


@dataclass(frozen=True)
class SpeedTrackingDebugController:
    target_speed_mps: float
    gain_N_per_mps: float
    auxiliary_fraction: float = 0.5
    data_provenance: str = "synthetic_debug"
    paper_eligible: bool = False

    def command(self, observation: LocalActorObservation) -> tuple[float, float]:
        demand = max(0.0, self.gain_N_per_mps * (observation.own_speed_mps - self.target_speed_mps))
        auxiliary = self.auxiliary_fraction * demand
        return demand - auxiliary, auxiliary


@dataclass(frozen=True)
class AuxiliaryFirstDescentController:
    target_speed_mps: float
    gain_N_per_mps: float
    nominal_auxiliary_cap_N: float
    data_provenance: str = "synthetic_debug"
    paper_eligible: bool = False

    def command(self, observation: LocalActorObservation) -> tuple[float, float]:
        demand = max(0.0, self.gain_N_per_mps * (observation.own_speed_mps - self.target_speed_mps))
        auxiliary = min(demand, self.nominal_auxiliary_cap_N)
        return demand - auxiliary, auxiliary

"""Auxiliary-brake interfaces; no manufacturer retarder map is embedded."""
from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from typing import Mapping, Protocol

from .truck_dynamics import analytical_first_order_response, first_order_lag_rate


class AuxiliaryForceInterface(Protocol):
    def force_limit_N(self, speed_mps: float, gear: int) -> float: ...


@dataclass(frozen=True)
class SyntheticDebugAuxiliaryEnvelope:
    """Nonphysical speed/gear/power envelope for interface validation only."""

    speed_points_mps: tuple[float, ...]
    availability_factors: tuple[float, ...]
    gear_force_limits_N: Mapping[int, float]
    power_limit_W: float
    data_provenance: str = "synthetic_debug"
    paper_eligible: bool = False

    def __post_init__(self) -> None:
        if len(self.speed_points_mps) < 2 or len(self.speed_points_mps) != len(self.availability_factors):
            raise ValueError("auxiliary envelope grids must have equal length >= 2")
        if any(b <= a for a, b in zip(self.speed_points_mps, self.speed_points_mps[1:])):
            raise ValueError("speed points must be strictly increasing")
        if any(not 0.0 <= value <= 1.0 for value in self.availability_factors):
            raise ValueError("availability factors must lie in [0, 1]")
        if self.power_limit_W <= 0.0 or any(value <= 0.0 for value in self.gear_force_limits_N.values()):
            raise ValueError("synthetic auxiliary limits must be positive")
        if self.data_provenance != "synthetic_debug" or self.paper_eligible:
            raise ValueError("this envelope is synthetic-debug and never paper eligible")

    def _availability(self, speed_mps: float) -> float:
        if speed_mps < 0.0:
            raise ValueError("speed must be nonnegative")
        if speed_mps <= self.speed_points_mps[0]:
            return self.availability_factors[0]
        if speed_mps >= self.speed_points_mps[-1]:
            return self.availability_factors[-1]
        index = bisect_right(self.speed_points_mps, speed_mps) - 1
        x0, x1 = self.speed_points_mps[index : index + 2]
        y0, y1 = self.availability_factors[index : index + 2]
        return y0 + (speed_mps - x0) * (y1 - y0) / (x1 - x0)

    def force_limit_N(self, speed_mps: float, gear: int) -> float:
        if gear not in self.gear_force_limits_N:
            raise ValueError(f"gear {gear} has no registered auxiliary limit")
        speed_limited = self.gear_force_limits_N[gear] * self._availability(speed_mps)
        if speed_mps == 0.0:
            return 0.0
        return max(0.0, min(speed_limited, self.power_limit_W / speed_mps))


def auxiliary_actuator_rate(realized_force_N: float, command_force_N: float, tau_a_s: float) -> float:
    return first_order_lag_rate(realized_force_N, command_force_N, tau_a_s)


def analytical_auxiliary_response_N(
    initial_force_N: float, command_force_N: float, elapsed_s: float, tau_a_s: float
) -> float:
    return analytical_first_order_response(initial_force_N, command_force_N, elapsed_s, tau_a_s)

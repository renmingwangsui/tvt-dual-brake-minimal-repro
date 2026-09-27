"""Gear state, dwell timer and permitted-transition interface."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GearboxParameters:
    gears: tuple[int, ...]
    permitted_shifts: frozenset[tuple[int, int]]
    minimum_dwell_s: float

    def __post_init__(self) -> None:
        if not self.gears or len(set(self.gears)) != len(self.gears):
            raise ValueError("gears must be nonempty and unique")
        if self.minimum_dwell_s < 0.0:
            raise ValueError("minimum dwell must be nonnegative")
        known = set(self.gears)
        if any(source not in known or target not in known for source, target in self.permitted_shifts):
            raise ValueError("permitted shift references an unknown gear")


@dataclass(frozen=True)
class GearState:
    gear: int
    dwell_time_s: float


def advance_dwell_timer(state: GearState, elapsed_s: float, parameters: GearboxParameters) -> GearState:
    if state.gear not in parameters.gears or elapsed_s < 0.0:
        raise ValueError("invalid gear state or elapsed time")
    return GearState(state.gear, state.dwell_time_s + elapsed_s)


def can_shift(state: GearState, target_gear: int, parameters: GearboxParameters) -> bool:
    return (
        target_gear in parameters.gears
        and (state.gear, target_gear) in parameters.permitted_shifts
        and state.dwell_time_s >= parameters.minimum_dwell_s
    )


def shift_gear(state: GearState, target_gear: int, parameters: GearboxParameters) -> GearState:
    if not can_shift(state, target_gear, parameters):
        raise ValueError("shift is not permitted or dwell time has not elapsed")
    return GearState(target_gear, 0.0)

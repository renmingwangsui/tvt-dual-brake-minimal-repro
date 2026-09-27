"""Position-indexed road-grade models.

Position increases in the vehicle's forward travel direction. Positive grade
angle is uphill and negative grade angle is downhill.
"""
from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from typing import Protocol


class RoadProfile(Protocol):
    def grade_rad(self, position_m: float) -> float: ...


@dataclass(frozen=True)
class ConstantGrade:
    value_rad: float

    def grade_rad(self, position_m: float) -> float:
        return self.value_rad


@dataclass(frozen=True)
class PiecewiseLinearGrade:
    positions_m: tuple[float, ...]
    grades_rad: tuple[float, ...]
    boundary_policy: str = "clamp"

    def __post_init__(self) -> None:
        if len(self.positions_m) < 2 or len(self.positions_m) != len(self.grades_rad):
            raise ValueError("road samples must have equal length >= 2")
        if any(b <= a for a, b in zip(self.positions_m, self.positions_m[1:])):
            raise ValueError("road positions must be strictly increasing")
        if self.boundary_policy not in {"clamp", "error"}:
            raise ValueError("boundary policy must be clamp or error")

    def grade_rad(self, position_m: float) -> float:
        if position_m < self.positions_m[0]:
            if self.boundary_policy == "error":
                raise ValueError("position lies below registered road domain")
            return self.grades_rad[0]
        if position_m > self.positions_m[-1]:
            if self.boundary_policy == "error":
                raise ValueError("position lies above registered road domain")
            return self.grades_rad[-1]
        if position_m == self.positions_m[-1]:
            return self.grades_rad[-1]
        index = bisect_right(self.positions_m, position_m) - 1
        x0, x1 = self.positions_m[index : index + 2]
        y0, y1 = self.grades_rad[index : index + 2]
        return y0 + (position_m - x0) * (y1 - y0) / (x1 - x0)


class InterpolatedRoadProfile(PiecewiseLinearGrade):
    """Named sampled-profile interface using the validated linear interpolant."""

    @classmethod
    def from_samples(
        cls,
        positions_m: tuple[float, ...],
        grades_rad: tuple[float, ...],
        boundary_policy: str = "clamp",
    ) -> "InterpolatedRoadProfile":
        return cls(positions_m, grades_rad, boundary_policy)

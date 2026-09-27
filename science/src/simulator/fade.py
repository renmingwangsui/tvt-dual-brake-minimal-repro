"""Calibratable brake-fade interface with explicit registered-domain policy."""
from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from typing import Protocol, Sequence


class FadeModel(Protocol):
    def value(self, temperature_K: float) -> float: ...
    def derivative_per_K(self, temperature_K: float) -> float: ...


@dataclass(frozen=True)
class TabulatedFadeCurve:
    temperatures_K: tuple[float, ...]
    factors: tuple[float, ...]
    outside_domain_policy: str = "error"
    data_provenance: str = "production_manifest_required"
    paper_eligible: bool = False

    def __post_init__(self) -> None:
        if len(self.temperatures_K) < 2 or len(self.temperatures_K) != len(self.factors):
            raise ValueError("fade table requires equal temperature/factor arrays of length >= 2")
        if any(b <= a for a, b in zip(self.temperatures_K, self.temperatures_K[1:])):
            raise ValueError("fade temperature grid must be strictly increasing")
        if any(not 0.0 < factor <= 1.0 for factor in self.factors):
            raise ValueError("fade factors must lie in (0, 1]")
        if any(b > a for a, b in zip(self.factors, self.factors[1:])):
            raise ValueError("fade table must be monotone nonincreasing")
        if self.outside_domain_policy not in {"error", "conservative_clip"}:
            raise ValueError("outside-domain policy must be error or conservative_clip")
        if self.data_provenance == "synthetic_debug" and self.paper_eligible:
            raise ValueError("synthetic fade data cannot be paper eligible")

    def _segment(self, temperature_K: float) -> int | None:
        low, high = self.temperatures_K[0], self.temperatures_K[-1]
        if not low <= temperature_K <= high:
            if self.outside_domain_policy == "error":
                raise ValueError("temperature is outside the registered fade domain")
            return None
        return min(bisect_right(self.temperatures_K, temperature_K) - 1, len(self.temperatures_K) - 2)

    def value(self, temperature_K: float) -> float:
        index = self._segment(temperature_K)
        if index is None:
            return min(self.factors)
        x0, x1 = self.temperatures_K[index : index + 2]
        y0, y1 = self.factors[index : index + 2]
        weight = (temperature_K - x0) / (x1 - x0)
        return y0 + weight * (y1 - y0)

    def derivative_per_K(self, temperature_K: float) -> float:
        index = self._segment(temperature_K)
        if index is None:
            return 0.0
        x0, x1 = self.temperatures_K[index : index + 2]
        y0, y1 = self.factors[index : index + 2]
        return (y1 - y0) / (x1 - x0)


def synthetic_debug_fade_curve(outside_domain_policy: str = "error") -> TabulatedFadeCurve:
    """Return a deliberately nonphysical monotone curve for software tests only."""
    return TabulatedFadeCurve(
        temperatures_K=(300.0, 400.0, 500.0, 600.0),
        factors=(1.0, 0.9, 0.65, 0.4),
        outside_domain_policy=outside_domain_policy,
        data_provenance="synthetic_debug",
        paper_eligible=False,
    )

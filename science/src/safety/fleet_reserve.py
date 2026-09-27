"""Fleet bottleneck reserve and differentiable soft-minimum utilities."""
from __future__ import annotations

from dataclasses import dataclass
from math import exp, log
from typing import Mapping, Sequence


@dataclass(frozen=True)
class FleetReserveResult:
    fleet_reserve: float
    critical_vehicle: int
    critical_step: int
    reserve_by_vehicle: tuple[float, ...]
    critical_limit_type: str

    @property
    def rho_fleet_H(self) -> float:
        return self.fleet_reserve


@dataclass(frozen=True)
class UpstreamSafetyMessage:
    """Age-bounded successor-to-predecessor recursive min aggregation."""

    rho_upstream_H: float
    critical_vehicle_id: int
    critical_future_step: int
    limiting_physical_constraint: str
    timestamp_s: float
    fallback_used: bool = False


def aggregate_upstream_message(
    local_vehicle_id: int,
    local_rho_cert_H: float,
    local_critical_step: int,
    local_limit_type: str,
    timestamp_s: float,
    received_successor: UpstreamSafetyMessage | None,
    maximum_message_age_s: float,
    registered_stale_fallback: UpstreamSafetyMessage,
) -> UpstreamSafetyMessage:
    """Propagate the minimum and argmin using only an allowed successor message.

    Packet loss and stale packets are replaced by the pre-registered
    conservative fallback; they are never silently treated as no risk.
    """
    if maximum_message_age_s < 0.0:
        raise ValueError("maximum message age must be nonnegative")
    local = UpstreamSafetyMessage(
        local_rho_cert_H, local_vehicle_id, local_critical_step,
        local_limit_type, timestamp_s, False,
    )
    received = received_successor
    if received is None or timestamp_s - received.timestamp_s > maximum_message_age_s:
        received = UpstreamSafetyMessage(
            registered_stale_fallback.rho_upstream_H,
            registered_stale_fallback.critical_vehicle_id,
            registered_stale_fallback.critical_future_step,
            registered_stale_fallback.limiting_physical_constraint,
            timestamp_s,
            True,
        )
    return received if received.rho_upstream_H < local.rho_upstream_H else local


def select_bottleneck_vehicle(vehicle_predictive_results: Sequence[object]) -> tuple[int, int]:
    if not vehicle_predictive_results:
        raise ValueError("at least one vehicle result is required")
    index = min(
        range(len(vehicle_predictive_results)),
        key=lambda i: float(getattr(vehicle_predictive_results[i], "min_reserve")),
    )
    return index, int(getattr(vehicle_predictive_results[index], "critical_step"))


def compute_fleet_reserve(vehicle_predictive_results: Sequence[object]) -> FleetReserveResult:
    vehicle, step = select_bottleneck_vehicle(vehicle_predictive_results)
    reserves = tuple(float(getattr(item, "min_reserve")) for item in vehicle_predictive_results)
    limit_type = str(getattr(vehicle_predictive_results[vehicle], "critical_limit_type", "unspecified"))
    return FleetReserveResult(min(reserves), vehicle, step, reserves, limit_type)


def softmin(values: Sequence[float], temperature: float) -> float:
    if not values:
        raise ValueError("softmin needs at least one value")
    if temperature <= 0.0:
        raise ValueError("softmin temperature must be positive")
    minimum = min(values)
    return minimum - temperature * log(sum(exp(-(value - minimum) / temperature) for value in values))

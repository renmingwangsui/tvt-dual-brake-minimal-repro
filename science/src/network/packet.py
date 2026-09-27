"""Typed V2V packets carrying historical, timestamped motion or safety data."""
from __future__ import annotations

from dataclasses import dataclass, replace
from math import isfinite


@dataclass(frozen=True)
class V2VPacket:
    source_vehicle_id: int
    target_vehicle_id: int
    generation_time_s: float
    sequence_number: int
    packet_kind: str
    position_m: float
    speed_mps: float
    acceleration_mps2: float
    declared_jerk_bound_mps3: float
    receive_time_s: float | None = None
    rho_H_cert: float | None = None
    critical_vehicle_id: int | None = None
    critical_prediction_step: int | None = None
    critical_component: str | None = None
    requested_deceleration_mps2: float | None = None
    safety_intent: str | None = None
    data_provenance: str = "synthetic_debug"
    paper_eligible: bool = False

    def __post_init__(self) -> None:
        if self.source_vehicle_id < 0 or self.target_vehicle_id < 0:
            raise ValueError("vehicle ids must be nonnegative")
        if self.source_vehicle_id == self.target_vehicle_id:
            raise ValueError("packet source and target must differ")
        if self.generation_time_s < 0.0 or self.sequence_number < 0:
            raise ValueError("packet time and sequence must be nonnegative")
        if self.packet_kind not in {"motion", "bottleneck", "safety_intent"}:
            raise ValueError("unsupported packet kind")
        numeric = (
            self.position_m,
            self.speed_mps,
            self.acceleration_mps2,
            self.declared_jerk_bound_mps3,
        )
        if not all(isfinite(value) for value in numeric):
            raise ValueError("packet motion fields must be finite")
        if self.speed_mps < 0.0 or self.declared_jerk_bound_mps3 < 0.0:
            raise ValueError("packet speed and jerk bound must be nonnegative")
        if self.receive_time_s is not None and self.receive_time_s < self.generation_time_s:
            raise ValueError("receive time cannot precede generation time")
        if self.data_provenance == "synthetic_debug" and self.paper_eligible:
            raise ValueError("synthetic-debug packets cannot be paper eligible")

    def delivered_at(self, receive_time_s: float) -> "V2VPacket":
        return replace(self, receive_time_s=receive_time_s)

    def age_s(self, current_time_s: float) -> float:
        age = current_time_s - self.generation_time_s
        if age < -1e-12:
            raise ValueError("cannot evaluate packet age before generation")
        return max(0.0, age)

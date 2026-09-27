"""Timestamp/order/consistency validation for delivered historical packets."""
from __future__ import annotations

from dataclasses import dataclass

from .packet import V2VPacket


@dataclass(frozen=True)
class DebugConsistencyTolerances:
    position_tolerance_m: float
    speed_tolerance_mps: float
    data_provenance: str = "synthetic_debug"
    paper_eligible: bool = False

    def __post_init__(self) -> None:
        if min(self.position_tolerance_m, self.speed_tolerance_mps) < 0.0:
            raise ValueError("consistency tolerances must be nonnegative")
        if self.data_provenance == "synthetic_debug" and self.paper_eligible:
            raise ValueError("synthetic-debug tolerances cannot be paper eligible")
        if self.paper_eligible and self.data_provenance != "literature_calibrated":
            raise ValueError("paper-eligible tolerances require literature calibration")


@dataclass(frozen=True)
class LocalMotionSensing:
    predecessor_position_m: float
    predecessor_speed_mps: float


@dataclass(frozen=True)
class PacketDecision:
    accepted: bool
    reason: str
    age_s: float
    packet: V2VPacket


def packet_consistent_with_local_sensing(
    packet: V2VPacket,
    current_time_s: float,
    sensing: LocalMotionSensing,
    tolerances: DebugConsistencyTolerances,
) -> tuple[bool, str]:
    age = packet.age_s(current_time_s)
    projected_speed = max(0.0, packet.speed_mps + packet.acceleration_mps2 * age)
    projected_position = (
        packet.position_m
        + packet.speed_mps * age
        + 0.5 * packet.acceleration_mps2 * age * age
    )
    if abs(projected_position - sensing.predecessor_position_m) > tolerances.position_tolerance_m:
        return False, "position_inconsistent_with_local_range"
    if abs(projected_speed - sensing.predecessor_speed_mps) > tolerances.speed_tolerance_mps:
        return False, "speed_inconsistent_with_local_range_rate"
    return True, "consistent"


class MessageBuffer:
    def __init__(self, maximum_age_s: float, tolerances: DebugConsistencyTolerances) -> None:
        if maximum_age_s < 0.0:
            raise ValueError("maximum message age must be nonnegative")
        self.maximum_age_s = maximum_age_s
        self.tolerances = tolerances
        self._latest: dict[tuple[int, str], V2VPacket] = {}
        self.decision_log: list[PacketDecision] = []

    def ingest(
        self,
        packet: V2VPacket,
        current_time_s: float,
        sensing: LocalMotionSensing | None = None,
    ) -> PacketDecision:
        age = packet.age_s(current_time_s)
        key = (packet.source_vehicle_id, packet.packet_kind)
        prior = self._latest.get(key)
        if age > self.maximum_age_s + 1e-12:
            decision = PacketDecision(False, "stale", age, packet)
        elif prior is not None and packet.sequence_number <= prior.sequence_number:
            decision = PacketDecision(False, "out_of_order", age, packet)
        elif sensing is not None and packet.packet_kind == "motion":
            valid, reason = packet_consistent_with_local_sensing(
                packet, current_time_s, sensing, self.tolerances
            )
            decision = PacketDecision(valid, "accepted" if valid else reason, age, packet)
        else:
            decision = PacketDecision(True, "accepted", age, packet)
        if decision.accepted:
            self._latest[key] = packet
        self.decision_log.append(decision)
        return decision

    def latest_valid(
        self, source_vehicle_id: int, packet_kind: str, current_time_s: float
    ) -> tuple[V2VPacket | None, str, float | None]:
        packet = self._latest.get((source_vehicle_id, packet_kind))
        if packet is None:
            return None, "missing", None
        age = packet.age_s(current_time_s)
        if age > self.maximum_age_s + 1e-12:
            return None, "stale", age
        return packet, "fresh", age

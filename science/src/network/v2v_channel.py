"""Event-driven V2V delay/loss channel; payloads remain historical snapshots."""
from __future__ import annotations

from dataclasses import dataclass, field
import heapq
import random

from .packet import V2VPacket


@dataclass(frozen=True)
class ChannelConfig:
    delay_model: str = "zero"
    fixed_delay_s: float = 0.0
    maximum_random_delay_s: float = 0.0
    long_tail_scale_s: float = 0.1
    long_tail_cap_s: float = 2.0
    packet_loss_probability: float = 0.0
    burst_loss_probability: float = 0.0
    burst_length: int = 0
    forced_loss_sequences: frozenset[int] = frozenset()
    forced_burst_starts: frozenset[int] = frozenset()
    random_seed: int = 20260917

    def __post_init__(self) -> None:
        if self.delay_model not in {"zero", "fixed", "random_bounded", "long_tail"}:
            raise ValueError("unsupported delay model")
        if min(self.fixed_delay_s, self.maximum_random_delay_s, self.long_tail_scale_s, self.long_tail_cap_s) < 0.0:
            raise ValueError("delay parameters must be nonnegative")
        if not 0.0 <= self.packet_loss_probability <= 1.0:
            raise ValueError("packet loss probability must lie in [0,1]")
        if not 0.0 <= self.burst_loss_probability <= 1.0 or self.burst_length < 0:
            raise ValueError("invalid burst loss parameters")


@dataclass(frozen=True)
class TransmissionResult:
    accepted_for_delivery: bool
    scheduled_receive_time_s: float | None
    reason: str


@dataclass(order=True)
class _Event:
    receive_time_s: float
    insertion_order: int
    packet: V2VPacket = field(compare=False)


class V2VChannel:
    def __init__(self, config: ChannelConfig) -> None:
        self.config = config
        self._rng = random.Random(config.random_seed)
        self._events: list[_Event] = []
        self._counter = 0
        self._burst_remaining_by_source: dict[int, int] = {}
        self.transmission_log: list[dict[str, object]] = []

    @property
    def pending_count(self) -> int:
        return len(self._events)

    def _delay_s(self) -> float:
        if self.config.delay_model == "zero":
            return 0.0
        if self.config.delay_model == "fixed":
            return self.config.fixed_delay_s
        if self.config.delay_model == "random_bounded":
            return self._rng.uniform(0.0, self.config.maximum_random_delay_s)
        if self.config.long_tail_scale_s == 0.0:
            return 0.0
        return min(self._rng.expovariate(1.0 / self.config.long_tail_scale_s), self.config.long_tail_cap_s)

    def _lost(self, packet: V2VPacket) -> tuple[bool, str]:
        remaining = self._burst_remaining_by_source.get(packet.source_vehicle_id, 0)
        forced_burst = packet.sequence_number in self.config.forced_burst_starts
        random_burst = self.config.burst_length > 0 and self._rng.random() < self.config.burst_loss_probability
        if forced_burst or random_burst:
            remaining = max(remaining, self.config.burst_length)
        if remaining > 0:
            self._burst_remaining_by_source[packet.source_vehicle_id] = remaining - 1
            return True, "burst_loss"
        if packet.sequence_number in self.config.forced_loss_sequences:
            return True, "forced_packet_loss"
        if self._rng.random() < self.config.packet_loss_probability:
            return True, "random_packet_loss"
        return False, "scheduled"

    def send(self, packet: V2VPacket, current_time_s: float) -> TransmissionResult:
        if abs(packet.generation_time_s - current_time_s) > 1e-9:
            raise ValueError("packet must enter the channel at its generation time")
        lost, reason = self._lost(packet)
        if lost:
            self.transmission_log.append({"packet": packet, "status": reason})
            return TransmissionResult(False, None, reason)
        receive = current_time_s + self._delay_s()
        delivered = packet.delivered_at(receive)
        heapq.heappush(self._events, _Event(receive, self._counter, delivered))
        self._counter += 1
        self.transmission_log.append({"packet": packet, "status": "scheduled", "receive_time_s": receive})
        return TransmissionResult(True, receive, "scheduled")

    def poll(self, current_time_s: float) -> tuple[V2VPacket, ...]:
        delivered: list[V2VPacket] = []
        while self._events and self._events[0].receive_time_s <= current_time_s + 1e-12:
            delivered.append(heapq.heappop(self._events).packet)
        return tuple(delivered)

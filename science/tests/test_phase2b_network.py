"""Event delay, loss, burst, age, order, consistency and reachable-message tests."""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from envs.observations import expand_predecessor_packet  # noqa: E402
from network.message_buffer import (  # noqa: E402
    DebugConsistencyTolerances,
    LocalMotionSensing,
    MessageBuffer,
)
from network.packet import V2VPacket  # noqa: E402
from network.v2v_channel import ChannelConfig, V2VChannel  # noqa: E402


def packet(sequence: int, generation: float = 0.0, position: float = 100.0) -> V2VPacket:
    return V2VPacket(0, 1, generation, sequence, "motion", position, 20.0, -1.0, 2.0)


channel = V2VChannel(ChannelConfig(delay_model="fixed", fixed_delay_s=0.2))
p = packet(0)
sent = channel.send(p, 0.0)
assert sent.accepted_for_delivery and sent.scheduled_receive_time_s == 0.2
assert channel.poll(0.19) == ()
delivered = channel.poll(0.2)
assert len(delivered) == 1
assert delivered[0].generation_time_s == 0.0
assert delivered[0].position_m == 100.0
assert delivered[0].receive_time_s == 0.2

loss = V2VChannel(ChannelConfig(forced_loss_sequences=frozenset({2})))
assert not loss.send(packet(2), 0.0).accepted_for_delivery
assert loss.poll(10.0) == ()

burst = V2VChannel(ChannelConfig(burst_length=2, forced_burst_starts=frozenset({4})))
assert not burst.send(packet(4), 0.0).accepted_for_delivery
assert not burst.send(packet(5), 0.0).accepted_for_delivery
assert burst.send(packet(6), 0.0).accepted_for_delivery

random_bounded = V2VChannel(ChannelConfig(delay_model="random_bounded", maximum_random_delay_s=0.3, random_seed=7))
scheduled = random_bounded.send(packet(7), 0.0)
assert 0.0 <= scheduled.scheduled_receive_time_s <= 0.3
long_tail = V2VChannel(ChannelConfig(delay_model="long_tail", long_tail_scale_s=0.2, long_tail_cap_s=0.5, random_seed=9))
scheduled = long_tail.send(packet(8), 0.0)
assert 0.0 <= scheduled.scheduled_receive_time_s <= 0.5

buffer = MessageBuffer(0.2, DebugConsistencyTolerances(2.0, 2.0))
fresh = packet(1).delivered_at(0.1)
decision = buffer.ingest(fresh, 0.1, LocalMotionSensing(101.995, 19.9))
assert decision.accepted and decision.age_s == 0.1
boundary = packet(2).delivered_at(0.2)
assert buffer.ingest(boundary, 0.2, LocalMotionSensing(103.98, 19.8)).accepted
over_age = packet(3).delivered_at(0.21)
assert buffer.ingest(over_age, 0.21, LocalMotionSensing(104.178, 19.79)).reason == "stale"

# A lower sequence arriving later is rejected without replacing the latest packet.
older = packet(1).delivered_at(0.15)
assert buffer.ingest(older, 0.15, LocalMotionSensing(102.989, 19.85)).reason == "out_of_order"
latest, reason, age = buffer.latest_valid(0, "motion", 0.2)
assert latest is not None and latest.sequence_number == 2 and reason == "fresh" and age == 0.2

inconsistent = packet(4, position=50.0).delivered_at(0.1)
reject = buffer.ingest(inconsistent, 0.1, LocalMotionSensing(102.0, 19.9))
assert not reject.accepted and "inconsistent" in reject.reason

reachable = expand_predecessor_packet(fresh, 0.1, 0.5)
assert reachable.speed_mps.low < reachable.speed_mps.high
assert reachable.acceleration_mps2.low < fresh.acceleration_mps2 < reachable.acceleration_mps2.high
assert reachable.position_m.low <= 101.995 <= reachable.position_m.high

try:
    V2VPacket(0, 1, 0.0, 99, "motion", 0.0, 0.0, 0.0, 0.0, paper_eligible=True)
except ValueError:
    pass
else:
    raise AssertionError("synthetic DEBUG packet cannot become paper eligible")

print("PASS: Phase 2B event channel, loss/burst, age/order, consistency, and reachability")

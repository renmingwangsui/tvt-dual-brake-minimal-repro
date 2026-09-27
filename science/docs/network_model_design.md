# Phase 2B V2V network model

## Packet and event semantics

`V2VPacket` contains source/target ids, generation and optional receive times, sequence number, historical position/speed/acceleration, declared jerk bound, certified reserve and critical vehicle/step/component metadata, requested deceleration and safety intent fields. Synthetic packets cannot be marked paper eligible.

`V2VChannel` is an actual event queue ordered by receive time and insertion order. `send` freezes the packet payload at generation, samples loss/delay, and schedules a delivered copy. `poll(t)` only releases due events. Delay is therefore not emulated by attaching an age to a current value: a delayed packet contains the historical state generated earlier.

Supported DEBUG channel modes are zero delay, fixed delay, bounded random delay and capped long-tail delay. Independent and burst losses are supported, with deterministic forced sequences for regression tests.

## Age, order and staleness

Packet age is `current_time - generation_time`. `MessageBuffer` accepts boundary-age packets and rejects packets strictly over the registered maximum age. It rejects sequence numbers no newer than the accepted packet for the same `(source, kind)`. Missing or stale safety messages invoke the registered conservative fallback; they are not treated as current truth.

## Local sensor consistency

Motion packets are projected from their historical generation state to the current time using their declared acceleration. The projected position and speed are checked against local range/range-rate sensing. Phase 2B tolerances are explicitly synthetic-debug values. Every decision logs acceptance or rejection and a reason such as stale, out-of-order, position-inconsistent or speed-inconsistent.

## Predecessor reachable information

`expand_predecessor_packet` expands historical motion with acceleration margin and jerk bound into intervals for position, speed and acceleration. These intervals feed the existing predictive reachability uncertainty object. Missing/stale packets create a conservative bounded-acceleration replacement based on current local sensing. True future predecessor trajectories are never inserted.

## Distributed bottleneck

Each controlled vehicle computes its own certified predictive reserve. A tail vehicle originates the recursive message; each predecessor calls the existing `aggregate_upstream_message` with only its local reserve and an actually received successor packet. Critical vehicle, prediction step, component and timestamp are preserved.

The centralized `min_i rho_i,H,cert` is computed after all local cycles for DEBUG diagnostics. It is absent from `LocalActorObservation` and is not passed to the decentralized safety controller. With zero delay and complete delivery, tail-to-head event delivery makes the head recursive value equal the centralized minimum. With delay, loss or staleness, equality is neither assumed nor logged as valid.

All network tests and closed-loop network runs are software validation only, not validation of a physical V2V channel.

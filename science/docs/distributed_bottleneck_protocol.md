# Distributed bottleneck protocol

Each truck forms its local certified predictive value and metadata. Starting at the downstream tail, truck `i` sends its permitted predecessor:

`(rho_upstream_H, critical_vehicle_id, critical_future_step, limiting_physical_constraint, timestamp)`.

The receiver computes the minimum between its own `rho_cert_H` and the received recursive value, propagating the metadata belonging to the minimum. With zero-delay complete delivery, the value at the head equals the centralized minimum. Under bounded delay it is an age-bounded upstream aggregate, not the instantaneous global fleet minimum.

Missing and stale packets are replaced by a pre-registered conservative communication-age fallback; they are never silently ignored. The implementation is `aggregate_upstream_message` in `src/safety/fleet_reserve.py`. Tests cover zero/bounded delay, packet loss, stale messages, critical-vehicle changes, and multi-hop argmin propagation.

`rho_fleet_centralized_H` is training/evaluation-only. `rho_upstream_local_H` is the deployment signal available from permitted messages.

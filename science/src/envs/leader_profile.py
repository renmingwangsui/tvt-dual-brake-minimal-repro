"""Causal DEBUG leader profiles with consistent position, speed and acceleration."""
from __future__ import annotations

from dataclasses import dataclass
from math import cos, pi, sin
from typing import Protocol


@dataclass(frozen=True)
class LeaderState:
    position_m: float
    speed_mps: float
    acceleration_mps2: float


class LeaderProfile(Protocol):
    jerk_bound_mps3: float
    def state_at(self, time_s: float) -> LeaderState: ...


@dataclass(frozen=True)
class ConstantSpeedLeader:
    initial_position_m: float
    speed_mps: float
    jerk_bound_mps3: float = 0.0

    def state_at(self, time_s: float) -> LeaderState:
        if time_s < 0.0:
            raise ValueError("time must be nonnegative")
        return LeaderState(self.initial_position_m + self.speed_mps * time_s, self.speed_mps, 0.0)


@dataclass(frozen=True)
class SmoothSpeedChangeLeader:
    initial_position_m: float
    initial_speed_mps: float
    peak_acceleration_mps2: float
    transition_duration_s: float

    @property
    def jerk_bound_mps3(self) -> float:
        return abs(self.peak_acceleration_mps2) * pi / (2.0 * self.transition_duration_s)

    def state_at(self, time_s: float) -> LeaderState:
        if time_s < 0.0 or self.transition_duration_s <= 0.0:
            raise ValueError("invalid leader profile time")
        duration = self.transition_duration_s
        acceleration = self.peak_acceleration_mps2
        active = min(time_s, duration)
        a = 0.5 * acceleration * (1.0 - cos(pi * active / duration)) if time_s < duration else 0.0
        v_change = 0.5 * acceleration * (active - duration / pi * sin(pi * active / duration))
        position_change = (
            0.25 * acceleration * active**2
            + acceleration * duration**2 / (2.0 * pi**2) * (cos(pi * active / duration) - 1.0)
        )
        position = self.initial_position_m + self.initial_speed_mps * active + position_change
        speed = max(0.0, self.initial_speed_mps + v_change)
        if time_s > duration:
            position += speed * (time_s - duration)
        return LeaderState(position, speed, a)


@dataclass(frozen=True)
class EmergencyBrakingPulseLeader:
    initial_position_m: float
    initial_speed_mps: float
    start_time_s: float
    pulse_duration_s: float
    peak_deceleration_mps2: float

    @property
    def jerk_bound_mps3(self) -> float:
        return self.peak_deceleration_mps2 * pi / self.pulse_duration_s

    def state_at(self, time_s: float) -> LeaderState:
        if time_s < 0.0 or self.start_time_s < 0.0 or self.pulse_duration_s <= 0.0 or self.peak_deceleration_mps2 <= 0.0:
            raise ValueError("invalid emergency pulse")
        if time_s <= self.start_time_s:
            return LeaderState(
                self.initial_position_m + self.initial_speed_mps * time_s,
                self.initial_speed_mps,
                0.0,
            )
        before_position = self.initial_position_m + self.initial_speed_mps * self.start_time_s
        elapsed = min(time_s - self.start_time_s, self.pulse_duration_s)
        duration = self.pulse_duration_s
        peak = self.peak_deceleration_mps2
        acceleration = -peak * sin(pi * elapsed / duration) ** 2 if elapsed < duration else 0.0
        speed_change = -0.5 * peak * (
            elapsed - duration / (2.0 * pi) * sin(2.0 * pi * elapsed / duration)
        )
        position_change = (
            self.initial_speed_mps * elapsed
            - 0.25 * peak * elapsed**2
            - peak * duration**2 / (8.0 * pi**2) * (cos(2.0 * pi * elapsed / duration) - 1.0)
        )
        speed = max(0.0, self.initial_speed_mps + speed_change)
        position = before_position + position_change
        if time_s > self.start_time_s + duration:
            position += speed * (time_s - self.start_time_s - duration)
        return LeaderState(position, speed, acceleration)

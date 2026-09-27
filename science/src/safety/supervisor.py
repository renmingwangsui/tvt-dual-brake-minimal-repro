"""Four-state anticipatory supervisor driven by predictive and hard reserves."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class SupervisorMode(str, Enum):
    NORMAL = "normal_actor"
    ANTICIPATORY = "anticipatory"
    CERTIFIED_BACKUP = "certified_backup"
    MINIMAL_RISK = "certificate_loss_minimal_risk"


@dataclass(frozen=True)
class SupervisorThresholds:
    predictive_trigger: float
    predictive_recovery: float
    hard_trigger: float
    hard_recovery: float
    recovery_samples: int

    def validate(self) -> None:
        if not self.predictive_trigger > self.hard_trigger >= 0.0:
            raise ValueError("require predictive_trigger > hard_trigger >= 0")
        if self.predictive_recovery <= self.predictive_trigger:
            raise ValueError("predictive recovery threshold must provide hysteresis")
        if self.hard_recovery <= self.hard_trigger:
            raise ValueError("hard recovery threshold must provide hysteresis")
        if self.recovery_samples < 1:
            raise ValueError("recovery_samples must be positive")


@dataclass(frozen=True)
class SupervisorDecision:
    mode: SupervisorMode
    anticipatory_triggered: bool
    hard_triggered: bool
    certificate_lost: bool
    recovery_counter: int
    action: str


@dataclass(frozen=True)
class BackupDomainConditions:
    h_collision_m: float
    h_temperature_K: float
    h_fade_N: float
    h_auxiliary_N: float
    psi1_collision_mps: float
    psi1_temperature_Kps: float
    rho_instantaneous: float
    speed_mps: float
    gear_dwell_valid: bool
    communication_uncertainty_valid: bool

    def contains_current_state(self) -> bool:
        return (
            min(
                self.h_collision_m,
                self.h_temperature_K,
                self.h_fade_N,
                self.h_auxiliary_N,
                self.psi1_collision_mps,
                self.psi1_temperature_Kps,
                self.rho_instantaneous,
                self.speed_mps,
            ) >= 0.0
            and self.gear_dwell_valid
            and self.communication_uncertainty_valid
        )


def predictive_supervisor_trigger(
    instantaneous_reserve: float,
    predictive_reserve: float,
    instantaneous_feasible: bool,
    hard_polytope_empty: bool,
    thresholds: SupervisorThresholds,
    previous_mode: SupervisorMode = SupervisorMode.NORMAL,
    recovery_counter: int = 0,
) -> SupervisorDecision:
    thresholds.validate()
    if hard_polytope_empty or not instantaneous_feasible:
        return SupervisorDecision(
            SupervisorMode.MINIMAL_RISK, True, True, True, 0,
            "maximum feasible auxiliary braking; thermal-capped friction; upstream stop request",
        )
    if instantaneous_reserve <= thresholds.hard_trigger:
        return SupervisorDecision(
            SupervisorMode.CERTIFIED_BACKUP, True, True, False, 0,
            "auxiliary-first certified allocation with minimum required friction",
        )
    if predictive_reserve <= thresholds.predictive_trigger:
        return SupervisorDecision(
            SupervisorMode.ANTICIPATORY, True, False, False, 0,
            "early auxiliary braking; headway expansion; upstream deceleration request",
        )
    recovering = previous_mode is not SupervisorMode.NORMAL
    recovery_ready = (
        predictive_reserve >= thresholds.predictive_recovery
        and instantaneous_reserve >= thresholds.hard_recovery
    )
    new_counter = recovery_counter + 1 if recovering and recovery_ready else 0
    if recovering and new_counter < thresholds.recovery_samples:
        return SupervisorDecision(previous_mode, False, False, False, new_counter, "hold recovery hysteresis")
    return SupervisorDecision(SupervisorMode.NORMAL, False, False, False, 0, "learned actor")


@dataclass(frozen=True)
class SafetyIntentMessage:
    predictive_reserve: float
    critical_step: int
    thermal_reserve_K: float | None = None
    requested_upstream_deceleration_mps2: float | None = None


def cooperative_mitigation_request(message: SafetyIntentMessage, trigger: float) -> dict[str, float | bool]:
    """Locally executable predecessor response to a compact successor message."""
    active = message.predictive_reserve <= trigger
    return {
        "active": active,
        "headway_increment_s": 0.5 if active else 0.0,
        "speed_bias_mps": -1.0 if active else 0.0,
        "requested_deceleration_mps2": (
            message.requested_upstream_deceleration_mps2 or (0.5 if active else 0.0)
        ),
    }

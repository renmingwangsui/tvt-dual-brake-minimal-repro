"""Rollout record for instantaneous, predictive, and fleet-level reserves."""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class SafetyRolloutRecord:
    time_s: float
    episode_id: int
    vehicle_id: int
    instantaneous_reserve: float
    predictive_reserve_vehicle: float
    fleet_predictive_reserve: float
    critical_vehicle_id: int
    critical_prediction_step: int
    friction_interval_nonempty: bool
    auxiliary_interval_nonempty: bool
    instantaneous_feasible: bool
    supervisor_mode: str
    predictive_trigger: bool
    hard_trigger: bool
    certificate_loss: bool
    delta_f_N: float = 0.0
    delta_a_N: float = 0.0
    collision_reserve_M_mps2: float = 0.0
    g_T0_K: float | None = None
    low_speed_thermal_feasible: bool = True
    critical_limit_type: str = "unspecified"
    tube_width_json: str = "{}"
    nominal_command_json: str = "[]"
    provisional_command_json: str = "[]"
    final_command_json: str = "[]"
    rho_ia_lower_by_step_json: str = "[]"
    rho_cert_H: float = 0.0
    rho_upstream_local_H: float = 0.0
    centralized_rho_fleet_H: float | None = None
    prediction_reverified: bool = False
    backup_control_widths_json: str = "[]"
    message_timestamp_s: float | None = None
    message_fallback_used: bool = False
    normalization_scale_id: str = "registered_physical"

    def to_row(self) -> dict[str, object]:
        return asdict(self)

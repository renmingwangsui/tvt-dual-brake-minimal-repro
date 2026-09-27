"""Dimensionally explicit helpers for literature-parameter calibration."""

from .validation import (
    cda,
    effective_heat_capacity,
    enforce_force_power_envelope,
    interpolate_road_profile,
    lb_to_kg,
    percent_grade_to_rad,
    torque_to_tire_force,
    validate_fade_curve,
    validate_positive_thermal,
    validate_temperature_domain,
    validate_time_constant,
    valid_uncertainty_box,
)

__all__ = [name for name in globals() if not name.startswith("_")]

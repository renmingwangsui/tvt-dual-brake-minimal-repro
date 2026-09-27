"""Reproduce Phase 2M-LIT-2 Reference_Truck conversions and bounds.

Only transcribed values in ``source_extracts.json`` are consumed.  The script
does not fit missing fade/actuator data. Its intermediate outputs remain
non-paper data until the Phase 2M-LIT-3 finalizer and 19-parameter gate pass.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "data" / "model_parameters" / "source_extracts.json"
OUTPUT = ROOT / "data" / "model_parameters" / "derived" / "reference_truck_parameters.json"

FT_TO_M = 0.3048
MPH_TO_MPS = 0.44704
BTU_TO_J = 1055.05585262
DEGF_TO_K = 5.0 / 9.0
SECONDS_PER_HOUR = 3600.0


def stopping_deceleration(speed_mph: float, distance_ft: float) -> float:
    """Return the constant-deceleration equivalent v^2/(2s) in m/s^2."""
    speed_mps = speed_mph * MPH_TO_MPS
    distance_m = distance_ft * FT_TO_M
    if speed_mps <= 0.0 or distance_m <= 0.0:
        raise ValueError("positive speed and stopping distance required")
    return speed_mps**2 / (2.0 * distance_m)


def btu_per_degf_to_j_per_k(value: float) -> float:
    return value * BTU_TO_J / DEGF_TO_K


def btu_per_hour_degf_to_w_per_k(value: float) -> float:
    return value * BTU_TO_J / SECONDS_PER_HOUR / DEGF_TO_K


def cooling_at_speed_w_per_k(k0: float, k1: float, speed_mps: float, count: int) -> float:
    speed_ft_per_s = speed_mps / FT_TO_M
    per_brake = k0 + k1 * speed_ft_per_s
    return count * btu_per_hour_degf_to_w_per_k(per_brake)


def main() -> int:
    raw = SOURCE.read_bytes()
    sources = json.loads(raw)["sources"]
    nhtsa = sources["NHTSA2010"]["extracts"]
    aashto = sources["AASHTO2004"]["extracts"]
    path = sources["YANAKIEV1998"]["extracts"]
    tamu = sources["VEGAMOOR2018"]["extracts"]

    mass_kg = nhtsa["gross_mass_kg"]["value"]
    speed_mph = nhtsa["stopping_test_speed_mph"]["value"]
    measured_a = stopping_deceleration(
        speed_mph, nhtsa["standard_scam_measured_stopping_distance_ft"]["value"]
    )
    model_a = stopping_deceleration(
        speed_mph, nhtsa["standard_scam_model_stopping_distance_ft"]["value"]
    )
    count = nhtsa["brake_assembly_count"]["value"]
    capacity = count * btu_per_degf_to_j_per_k(
        nhtsa["thermal_capacity_per_brake_btu_per_degF"]["value"]
    )
    k0 = nhtsa["cooling_k0_btu_per_h_degF"]["value"]
    k1 = nhtsa["cooling_k1_btu_s_per_h_degF_ft"]["value"]
    apply_tau = path["apply_time_constant_s"]["value"]

    result = {
        "artifact_type": "source-derived Reference_Truck closure quantities",
        "phase": "2M-LIT-2",
        "paper_eligible": False,
        "source_file": SOURCE.relative_to(ROOT).as_posix(),
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "derivations": {
            "vehicle_length_m": {
                "nominal": aashto["wb65_overall_length_ft"]["value"] * FT_TO_M,
                "lower": aashto["wb62_overall_length_ft"]["value"] * FT_TO_M,
                "upper": aashto["wb65_overall_length_ft"]["value"] * FT_TO_M,
                "equation": "m = ft * 0.3048",
            },
            "maximum_friction_braking_force_n": {
                "nominal": mass_kg * measured_a,
                "lower": mass_kg * measured_a,
                "upper": mass_kg * model_a,
                "measured_equivalent_deceleration_mps2": measured_a,
                "model_target_equivalent_deceleration_mps2": model_a,
                "equations": ["a_eq = v0^2/(2*s)", "F_max = m*a_eq"],
                "scope": "straight-line full S-cam braking at GVWR; constant-deceleration equivalent",
            },
            "lumped_thermal_capacity_j_per_k": {
                "nominal": capacity,
                "lower": capacity,
                "upper": capacity,
                "equation": "C_total = 10 * mc * 1055.05585262/(5/9)",
                "scope": "model-equivalent aggregation of ten wheel-end thermal states under one uniform repository state",
            },
            "cooling_coefficient_w_per_k": {
                "nominal_at_50_kmh": cooling_at_speed_w_per_k(k0, k1, 50.0 / 3.6, count),
                "lower_at_30_kmh": cooling_at_speed_w_per_k(k0, k1, 30.0 / 3.6, count),
                "upper_at_50_kmh": cooling_at_speed_w_per_k(k0, k1, 50.0 / 3.6, count),
                "equation": "k_total(v)=10*(k0+k1*v_ft_s)*1055.05585262/[3600*(5/9)]",
                "scope": "30-50 km/h literature benchmark envelope; not a statistical interval",
            },
            "predecessor_motion_bounds": {
                "acceleration_lower_mps2": -measured_a,
                "acceleration_upper_mps2": tamu["maximum_observed_acceleration_mps2"]["value"],
                "jerk_abs_upper_mps3": measured_a / apply_tau,
                "equation": "j_bound = a_emergency/tau_apply",
                "scope": "cross-source conservative reachable-set bound; jerk is derived, not measured",
            },
        },
        "unresolved_by_design": [
            "friction_command_slew",
            "auxiliary_actuator_lag",
            "auxiliary_command_slew",
            "auxiliary_speed_gear_power_envelope",
            "braking_to_thermal_conversion",
            "critical_brake_temperature",
            "thermal_fade_curve",
            "ambient_operating_envelope",
            "residual_heat_bound"
        ],
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print(OUTPUT.relative_to(ROOT).as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

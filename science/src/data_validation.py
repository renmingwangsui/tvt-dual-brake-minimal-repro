"""Strict production-data readiness validation.

The validator never supplies defaults and never infers missing values.  A full
simulation is permitted only when every declared requirement is present in a
production manifest and carries complete provenance metadata.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping


READY_FOR_FULL_SIMULATION = "READY_FOR_FULL_SIMULATION"
SCAFFOLD_ONLY_DATA_REQUIRED = "SCAFFOLD_ONLY_DATA_REQUIRED"
PROJECT_ROOT = Path(__file__).resolve().parents[1]

REQUIRED_METADATA = (
    "unit",
    "source",
    "vehicle_class",
    "provenance",
    "calibration_split",
    "uncertainty_source",
)

REQUIRED_ENTRIES: Mapping[str, tuple[str, ...]] = {
    "vehicle.mass": ("kg",),
    "vehicle.length": ("m",),
    "vehicle.payload_configuration": ("kg", "1"),
    "vehicle.rolling_resistance": ("1", "N"),
    "vehicle.CdA": ("m2", "m^2"),
    "friction_brake.maximum_force": ("N",),
    "friction_brake.actuator_lag": ("s",),
    "auxiliary_brake.actuator_lag": ("s",),
    "auxiliary_brake.force_torque_map": ("N", "N*m", "Nm"),
    "auxiliary_brake.power_map": ("W",),
    "auxiliary_brake.speed_dependence": ("m/s", "rad/s"),
    "auxiliary_brake.gear_dependence": ("1",),
    "auxiliary_brake.gear_dwell_shift_logic": ("s",),
    "thermal.capacity": ("J/K",),
    "thermal.cooling_coefficient": ("W/K",),
    "thermal.eta": ("1",),
    "thermal.ambient_range": ("K",),
    "thermal.critical_temperature": ("K",),
    "thermal.residual_heat_model_bounds": ("W", "W/s"),
    "fade.phi_T": ("1",),
    "fade.calibration_range": ("K",),
    "fade.uncertainty_bound": ("1",),
    "fade.validation_split": ("1",),
    "road.longitudinal_position": ("m",),
    "road.grade_theta": ("rad",),
    "road.coordinate_convention": ("1",),
    "road.grade_rate_bound": ("rad/m", "1/m"),
    "communication.delay_distribution_or_trace": ("s",),
    "communication.packet_loss_model_or_trace": ("1",),
    "communication.message_age_limit": ("s",),
    "communication.stale_message_policy": ("1",),
    "predecessor.emergency_deceleration_bound": ("m/s2", "m/s^2"),
    "predecessor.acceleration_bound": ("m/s2", "m/s^2"),
    "predecessor.jerk_bound": ("m/s3", "m/s^3"),
    "uncertainty.robust_margin_intervals": ("mixed",),
    "uncertainty.predictive_reachability_intervals": ("mixed",),
    "uncertainty.integration_error_bound": ("mixed",),
    "computation.controller_sampling_time": ("s",),
    "computation.simulator_internal_step": ("s",),
    "computation.target_hardware": ("1",),
}

POSITIVE_NUMERIC = {
    "vehicle.mass",
    "vehicle.length",
    "friction_brake.maximum_force",
    "friction_brake.actuator_lag",
    "auxiliary_brake.actuator_lag",
    "thermal.capacity",
    "thermal.cooling_coefficient",
    "thermal.critical_temperature",
    "communication.message_age_limit",
    "predecessor.emergency_deceleration_bound",
    "predecessor.acceleration_bound",
    "predecessor.jerk_bound",
    "computation.controller_sampling_time",
    "computation.simulator_internal_step",
}

PLACEHOLDERS = {
    "",
    "unknown",
    "todo",
    "tbd",
    "data_required",
    "author_decision",
    "synthetic_debug",
}


@dataclass(frozen=True)
class ReadinessIssue:
    requirement: str
    reason: str


def _placeholder(value: Any) -> bool:
    return isinstance(value, str) and value.strip().lower() in PLACEHOLDERS


def _entry_payload(entry: Mapping[str, Any]) -> tuple[str | None, Any]:
    present = [(key, entry.get(key)) for key in ("value", "map", "file") if key in entry]
    if len(present) != 1:
        return None, None
    return present[0]


def _hash_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _portable_path(path: Path) -> str:
    """Use repository-relative POSIX paths for repository-owned artifacts."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def _numeric_value(entry: Mapping[str, Any]) -> float | None:
    value = entry.get("value")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def validate_production_manifest(manifest_path: Path) -> dict[str, Any]:
    """Return an auditable readiness result without inventing defaults."""
    manifest_path = manifest_path.resolve()
    missing: list[ReadinessIssue] = []
    invalid: list[ReadinessIssue] = []
    checked_files: list[dict[str, str]] = []

    if not manifest_path.is_file():
        missing.extend(
            ReadinessIssue(name, "production manifest is absent")
            for name in REQUIRED_ENTRIES
        )
        return {
            "mode": SCAFFOLD_ONLY_DATA_REQUIRED,
            "manifest_path": _portable_path(manifest_path),
            "manifest_found": False,
            "required_entry_count": len(REQUIRED_ENTRIES),
            "validated_entry_count": 0,
            "missing": [asdict(item) for item in missing],
            "invalid": [],
            "checked_files": [],
        }

    try:
        document = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        invalid.append(ReadinessIssue("manifest", f"cannot parse JSON: {exc}"))
        document = {}

    entries = document.get("entries", {}) if isinstance(document, dict) else {}
    if not isinstance(entries, dict):
        invalid.append(ReadinessIssue("manifest.entries", "must be a JSON object"))
        entries = {}

    validated = 0
    for name, allowed_units in REQUIRED_ENTRIES.items():
        entry = entries.get(name)
        if not isinstance(entry, dict):
            missing.append(ReadinessIssue(name, "required entry is absent"))
            continue
        reasons: list[str] = []
        kind, payload = _entry_payload(entry)
        if kind is None:
            reasons.append("provide exactly one of value, map, or file")
        elif payload is None or _placeholder(payload):
            reasons.append("payload is empty or a placeholder")
        elif kind == "map" and not isinstance(payload, (dict, list)):
            reasons.append("map payload must be an object or array")

        for key in REQUIRED_METADATA:
            value = entry.get(key)
            if value is None or _placeholder(value):
                reasons.append(f"missing or placeholder metadata: {key}")

        unit = entry.get("unit")
        if isinstance(unit, str) and unit not in allowed_units:
            reasons.append(f"unit {unit!r} not in allowed SI units {allowed_units}")

        if kind == "file" and isinstance(payload, str) and not _placeholder(payload):
            referenced = (manifest_path.parent / payload).resolve()
            if not referenced.is_file():
                reasons.append(f"referenced file does not exist: {_portable_path(referenced)}")
            else:
                expected_hash = entry.get("sha256")
                actual_hash = _hash_file(referenced)
                if not isinstance(expected_hash, str) or expected_hash.lower() != actual_hash:
                    reasons.append("file sha256 is absent or does not match")
                checked_files.append({"path": _portable_path(referenced), "sha256": actual_hash})

        numeric = _numeric_value(entry)
        if name in POSITIVE_NUMERIC and numeric is not None and numeric <= 0.0:
            reasons.append("value must be strictly positive")
        if name == "thermal.eta" and numeric is not None and not (0.0 < numeric <= 1.0):
            reasons.append("eta must lie in (0, 1]")
        if name == "road.longitudinal_position" and kind == "map" and isinstance(payload, list):
            if not all(isinstance(x, (int, float)) for x in payload):
                reasons.append("position grid must be numeric")
            elif any(b <= a for a, b in zip(payload, payload[1:])):
                reasons.append("position grid must be strictly increasing")

        if reasons:
            invalid.extend(ReadinessIssue(name, reason) for reason in reasons)
        else:
            validated += 1

    control = entries.get("computation.controller_sampling_time", {})
    internal = entries.get("computation.simulator_internal_step", {})
    control_dt = _numeric_value(control) if isinstance(control, dict) else None
    internal_dt = _numeric_value(internal) if isinstance(internal, dict) else None
    if control_dt is not None and internal_dt is not None and internal_dt > control_dt:
        invalid.append(ReadinessIssue(
            "computation.simulator_internal_step",
            "internal integration step must not exceed controller sampling time",
        ))

    mode = READY_FOR_FULL_SIMULATION if not missing and not invalid else SCAFFOLD_ONLY_DATA_REQUIRED
    return {
        "mode": mode,
        "manifest_path": _portable_path(manifest_path),
        "manifest_found": True,
        "required_entry_count": len(REQUIRED_ENTRIES),
        "validated_entry_count": validated,
        "missing": [asdict(item) for item in missing],
        "invalid": [asdict(item) for item in invalid],
        "checked_files": checked_files,
    }

"""Strict fail-closed gate for the frozen Phase 2M-LIT-3 configuration."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "data" / "model_parameters" / "parameter_provenance.yaml"
SOURCES = ROOT / "data" / "model_parameters" / "source_extracts.json"
CONFIG = ROOT / "configs" / "model_simulation" / "literature_calibrated_v1.yaml"
ALLOWED = {"DIRECT", "DERIVED", "DIGITIZED", "IDENTIFIED", "ASSUMED", "DEBUG_ONLY"}
ELIGIBLE = {"DIRECT", "DERIVED", "DIGITIZED", "IDENTIFIED"}
EXPECTED_IDS = (
    "vehicle_mass", "vehicle_length", "rolling_resistance", "drag_area",
    "maximum_friction_braking_force", "friction_actuator_lag",
    "auxiliary_actuator_lag", "auxiliary_speed_gear_power_envelope",
    "lumped_thermal_capacity", "cooling_coefficient", "braking_to_thermal_conversion",
    "critical_brake_temperature", "thermal_fade_curve", "ambient_operating_envelope",
    "residual_heat_bound", "long_downhill_road_profile", "predecessor_emergency_bounds",
    "communication_delay_model", "communication_packet_burst_loss",
)
DERIVATION_OUTPUTS = {
    "drag_area": "data/model_parameters/derived/longitudinal_parameters.json",
    "maximum_friction_braking_force": "data/model_parameters/derived/reference_truck_parameters.json",
    "lumped_thermal_capacity": "data/model_parameters/derived/reference_truck_parameters.json",
    "cooling_coefficient": "data/model_parameters/derived/reference_truck_parameters.json",
    "long_downhill_road_profile": "data/model_parameters/derived/longitudinal_parameters.json",
    "predecessor_emergency_bounds": "data/model_parameters/derived/predecessor_bounds.json",
    "auxiliary_actuator_lag": "data/model_parameters/derived/phase2m_lit3_parameters.json",
    "braking_to_thermal_conversion": "data/model_parameters/derived/phase2m_lit3_parameters.json",
    "critical_brake_temperature": "data/model_parameters/derived/phase2m_lit3_parameters.json",
    "residual_heat_bound": "data/model_parameters/derived/thermal_model_discrepancy.json",
}
REQUIRED_FIELDS = (
    "parameter_id", "symbol", "vehicle_class", "nominal_value", "SI_unit",
    "uncertainty_lower", "uncertainty_upper", "uncertainty_status", "provenance_status",
    "source_title", "authors", "year", "DOI / manufacturer document ID",
    "page/table/figure/equation", "original_value", "original_unit", "conversion_to_SI",
    "derivation_script", "digitization_file", "identification_script", "cross_source",
    "physical_scope", "validation_scope", "paper_eligible", "notes", "resolved", "source_id",
)


def load_json_yaml(path: Path) -> dict[str, Any]:
    """The canonical YAML is emitted as JSON, a strict YAML 1.2 subset."""
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inspect() -> dict[str, Any]:
    reasons: list[str] = []
    if not REGISTRY.is_file():
        return {"ready": False, "reasons": ["parameter_registry_missing"]}
    if not SOURCES.is_file():
        return {"ready": False, "reasons": ["source_extract_registry_missing"]}
    try:
        registry = load_json_yaml(REGISTRY)
        sources = json.loads(SOURCES.read_text(encoding="utf-8"))["sources"]
    except (OSError, ValueError, KeyError) as exc:
        return {"ready": False, "reasons": [f"registry_parse_error:{exc}"]}

    parameters = registry.get("parameters", [])
    ids = [item.get("parameter_id") for item in parameters if isinstance(item, dict)]
    if len(parameters) != 19:
        reasons.append(f"required_parameter_count:{len(parameters)}")
    if len(set(ids)) != len(ids):
        reasons.append("duplicate_parameter_ids")
    if set(ids) != set(EXPECTED_IDS):
        reasons.append("required_parameter_set_mismatch")

    for item in parameters:
        pid = str(item.get("parameter_id", "unnamed"))
        missing_fields = [field for field in REQUIRED_FIELDS if field not in item]
        if missing_fields:
            reasons.append(f"{pid}:missing_fields:{','.join(missing_fields)}")
            continue
        status = item["provenance_status"]
        if status not in ALLOWED:
            reasons.append(f"{pid}:invalid_status")
        if item["resolved"] is not True:
            reasons.append(f"{pid}:unresolved")
        if status not in ELIGIBLE:
            reasons.append(f"{pid}:ineligible_status:{status}")
        if item["paper_eligible"] is not True:
            reasons.append(f"{pid}:paper_eligible_false")
        if item["nominal_value"] is None:
            reasons.append(f"{pid}:nominal_missing")
        if not item["SI_unit"]:
            reasons.append(f"{pid}:unit_missing")
        if item["uncertainty_lower"] is None or item["uncertainty_upper"] is None:
            reasons.append(f"{pid}:uncertainty_incomplete")
        source_id = item["source_id"]
        if not source_id or not item["source_title"] or not item["page/table/figure/equation"]:
            reasons.append(f"{pid}:source_traceability_incomplete")
        elif any(part not in sources for part in str(source_id).split("+")):
            reasons.append(f"{pid}:source_extract_missing")
        if status == "DERIVED":
            script = item["derivation_script"]
            if not script or not (ROOT / script).is_file():
                reasons.append(f"{pid}:derivation_artifact_missing")
            output = DERIVATION_OUTPUTS.get(pid)
            if not output or not (ROOT / output).is_file():
                reasons.append(f"{pid}:derived_output_missing")
        if status == "DIGITIZED":
            artifact = item["digitization_file"]
            if not artifact or not (ROOT / artifact).is_file():
                reasons.append(f"{pid}:digitization_artifact_missing")
        if status == "IDENTIFIED":
            script = item["identification_script"]
            if not script or not (ROOT / script).is_file():
                reasons.append(f"{pid}:identification_artifact_missing")

    if not CONFIG.is_file():
        reasons.append("frozen_literature_config_missing")
    else:
        try:
            config = load_json_yaml(CONFIG)
        except (OSError, ValueError) as exc:
            reasons.append(f"config_parse_error:{exc}")
        else:
            if config.get("configuration_frozen") is not True:
                reasons.append("configuration_not_frozen")
            if config.get("mode") != "LITERATURE_CALIBRATED":
                reasons.append("configuration_mode_invalid")
            if config.get("provenance_sha256") != sha256(REGISTRY):
                reasons.append("provenance_hash_mismatch")
            if config.get("source_extracts_sha256") != sha256(SOURCES):
                reasons.append("source_data_hash_mismatch")
            if not config.get("parameter_hashes"):
                reasons.append("parameter_hashes_missing")
            if not config.get("repository_state_id") or not config.get("repository_file_hashes"):
                reasons.append("repository_state_missing")
            partition = config.get("constraint_partition", {})
            if partition.get("friction_command_slew_hard_constraint") is not False:
                reasons.append("friction_slew_partition_invalid")
            if partition.get("auxiliary_command_slew_hard_constraint") is not False:
                reasons.append("auxiliary_slew_partition_invalid")
            if partition.get("friction_actuator_lag_retained") is not True or partition.get("auxiliary_actuator_lag_retained") is not True:
                reasons.append("actuator_lag_partition_invalid")

    return {
        "ready": not reasons,
        "required": 19,
        "counts": {status: sum(item.get("provenance_status") == status for item in parameters) for status in sorted(ALLOWED)},
        "unresolved": sum(item.get("resolved") is not True for item in parameters),
        "reasons": sorted(set(reasons)),
        "registry_sha256": sha256(REGISTRY),
        "source_extracts_sha256": sha256(SOURCES),
        "config_exists": CONFIG.is_file(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true", help="also print the machine-readable audit")
    args = parser.parse_args()
    report = inspect()
    status = "READY_FOR_PAPER_CANDIDATE_MODEL_SIMULATION" if report["ready"] else "LITERATURE_CALIBRATION_INCOMPLETE"
    print(status)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

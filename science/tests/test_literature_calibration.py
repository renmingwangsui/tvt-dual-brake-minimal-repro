"""Phase 2M-LIT-3 provenance, digitization, mapping, and fail-closed gate."""
from __future__ import annotations

import atexit
import hashlib
import json
from math import isclose
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# The legacy finalizer below refreshes the config's manuscript-source hashes.
# Keep this regression test read-only with respect to the frozen evidence input.
_frozen_config_path = ROOT / "configs/model_simulation/literature_calibrated_v1.yaml"
_frozen_config_bytes = _frozen_config_path.read_bytes()


def _restore_frozen_config() -> None:
    if _frozen_config_path.read_bytes() != _frozen_config_bytes:
        _frozen_config_path.write_bytes(_frozen_config_bytes)


atexit.register(_restore_frozen_config)

from calibration.validation import (  # noqa: E402
    cda, effective_heat_capacity, enforce_force_power_envelope,
    interpolate_road_profile, lb_to_kg, percent_grade_to_rad,
    torque_to_tire_force, validate_fade_curve, validate_positive_thermal,
    validate_temperature_domain, validate_time_constant, valid_uncertainty_box,
)

assert isclose(lb_to_kg(33237.0), 15076.04960169, rel_tol=1e-12)
assert isclose(cda(0.7963, 9.5), 7.56485, rel_tol=1e-12)
assert isclose(percent_grade_to_rad(-10.0), -0.09966865249116204, rel_tol=1e-12)
assert effective_heat_capacity([20.0, 20.0], [500.0, 500.0]) == 20_000.0
assert isclose(torque_to_tire_force(1000.0, 2.0, 3.0, 0.9, 0.5), 10_800.0)
assert validate_fade_curve([300.0, 400.0, 500.0], [1.0, 0.9, 0.7])
assert validate_temperature_domain([280.0, 400.0, 700.0], 300.0, 650.0)
assert validate_positive_thermal(1.0, 1.0)
assert validate_time_constant(0.25)
assert enforce_force_power_envelope(10_000.0, 20.0, 12_000.0, 250_000.0)
assert isclose(interpolate_road_profile([0.0, 100.0], [-0.1, -0.05], 50.0), -0.075)
assert valid_uncertainty_box([0.0, -1.0], [0.5, 0.0], [1.0, 1.0])

for script in (
    "scripts/calibration/derive_longitudinal_parameters.py",
    "scripts/calibration/derive_reference_truck_parameters.py",
    "scripts/calibration/derive_predecessor_bounds.py",
    "scripts/calibration/build_uncertainty_intervals.py",
    "scripts/calibration/finalize_phase2m_lit3.py",
):
    completed = subprocess.run([sys.executable, str(ROOT / script)], cwd=ROOT, capture_output=True, text=True, check=False)
    assert completed.returncode == 0, (script, completed.stdout, completed.stderr)

registry_path = ROOT / "data/model_parameters/parameter_provenance.yaml"
registry = json.loads(registry_path.read_text(encoding="utf-8"))
parameters = registry["parameters"]
assert len(parameters) == 19 == registry["required_parameter_count"]
assert len({item["parameter_id"] for item in parameters}) == 19
assert all(item["resolved"] is True and item["paper_eligible"] is True for item in parameters)
assert sum(item["provenance_status"] == "DIRECT" for item in parameters) == 7
assert sum(item["provenance_status"] == "DERIVED" for item in parameters) == 10
assert sum(item["provenance_status"] == "DIGITIZED" for item in parameters) == 2
assert not {"friction_command_slew", "auxiliary_command_slew"} & {item["parameter_id"] for item in parameters}

by_id = {item["parameter_id"]: item for item in parameters}
assert isclose(by_id["vehicle_mass"]["nominal_value"], 33200.0)
assert isclose(by_id["maximum_friction_braking_force"]["nominal_value"], 123603.03444794951)
assert isclose(by_id["friction_actuator_lag"]["nominal_value"], 0.14)
assert isclose(by_id["auxiliary_actuator_lag"]["nominal_value"], 0.25)
assert isclose(by_id["braking_to_thermal_conversion"]["nominal_value"], 1.0)
assert isclose(by_id["ambient_operating_envelope"]["nominal_value"], 298.15)
assert isclose(by_id["critical_brake_temperature"]["nominal_value"], 463.708161407693, rel_tol=1e-12)

digitization_path = ROOT / "data/model_parameters/source_points/phase2m_lit3_digitization.json"
digitization = json.loads(digitization_path.read_text(encoding="utf-8"))
fade = digitization["fade"]
curve = fade["selected_conservative_curve"]
assert fade["label"] == "DIGITIZED_REPRESENTATIVE_HEAVY_TRUCK_S_CAM_FADE"
assert len(curve) == 8 and all(a["phi"] >= b["phi"] for a, b in zip(curve, curve[1:]))
assert fade["axis_calibration"]["pixel_uncertainty"] == {"x_px": 2.0, "y_px": 2.0}
retarder = digitization["retarder"]
assert len(retarder["raw_digitized_points"]) == 11
assert retarder["power_limits_W"] == {"continuous": 300000.0, "peak": 450000.0}
for point in retarder["raw_digitized_points"]:
    assert point["continuous_torque_envelope_Nm"] <= point["measured_300kPa_torque_Nm"] + 1e-9
    assert point["peak_torque_envelope_Nm"] <= point["measured_300kPa_torque_Nm"] + 1e-9
    expected_force = point["continuous_torque_envelope_Nm"] * 5.7 * 0.97 / 0.492
    assert isclose(point["continuous_longitudinal_force_N"], expected_force, rel_tol=1e-12)

residual_path = ROOT / "data/model_parameters/derived/thermal_model_discrepancy.json"
residual = json.loads(residual_path.read_text(encoding="utf-8"))
assert residual["label"] == "DERIVED_MODEL_DISCREPANCY_BOUND"
assert len(residual["samples"]) == 40
assert isclose(residual["maximum_absolute_residual_W"], 92551.5731662068, rel_tol=1e-12)
assert by_id["residual_heat_bound"]["uncertainty_lower"] == -residual["maximum_absolute_residual_W"]
assert by_id["residual_heat_bound"]["uncertainty_upper"] == residual["maximum_absolute_residual_W"]

config_path = ROOT / "configs/model_simulation/literature_calibrated_v1.yaml"
config = json.loads(config_path.read_text(encoding="utf-8"))
assert config["configuration_frozen"] is True
assert config["paper_eligible_requested"] is False
assert config["paper_eligible_results_generated"] is False
assert config["provenance_sha256"] == hashlib.sha256(registry_path.read_bytes()).hexdigest()
assert config["constraint_partition"]["friction_command_slew_hard_constraint"] is False
assert config["constraint_partition"]["auxiliary_command_slew_hard_constraint"] is False

gate = subprocess.run([sys.executable, str(ROOT / "scripts/check_literature_calibrated_readiness.py")], cwd=ROOT, capture_output=True, text=True, check=False)
assert gate.returncode == 0, (gate.stdout, gate.stderr)
assert gate.stdout.strip() == "READY_FOR_PAPER_CANDIDATE_MODEL_SIMULATION"

controller = (ROOT / "src/controllers/integrated_safety_controller.py").read_text(encoding="utf-8")
assert '"friction_rate_lower"' not in controller and '"auxiliary_rate_lower"' not in controller
print("PASS: Phase 2M-LIT-3 final 19-parameter provenance closure and frozen gate")

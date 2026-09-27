"""Reproduce heavy-truck predecessor bounds for the Reference_Truck audit."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


FT_TO_M = 0.3048
MPH_TO_MPS = 0.44704


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "data" / "model_parameters" / "source_extracts.json"
OUTPUT = ROOT / "data" / "model_parameters" / "derived" / "predecessor_bounds.json"


def main() -> int:
    raw = SOURCE.read_bytes()
    sources = json.loads(raw)["sources"]
    nhtsa = sources["NHTSA2010"]["extracts"]
    tamu = sources["VEGAMOOR2018"]["extracts"]
    path = sources["YANAKIEV1998"]["extracts"]
    speed_mps = nhtsa["stopping_test_speed_mph"]["value"] * MPH_TO_MPS
    distance_m = nhtsa["standard_scam_measured_stopping_distance_ft"]["value"] * FT_TO_M
    a_min = -(speed_mps**2 / (2.0 * distance_m))
    a_max = tamu["maximum_observed_acceleration_mps2"]["value"]
    tau = path["apply_time_constant_s"]["value"]
    result = {
        "artifact_type": "cross-source heavy-truck reachable-set bound",
        "paper_eligible": False,
        "source_file": SOURCE.relative_to(ROOT).as_posix(),
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "acceleration_lower_mps2": a_min,
        "acceleration_upper_mps2": a_max,
        "jerk_abs_upper_mps3": abs(a_min) / tau,
        "equation": "j_scale = abs(a_min) / tau_f",
        "limitation": "Emergency deceleration is a constant-deceleration equivalent and jerk is a derived conservative scale, not a measured jerk trace.",
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print(OUTPUT.relative_to(ROOT).as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

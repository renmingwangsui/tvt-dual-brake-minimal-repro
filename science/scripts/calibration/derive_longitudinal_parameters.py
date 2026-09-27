"""Reproduce only longitudinal/road quantities supported by source extracts."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "data" / "model_parameters" / "source_extracts.json"
OUTPUT = ROOT / "data" / "model_parameters" / "derived" / "longitudinal_parameters.json"


def main() -> int:
    raw = SOURCE.read_bytes()
    sources = json.loads(raw)["sources"]
    nrel = sources["WANG2015"]["extracts"]
    road = sources["DEVIKA2022"]["extracts"]
    result = {
        "artifact_type": "source-derived calibration quantity",
        "source_file": SOURCE.relative_to(ROOT).as_posix(),
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "derivations": {
            "wang_baseline_weight_kg_not_selected_as_reference_truck_mass": {
                "value": nrel["baseline_weight_lb"]["value"] * 0.45359237,
                "equation": "kg = lb * 0.45359237",
            },
            "drag_area_m2": {
                "value": nrel["drag_coefficient"]["value"] * nrel["frontal_area_m2"]["value"],
                "equation": "CdA = Cd * A",
            },
            "road_profile": {
                "position_m": [0.0, road["road_length_m"]["value"]],
                "grade_rad": [math.atan(road["road_grade_percent"]["value"] / 100.0)] * 2,
                "equation": "theta = atan(grade_percent/100)",
                "scope": "literature-defined constant-grade benchmark; not a surveyed road",
            },
        },
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print(OUTPUT.relative_to(ROOT).as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

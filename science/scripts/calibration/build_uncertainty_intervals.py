"""Register what the selected sources do and do not support about uncertainty."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "data" / "model_parameters" / "source_extracts.json"
REGISTRY = ROOT / "data" / "model_parameters" / "parameter_provenance.yaml"
OUTPUT = ROOT / "data" / "model_parameters" / "derived" / "uncertainty_intervals.json"


def main() -> int:
    raw = SOURCE.read_bytes()
    registry_raw = REGISTRY.read_bytes()
    parameters = json.loads(registry_raw)["parameters"]
    resolved = [item for item in parameters if item["resolved"] is True]
    result = {
        "artifact_type": "uncertainty evidence audit",
        "phase": "2M-LIT-2",
        "source_file": SOURCE.relative_to(ROOT).as_posix(),
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "registry_file": REGISTRY.relative_to(ROOT).as_posix(),
        "registry_sha256": hashlib.sha256(registry_raw).hexdigest(),
        "records": {
            item["parameter_id"]: {
                "nominal": item["nominal_value"],
                "lower": item["uncertainty_lower"],
                "upper": item["uncertainty_upper"],
                "uncertainty_status": item["uncertainty_status"],
                "source_id": item["source_id"],
                "paper_eligible": False,
            }
            for item in resolved
        },
        "resolved_record_count": len(resolved),
        "unresolved_parameter_ids": [item["parameter_id"] for item in parameters if item["resolved"] is not True],
        "global_status": "INCOMPLETE",
        "note": "Literature ranges, derived envelopes, and sensitivity-only intervals remain distinct; no uniform plus/minus percentage is introduced.",
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print(OUTPUT.relative_to(ROOT).as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Execute the strict data-availability gate and write audit artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data_validation import (  # noqa: E402
    READY_FOR_FULL_SIMULATION,
    validate_production_manifest,
)


def _markdown(report: dict[str, object]) -> str:
    missing = report["missing"]
    invalid = report["invalid"]
    lines = [
        "# Simulation readiness report",
        "",
        f"**READINESS MODE: `{report['mode']}`**",
        "",
        "This is a strict evidence gate. It does not infer defaults, fill parameters, or accept synthetic-debug data as production evidence.",
        "",
        "## Manifest",
        "",
        f"- Expected production manifest: `{report['manifest_path']}`",
        f"- Manifest found: `{str(report['manifest_found']).lower()}`",
        f"- Required entries: `{report['required_entry_count']}`",
        f"- Fully validated entries: `{report['validated_entry_count']}`",
        "",
        "## Repository capability finding",
        "",
        "Reference implementations exist for the safety core, instantaneous/predictive certificates, interval reachability, set-valued affine backup, distributed bottleneck aggregation, two-pass callback order, supervisor logic, actor parameterization and scalar learning losses.",
        "",
        "The Phase 2A continuous simulator, Phase 2B heterogeneous platoon/V2V/safety integration, and Phase 2C-1 NumPy MAPPO core exist and pass synthetic-debug software tests, but are not physically or network calibrated and are not paper eligible. Differentiable-QP backward propagation, matched baselines, immutable experiment logger, production experiment manager, statistics pipeline, generated-table pipeline, paper-eligible checkpoints and claim-bearing logs remain absent. The reference safety core includes the explicit low-speed state-only `g_T0` gate. See `docs/repository_inventory.md`.",
        "",
        "## Missing requirements",
        "",
    ]
    if missing:
        lines.extend(
            f"- `{item['requirement']}` — {item['reason']}" for item in missing
        )
    else:
        lines.append("None.")
    lines.extend(["", "## Invalid requirements", ""])
    if invalid:
        lines.extend(
            f"- `{item['requirement']}` — {item['reason']}" for item in invalid
        )
    else:
        lines.append("None.")
    lines.extend([
        "",
        "## Decision",
        "",
        "Production simulation/training, publication figures, numerical result tables, and manuscript result updates are prohibited unless this gate returns `READY_FOR_FULL_SIMULATION`. Explicitly marked synthetic DEBUG software-validation smoke runs remain non-paper-eligible.",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "data" / "production" / "physical_manifest.json",
    )
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()

    report = validate_production_manifest(args.manifest)
    if not args.no_write:
        artifacts = ROOT / "artifacts"
        docs = ROOT / "docs"
        artifacts.mkdir(parents=True, exist_ok=True)
        docs.mkdir(parents=True, exist_ok=True)
        (artifacts / "readiness_report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        (docs / "readiness_report.md").write_text(_markdown(report), encoding="utf-8")

    print(report["mode"])
    print(f"validated={report['validated_entry_count']}/{report['required_entry_count']}")
    print(f"missing={len(report['missing'])} invalid={len(report['invalid'])}")
    return 0 if report["mode"] == READY_FOR_FULL_SIMULATION else 2


if __name__ == "__main__":
    raise SystemExit(main())

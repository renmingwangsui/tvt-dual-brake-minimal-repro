"""Generate manuscript-facing table from the immutable Zhou comparison."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results/external_baselines/zhou_tits_2025/formal_v1"
MANIFEST = RESULT_ROOT / "result_manifest.json"
SUMMARY = RESULT_ROOT / "comparison_summary.json"
OUTPUT = ROOT / "generated/final/tables/table_external_baseline.tex"
EXPECTED_MANIFEST_SHA256 = "5609dd929e100711de562f0e1b1b9c4de013f9bae12c6f4b97fd41dd3a35c357"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reserve(value: float) -> str:
    if abs(value) >= 10_000:
        exponent = len(str(int(abs(value)))) - 1
        coefficient = value / (10**exponent)
        return f"${coefficient:.2f}\\times10^{{{exponent}}}$"
    return f"{value:.3f}"


def main() -> int:
    if sha256(MANIFEST) != EXPECTED_MANIFEST_SHA256:
        raise RuntimeError("frozen external-baseline manifest hash mismatch")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for relative, expected in manifest["file_hashes"].items():
        if sha256(RESULT_ROOT / relative) != expected:
            raise RuntimeError(f"frozen external-baseline file hash mismatch: {relative}")

    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    runs = {(item["scenario"], item["method"]): item["metrics"] for item in summary["runs"]}
    scenarios = (
        ("nominal_long_descent", "Nominal descent"),
        ("hot_start_413.7081614076934K", "Hot 413.708 K"),
        ("hot_start_453.7081614076934K", "Hot 453.708 K"),
        ("hot_start_461.7081614076934K", "Hot 461.708 K"),
        ("emergency_brake_during_descent", "Emergency descent"),
        ("communication_nominal_delay", "Nominal delay"),
    )
    methods = (("accepted_method", "Proposed"), ("adapted_zhou", "Adapted Zhou"))
    lines = [
        "\\begin{tabular}{llrrrrrr}",
        "\\toprule",
        "Scenario & Method & Coll. & Peak $T$ (K) & Fade & $\\min\\rho^{H,\\mathrm{cert}}$ & Speed RMSE & P99 (ms) \\\\",
        "\\midrule",
    ]
    for scenario_id, scenario_label in scenarios:
        for method_id, method_label in methods:
            metrics = runs[(scenario_id, method_id)]
            lines.append(
                f"{scenario_label} & {method_label} & {int(metrics['collision_count'])} & "
                f"{metrics['peak_brake_temperature_K']:.2f} & "
                f"{int(metrics['fade_envelope_violation_count'])} & "
                f"{reserve(metrics['min_rho_H_cert'])} & "
                f"{metrics['speed_rmse_mps']:.3f} & "
                f"{metrics['controller_runtime_p99_ms']:.3f} \\\\" 
            )
        if scenario_id != scenarios[-1][0]:
            lines.append("\\addlinespace[1pt]")
    lines.extend(("\\bottomrule", "\\end{tabular}"))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"PASS: {OUTPUT.relative_to(ROOT)} sha256={sha256(OUTPUT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

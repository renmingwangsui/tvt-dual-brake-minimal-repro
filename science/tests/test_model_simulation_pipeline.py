"""Reduced deterministic integration test for the Phase 2M M1--M10 pipeline."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from analysis.model_based.analyze import analyze_model_simulations  # noqa: E402
from analysis.model_based.figures import generate_debug_figures  # noqa: E402
from experiments.model_based.pipeline import DEFAULT_CONFIG, DEFAULT_PARAMETERS, run_debug_suite  # noqa: E402
from experiments.model_based.provenance import evaluate_paper_eligibility  # noqa: E402


base = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
config = deepcopy(base)
config["steps"].update({"M2": 2, "M3": 1, "M4": 2, "M5": 1, "M6": 1, "M7": 2, "M9": 1, "M10_repetitions": 1})
config["M1"] = {
    "speed_mps": [5.0], "temperature_K": [350.0], "grade_rad": [-0.03],
    "mass_scale": [1.0], "auxiliary_availability": [1.0],
}
config["M3"]["initial_temperature_K"] = [350.0, 642.0]
config["M5"]["horizons"] = [1, 3]
config["M6"]["joint_monte_carlo_samples"] = 1
config["M9"] = {"platoon_sizes": [3], "disturbances": ["smooth_braking_pulse"]}

with TemporaryDirectory() as temporary:
    temporary_path = Path(temporary)
    config_path = temporary_path / "config.json"
    output = temporary_path / "output"
    figures = temporary_path / "figures"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    manifest = run_debug_suite(config_path, DEFAULT_PARAMETERS, output)
    assert manifest["mode"] == "DEBUG_SYNTHETIC"
    assert manifest["paper_eligible"] is False
    assert [item["experiment_id"] for item in manifest["experiments"]] == [f"M{i}" for i in range(1, 11)]
    for experiment in (f"m{i}" for i in range(1, 11)):
        path = output / f"{experiment}_records.jsonl"
        assert path.is_file() and path.stat().st_size > 0
        first = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
        assert first["paper_eligible"] is False
        assert first["data_provenance"] == "synthetic_debug"
        assert len(first["config_hash"]) == 64 and len(first["source_provenance_hash"]) == 64

    m8 = [json.loads(line) for line in (output / "m8_records.jsonl").read_text(encoding="utf-8").splitlines()]
    assert {row["speed_mps"] for row in m8} == {0.0, 0.1, 0.5, 0.6}
    assert {row["initial_temperature_K"] for row in m8} == {350.0, 652.0}
    assert all(row["g_T0_K"] is not None for row in m8 if row["speed_mps"] <= 0.5)
    assert all(row["g_T0_K"] is None for row in m8 if row["speed_mps"] > 0.5)
    assert any(
        row["low_speed_thermal_feasible"] is False
        for row in m8
        if row["initial_temperature_K"] == 652.0 and row["speed_mps"] <= 0.5
    )

    m2 = [json.loads(line) for line in (output / "m2_records.jsonl").read_text(encoding="utf-8").splitlines()]
    assert {row["method"] for row in m2} == {"friction_only", "friction_first", "auxiliary_first", "hard_qp_dual"}
    assert all("friction_upper_bound_N" in row and "collision_demand_mps2" in row for row in m2)
    m5 = [json.loads(line) for line in (output / "m5_records.jsonl").read_text(encoding="utf-8").splitlines()]
    assert all(row["stored_prefix_monotonicity_verified"] for row in m5)
    m7 = [json.loads(line) for line in (output / "m7_records.jsonl").read_text(encoding="utf-8").splitlines()]
    assert all("missed_anticipatory_trigger" in row and "false_anticipatory_trigger" in row for row in m7)

    m10 = [json.loads(line) for line in (output / "m10_records.jsonl").read_text(encoding="utf-8").splitlines()]
    timing_keys = {"hard_row_construction_s", "qp_solve_s", "predictive_tube_s", "distributed_bottleneck_s", "supervisor_s", "reverification_s", "total_controller_s"}
    assert timing_keys.issubset(m10[0])
    assert all(m10[0][key] >= 0.0 for key in timing_keys)

    summary = analyze_model_simulations(output)
    assert summary["paper_eligible"] is False
    assert set(summary["experiments"]) == {f"M{i}" for i in range(1, 11)}
    generated = generate_debug_figures(output, figures)
    assert len(generated) == 8
    assert all("SYNTHETIC DEBUG" in path.read_text(encoding="utf-8") for path in generated)

registry = json.loads(DEFAULT_PARAMETERS.read_text(encoding="utf-8"))
decision = evaluate_paper_eligibility(base, registry, {"paper_eligible": False})
assert not decision.eligible and decision.reasons

print("PASS: Phase 2M reduced M1-M10 pipeline/logging/analysis/figures/provenance gate")

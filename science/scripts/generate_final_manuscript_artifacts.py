"""Generate final-manuscript tables, macros, figures, and provenance.

Every quantitative value is read from immutable Phase 2M / Phase 2C-3
artifacts.  This script never writes into the frozen evidence directories.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "generated" / "final"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
MODEL = ROOT / "results" / "paper_candidate" / "phase2m-paper-d10d13eb-6634280c" / "analysis_summary.json"
FOLLOWUP = ROOT / "results" / "paper_candidate" / "phase2m-followup-d10d13eb-912a1958" / "analysis_summary.json"
LEARNING = ROOT / "results" / "formal_learning" / "paired_statistics.json"
LEARNING_MANIFEST = ROOT / "results" / "formal_learning" / "result_manifest.json"
PROVENANCE = ROOT / "data" / "model_parameters" / "parameter_provenance.yaml"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fmt(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}g}"


def write(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8", newline="\n")


def main() -> int:
    model = json.loads(MODEL.read_text(encoding="utf-8"))["experiments"]
    follow = json.loads(FOLLOWUP.read_text(encoding="utf-8"))
    learning = json.loads(LEARNING.read_text(encoding="utf-8"))
    lman = json.loads(LEARNING_MANIFEST.read_text(encoding="utf-8"))
    provenance = json.loads(PROVENANCE.read_text(encoding="utf-8"))

    selected = {
        "vehicle_mass", "maximum_friction_braking_force", "friction_actuator_lag",
        "auxiliary_actuator_lag", "lumped_thermal_capacity", "cooling_coefficient",
        "critical_brake_temperature", "ambient_operating_envelope", "residual_heat_bound",
    }
    labels = {
        "vehicle_mass": "Vehicle mass",
        "maximum_friction_braking_force": "Maximum friction force",
        "friction_actuator_lag": "Friction actuator lag",
        "auxiliary_actuator_lag": "Auxiliary actuator lag",
        "lumped_thermal_capacity": "Lumped thermal capacity",
        "cooling_coefficient": "Cooling coefficient",
        "critical_brake_temperature": "Critical brake temperature",
        "ambient_operating_envelope": "Ambient temperature",
        "residual_heat_bound": "Nominal residual heat",
    }
    symbols = {
        "vehicle_mass": r"m", "maximum_friction_braking_force": r"\bar u^f",
        "friction_actuator_lag": r"\tau_f", "auxiliary_actuator_lag": r"\tau_a",
        "lumped_thermal_capacity": r"C", "cooling_coefficient": r"k",
        "critical_brake_temperature": r"T_{\rm crit}",
        "ambient_operating_envelope": r"T_a", "residual_heat_bound": r"q_w",
    }
    parameter_rows = []
    for item in provenance["parameters"]:
        if item["parameter_id"] not in selected:
            continue
        nominal = item["nominal_value"]
        if isinstance(nominal, (int, float)):
            nominal = fmt(float(nominal), 6)
        parameter_rows.append((labels[item["parameter_id"]], symbols[item["parameter_id"]], str(nominal), item["SI_unit"], item["provenance_status"]))
    body = "\n".join(f"{a} & ${b}$ & {c} & {d} & {e} \\\\" for a, b, c, d, e in parameter_rows)
    write(TABLES / "table_final_parameters.tex", rf"""\begin{{table*}}[!t]
\centering\scriptsize
\caption{{Selected literature-calibrated parameters; full intervals and source-level provenance are archived with the configuration.}}
\label{{tab:final-parameters}}
\begin{{tabular}}{{lllll}}
\toprule
Parameter & Symbol & Nominal & SI unit & Provenance \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table*}}
""")

    m1, m3 = model["M1"], model["M3"]
    model_rows = [
        ("M1", "Dual/friction feasible points", f'{m1["dual_brake_feasible_count"]}/{m1["records"]}; {m1["friction_only_feasible_count"]}/{m1["records"]}', "No feasible-count increase"),
        ("M1", "Positive dual-reserve gain", f'{m1["positive_margin_gain_count"]}/{m1["records"]}', "Margin improvement only"),
    ]
    for temperature, row in sorted(m3.items(), key=lambda item: float(item[0])):
        model_rows.append(("M3", f'Certificate loss at $T_0={fmt(float(temperature))}$ K', f'{fmt(row["first_hard_certificate_loss_time_s"])} s', row["first_critical_component"].replace("_", r"\_")))
    body = "\n".join(f"{a} & {b} & {c} & {d} \\\\" for a, b, c, d in model_rows)
    write(TABLES / "table_final_model_evidence.tex", rf"""\begin{{table*}}[!t]
\centering\scriptsize
\caption{{Selected model-based evidence from the Phase 2M analysis.}}
\label{{tab:final-model-evidence}}
\begin{{tabular}}{{llll}}
\toprule
Study & Quantity & Result & Interpretation \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table*}}
""")

    pw = follow["predictive_warning"]
    m7, m8, m9 = follow["M7"], follow["M8"], follow["M9"]
    macros = {
        "FinalWarningScenarios": pw["scenarios"],
        "FinalTrueWarnings": pw["classification_counts"]["TRUE_EARLY_WARNING"],
        "FinalConservativeWarnings": pw["classification_counts"]["CONSERVATIVE_WARNING_WITHOUT_BOUNDARY"],
        "FinalSimultaneousWarnings": pw["classification_counts"]["SIMULTANEOUS_WARNING"],
        "FinalRobustWarnings": pw["non_isolated_non_numerical_true_early_warning_count"],
        "FinalMSevenMisses": m7["missed_trigger_count"],
        "FinalRhoJump": fmt(m8["hot_rho_jump"], 7),
        "FinalGTwo": fmt(m9["max_G2"], 6),
        "FinalGInf": fmt(m9["max_Ginf"], 6),
        "FinalGradientValid": fmt(lman["diffqp_gradient_valid_fraction"]),
        "FinalFallback": fmt(lman["diffqp_fallback_fraction"]),
    }
    write(TABLES / "final_claim_macros.tex", "\n".join(rf"\newcommand{{\{key}}}{{{value}}}" for key, value in macros.items()) + "\n")

    metrics = [
        ("Final return", "final_evaluation_return"),
        ("Return AUC (sample efficiency)", "evaluation_return_auc"),
        ("QP intervention rate", "final_qp_intervention_frequency"),
        ("QP intervention norm (N)", "final_qp_intervention_norm"),
        ("Nominal hard-feasible rate", "final_nominal_hard_feasible_rate"),
    ]
    rows = []
    for label, key in metrics:
        row = learning[key]
        lo, hi = row["paired_bootstrap_95_ci"]
        rows.append((label, f'{fmt(row["diffqp_mean"], 6)} ({fmt(row["diffqp_std"], 5)})', f'{fmt(row["stopgrad_mean"], 6)} ({fmt(row["stopgrad_std"], 5)})', fmt(row["paired_mean"], 6), f'[{fmt(lo, 4)}, {fmt(hi, 4)}]'))
    body = "\n".join(f"{a} & {b} & {c} & {d} & {e} \\\\" for a, b, c, d, e in rows)
    write(TABLES / "table_final_learning_evidence.tex", rf"""\begin{{table*}}[!t]
\centering\scriptsize
\caption{{Ten-paired-seed formal learning evidence. Equality in this campaign reflects the registered zero-gradient fallback on every DiffQP sample, not general method equivalence.}}
\label{{tab:final-learning-evidence}}
\begin{{tabular}}{{lrrrr}}
\toprule
Metric & DiffQP mean (SD) & StopGrad mean (SD) & Paired difference & 95\% CI \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table*}}
""")

    converter = shutil.which("rsvg-convert")
    if converter is None:
        raise RuntimeError("rsvg-convert is required to create PDF figures")
    sources = {
        "model_feasibility.pdf": ROOT / "generated" / "figures" / "figure_m1_physical_feasibility.svg",
        "communication_bottleneck.pdf": ROOT / "generated" / "figures" / "figure_m7_communication.svg",
        "learning_return.pdf": ROOT / "results" / "formal_learning" / "figures" / "learning_figure_1_return.svg",
        "learning_diagnostics.pdf": ROOT / "results" / "formal_learning" / "figures" / "learning_figure_6_diffqp_diagnostics.svg",
    }
    FIGURES.mkdir(parents=True, exist_ok=True)
    for name, source in sources.items():
        subprocess.run([converter, "--format=pdf", "--output", str(FIGURES / name), str(source)], check=True)

    produced = sorted([*TABLES.glob("*.tex"), *FIGURES.glob("*.pdf")])
    manifest = {
        "generator": Path(__file__).relative_to(ROOT).as_posix(),
        "immutable_inputs": {str(path.relative_to(ROOT)): sha256(path) for path in (MODEL, FOLLOWUP, LEARNING, LEARNING_MANIFEST, PROVENANCE)},
        "outputs": {str(path.relative_to(ROOT)): sha256(path) for path in produced},
        "quantitative_values_manually_entered_in_manuscript": False,
    }
    write(OUT / "manifest.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"PASS: generated {len(produced)} final manuscript artifacts")
    return 0


def generate_consistency_pdfs() -> int:
    """Regenerate only the two patched model-based manuscript figures."""
    converter = shutil.which("rsvg-convert")
    if converter is None:
        raise RuntimeError("rsvg-convert is required to create PDF figures")
    sources = {
        "model_feasibility.pdf": ROOT / "generated" / "figures" / "figure_m1_physical_feasibility.svg",
        "communication_bottleneck.pdf": ROOT / "generated" / "figures" / "figure_m7_communication.svg",
    }
    FIGURES.mkdir(parents=True, exist_ok=True)
    for name, source in sources.items():
        subprocess.run([converter, "--format=pdf", "--output", str(FIGURES / name), str(source)], check=True)

    produced = sorted([*TABLES.glob("*.tex"), *FIGURES.glob("*.pdf")])
    manifest = {
        "generator": Path(__file__).relative_to(ROOT).as_posix(),
        "immutable_inputs": {str(path.relative_to(ROOT)): sha256(path) for path in (MODEL, FOLLOWUP, LEARNING, LEARNING_MANIFEST, PROVENANCE)},
        "outputs": {str(path.relative_to(ROOT)): sha256(path) for path in produced},
        "quantitative_values_manually_entered_in_manuscript": False,
    }
    write(OUT / "manifest.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"PASS: regenerated {len(sources)} consistency-patch figures")
    return 0


if __name__ == "__main__":
    if "--consistency-figures-only" in sys.argv:
        raise SystemExit(generate_consistency_pdfs())
    raise SystemExit(main())

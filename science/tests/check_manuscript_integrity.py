"""Lightweight citation, label, placeholder, and claim-traceability audit."""
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
MANUSCRIPT = ROOT / "manuscript"
text = "\n".join(p.read_text(encoding="utf-8") for p in MANUSCRIPT.rglob("*.tex"))
main_tex = (MANUSCRIPT / "main.tex").read_text(encoding="utf-8")
abstract = main_tex.split(r"\begin{abstract}", 1)[1].split(r"\end{abstract}", 1)[0]
bib = (MANUSCRIPT / "references.bib").read_text(encoding="utf-8")
keys = set(re.findall(r"@[A-Za-z]+\{([^,]+),", bib))
cited = set()
for group in re.findall(r"\\cite\{([^}]+)\}", text):
    cited.update(key.strip() for key in group.split(","))
errors = []
if cited - keys:
    errors.append("missing bibliography keys: " + ", ".join(sorted(cited - keys)))
for token in (
    "eq:reserve", "eq:rho", "eq:gT0", "g_{i,k}^{T0}",
    "predefined zero-gradient fallback", "M_i", r"\rho_i", "eq:zohthermal",
    "eq:zoherror", "eq:ratebound", "eq:auxcbf", "eq:predlower",
    "eq:rhocert", "eq:minconsensus", "eq:backupdomain", "eq:fleetreserve",
    r"\mathsf{Feas}_{i,k}", r"\rho_{\rm air}", "modeled driveline-force contribution",
    "lumped residual longitudinal force/disturbance term", "locally estimated road grade",
    "elapsed dwell time", "most recently accepted safety message", r"\bar r_{i,v}:=",
    r"\epsilon_g>0", "locally Lipschitz extended class-", r"W_i\succeq0",
    r"H_{{\rm QP},i}=W_i+2\epsilon I\succ0", r"p:=u^{\rm RL}",
    "componentwise as an outward-rounded interval enclosure",
    "A midpoint or center-point backup trajectory is not certified",
    "The primary contribution is feasibility certification and physical conflict attribution",
    "registered temperature safety buffer", r"\beta\ge0",
    r"\lambda_I\ge0", r"C_{\min}>0",
    "registered numerical-integration and sampled-model error bound",
    "an overbar on a rate or magnitude denotes its registered nonnegative upper or absolute bound",
):
    if token not in text:
        errors.append("required manuscript token absent: " + token)
trace = (ROOT / "docs" / "claim_evidence_matrix.md").read_text(encoding="utf-8")
if "CLAIM" not in trace or "CODE" not in trace or "EXPERIMENT" not in trace:
    errors.append("claim traceability matrix incomplete")
for relative in (
    "docs/final_claim_evidence_matrix.md",
    "docs/final_submission_audit.md",
    "docs/final_limitations.md",
    "docs/final_revision_summary.md",
    "generated/final/manifest.json",
    "generated/final/tables/table_final_learning_evidence.tex",
):
    if not (ROOT / relative).is_file():
        errors.append(f"missing final artifact: {relative}")
for prohibited in (r"\delta_i^j", r"\delta_j", "improves learning", "globally differentiable", "real-vehicle validated"):
    if prohibited in text:
        errors.append(f"prohibited final-manuscript claim/variable present: {prohibited}")
if r"u_i=[u_i^f,u_i^a]^\top\in\mathbb R^2" not in text:
    errors.append("canonical two-dimensional QP declaration absent")
for token in (
    "99th-percentile controller time",
    "linear independence constraint qualification",
):
    if token not in text:
        errors.append("required technical wording absent from manuscript: " + token)
for token in ("Optimization and control", "Methods for safety"):
    if token not in main_tex:
        errors.append("required Index Term absent: " + token)
if "P99 controller time" in main_tex:
    errors.append("unexpanded P99 remains in Abstract")
for acronym in ("QP", "LICQ", "KKT", "DiffQP", "StopGradient"):
    if re.search(rf"\b{acronym}\b", abstract):
        errors.append("unnecessary Abstract abbreviation remains: " + acronym)
expected_index_terms = (
    "Optimization and control; Methods for safety; Freight transportation and logistics; "
    "Connected and Autonomous Vehicles; predictive feasibility certification; brake thermal fade"
)
if expected_index_terms not in main_tex:
    errors.append("final Index Terms do not match the required metadata string")
if errors:
    print("FAIL: " + " | ".join(errors))
    sys.exit(1)
print(f"PASS: {len(cited)} cited keys resolve; required certificate/evidence markers present")

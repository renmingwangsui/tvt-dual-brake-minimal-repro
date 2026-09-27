"""Create a minimal, reproducible IEEE manuscript source archive."""
from __future__ import annotations

import hashlib
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "dist" / "ieee_tits_submission_source.zip"
MANIFEST = ROOT / "SOURCE_MANIFEST.sha256"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    files = []
    for pattern in ("manuscript/*.tex", "manuscript/*.bib", "manuscript/sections/*.tex", "manuscript/figures/*.tex", "manuscript/figures/*.png", "manuscript/figures/*.pdf", "manuscript/tables/*.tex"):
        files.extend(ROOT.glob(pattern))
    files.append(ROOT / "manuscript" / "main.bbl")
    files.append(ROOT / "template" / "IEEEtran.cls")
    files.extend((ROOT / "generated" / "final").rglob("*"))
    files.extend((ROOT / "generated" / "submission_revision" / "figures").glob("*"))
    files.extend((ROOT / "results" / "submission_revision").glob("*"))
    for name in ("submission_revision.py", "generate_submission_figures.py"):
        files.append(ROOT / "scripts" / name)
    for name in ("revision_changelog.md", "claim_evidence_matrix_submission.md", "submission_audit.md", "interval_arithmetic_soundness.md", "predictive_interval_implementation_audit.md"):
        files.append(ROOT / "docs" / name)
    for name in (
        "table_m_b_controller_comparison.tex", "table_m_c_predictive_certificate.tex",
        "table_m_d_robustness.tex", "table_m_e_communication.tex", "table_m_f_runtime.tex",
        "phase2m_followup_claim_macros.tex",
    ):
        files.append(ROOT / "generated" / "tables" / name)
    files.append(ROOT / "results" / "formal_learning" / "tables" / "table_L1_protocol.tex")
    files = sorted({path for path in files if path.is_file()})
    manifest = "\n".join(f"{sha(path)}  {path.relative_to(ROOT).as_posix()}" for path in files) + "\n"
    MANIFEST.write_text(manifest, encoding="utf-8", newline="\n")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(OUTPUT, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in files:
            archive.write(path, path.relative_to(ROOT).as_posix())
        archive.writestr("SOURCE_MANIFEST.sha256", manifest)
        archive.writestr(
            "README.txt",
            "Compile from manuscript/ with latexmk -pdf main.tex. "
            "Author metadata remains anonymized and must be supplied at the appropriate submission stage.\n",
        )
    print(f"PASS: {OUTPUT.relative_to(ROOT)} files={len(files)} sha256={sha(OUTPUT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

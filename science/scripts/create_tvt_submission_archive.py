"""Package the independent TVT LaTeX version and its referenced assets."""
from __future__ import annotations

import hashlib
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "dist" / "ieee_tvt_submission_source.zip"
MANIFEST = ROOT / "TVT_SOURCE_MANIFEST.sha256"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    files: set[Path] = set()
    for pattern in (
        "manuscript_tvt/*.tex",
        "manuscript_tvt/*.bib",
        "manuscript_tvt/*.bbl",
        "manuscript_tvt/*.md",
        "manuscript_tvt/sections/*.tex",
        "manuscript_tvt/figures/*.tex",
        "manuscript_tvt/figures/*.png",
        "manuscript_tvt/figures/*.pdf",
        "manuscript_tvt/tables/*.tex",
        "generated/tables/*.tex",
        "generated/submission_revision/figures/*.pdf",
        "template/IEEEtran.cls",
    ):
        files.update(ROOT.glob(pattern))
    files.update(path for path in (ROOT / "generated" / "final").rglob("*") if path.is_file())
    files = {path for path in files if path.is_file()}
    manifest = "\n".join(
        f"{digest(path)}  {path.relative_to(ROOT).as_posix()}"
        for path in sorted(files)
    ) + "\n"
    MANIFEST.write_text(manifest, encoding="utf-8", newline="\n")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(OUTPUT, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(files):
            archive.write(path, path.relative_to(ROOT).as_posix())
        archive.writestr(MANIFEST.name, manifest)
        archive.writestr(
            "README_TVT.txt",
            "Compile from manuscript_tvt/ with latexmk -pdf -interaction=nonstopmode "
            "-halt-on-error main.tex. This is an anonymous author-review draft; "
            "author metadata and any required AI-use disclosure remain pending.\n",
        )
    print(f"TVT archive: {OUTPUT}; files={len(files)}; sha256={digest(OUTPUT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

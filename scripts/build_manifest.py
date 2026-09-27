"""Write SHA256SUMS for the immutable reproduction inputs, excluding build outputs."""

from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "SHA256SUMS"
EXCLUDED_DIRS = {".git", ".venv", "__pycache__", ".pytest_cache"}
EXCLUDED_SUFFIXES = {".aux", ".blg", ".fdb_latexmk", ".fls", ".log", ".out", ".pyc"}


def included(path: Path) -> bool:
    relative = path.relative_to(ROOT)
    if path == MANIFEST or not path.is_file() or path.is_symlink():
        return False
    if any(part in EXCLUDED_DIRS for part in relative.parts):
        return False
    if relative.parts[:2] == ("science", "artifacts"):
        return False
    if path.suffix in EXCLUDED_SUFFIXES or path.name == "main.pdf":
        return False
    return True


def main() -> None:
    paths = sorted((path for path in ROOT.rglob("*") if included(path)), key=lambda p: p.relative_to(ROOT).as_posix())
    lines = []
    for path in paths:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {path.relative_to(ROOT).as_posix()}")
    MANIFEST.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"MANIFEST_FILES={len(lines)}")


if __name__ == "__main__":
    main()

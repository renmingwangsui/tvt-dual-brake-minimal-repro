"""Validate immutable result inputs before any paper figure can be generated."""
from pathlib import Path
import csv
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
MANIFEST = RESULTS / "manifest.json"
REQUIRED_META = {"code_commit", "config_sha256", "environment", "hardware", "files"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    if not MANIFEST.exists():
        print("[EXPERIMENT REQUIRED] results/manifest.json is missing")
        return 2
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    missing = REQUIRED_META.difference(data)
    if missing:
        print("[DATA REQUIRED] manifest fields missing: " + ", ".join(sorted(missing)))
        return 2
    for item in data["files"]:
        path = RESULTS / item["path"]
        if not path.is_file() or sha256(path) != item["sha256"]:
            print(f"[DATA REQUIRED] hash mismatch or missing file: {path}")
            return 2
    print("PASS: result manifest and file hashes verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())


"""Generate Phase 2M figures with a fail-closed publication gate."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.model_based.figures import generate_debug_figures  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "artifacts" / "model_based" / "debug")
    parser.add_argument("--output", type=Path, default=ROOT / "figures" / "model_based" / "debug")
    parser.add_argument(
        "--allow-debug", action="store_true",
        help="generate visibly watermarked diagnostic figures; never makes them paper eligible",
    )
    args = parser.parse_args()
    manifest = json.loads((args.input / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("paper_eligible") is not True and not args.allow_debug:
        print("REFUSED: input is not paper eligible; use --allow-debug only for watermarked diagnostics", file=sys.stderr)
        return 2
    if manifest.get("mode") == "DEBUG_SYNTHETIC":
        paths = generate_debug_figures(args.input, args.output)
        print(f"PASS: generated {len(paths)} watermarked DEBUG figures; paper_eligible=false")
        return 0
    print("REFUSED: literature-calibrated figure path is not enabled until provenance validation passes", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

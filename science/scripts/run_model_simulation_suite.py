"""Run all Phase 2M experiments in explicitly synthetic DEBUG mode."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from analysis.model_based.analyze import analyze_model_simulations  # noqa: E402
from experiments.model_based.pipeline import (  # noqa: E402
    DEFAULT_CONFIG, DEFAULT_OUTPUT, DEFAULT_PARAMETERS, run_debug_suite,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--parameters", type=Path, default=DEFAULT_PARAMETERS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    manifest = run_debug_suite(args.config.resolve(), args.parameters.resolve(), args.output.resolve())
    analyze_model_simulations(args.output.resolve())
    print(f"PASS: {len(manifest['experiments'])} DEBUG experiments; paper_eligible=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

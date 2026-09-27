"""Fail-closed Phase 2M-PAPER entrypoint.

The campaign runner reuses an already completed, hash-matching immutable
campaign; it does not overwrite raw records.  Analysis then verifies all raw
hashes before deriving artifacts.
"""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from experiments.model_based.paper_pipeline import run_paper_suite, validate_frozen_inputs  # noqa: E402
from analysis.model_based.paper_analysis import generate_all  # noqa: E402


if __name__ == "__main__":
    inputs = validate_frozen_inputs()
    campaign = run_paper_suite(inputs)
    result = generate_all(campaign)
    print(f"PHASE 2M-PAPER: PASS ({result['integrity']['run_manifests_verified']} immutable runs verified)")

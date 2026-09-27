"""Verify and analyze the immutable Phase 2M targeted follow-up."""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from analysis.model_based.followup_analysis import generate  # noqa: E402


if __name__ == "__main__":
    summary = generate()
    print(f"PHASE 2M-FOLLOWUP ANALYSIS: PASS ({summary['new_paper_eligible_followup_runs']} runs)")

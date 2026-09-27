"""Verify and analyze the existing immutable Phase 2M-PAPER campaign."""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from analysis.model_based.paper_analysis import generate_all  # noqa: E402


if __name__ == "__main__":
    result = generate_all()
    print(f"PHASE 2M-PAPER ANALYSIS: PASS ({len(result['figures'])} figures, {len(result['tables'])} tables)")

"""Run only the frozen Phase 2M targeted follow-up experiments."""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from experiments.model_based.followup_pipeline import run_followup  # noqa: E402


if __name__ == "__main__":
    print(f"PHASE 2M-FOLLOWUP: PASS ({run_followup()})")

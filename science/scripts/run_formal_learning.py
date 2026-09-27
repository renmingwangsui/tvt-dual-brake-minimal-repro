#!/usr/bin/env python3
"""Execute the frozen Phase 2C-3 formal campaign once."""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from experiments.learning.formal_pipeline import main


if __name__ == "__main__":
    raise SystemExit(main())

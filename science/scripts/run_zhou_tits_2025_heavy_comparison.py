"""Run the frozen matched heavy-duty external-baseline comparison once."""
from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from external_baselines.zhou_heavy_comparison import run_formal_comparison


if __name__ == "__main__":
    run_formal_comparison()

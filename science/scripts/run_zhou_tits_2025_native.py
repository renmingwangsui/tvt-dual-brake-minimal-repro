"""Execute the frozen Zhou et al. native-reproduction gate exactly once."""
from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from external_baselines.zhou_tits_2025 import run_native_campaign


if __name__ == "__main__":
    run_native_campaign()

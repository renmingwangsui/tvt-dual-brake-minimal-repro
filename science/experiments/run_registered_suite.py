"""Gate for the pre-registered vehicle/RL study; never substitutes synthetic data."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments" / "configs" / "registered_suite.yaml"
BLOCKERS = ("DATA_REQUIRED", "AUTHOR_DECISION")


def main() -> int:
    text = CONFIG.read_text(encoding="utf-8")
    unresolved = sorted({token for token in BLOCKERS if token in text})
    if unresolved:
        print("[EXPERIMENT REQUIRED] registered suite is blocked by: " + ", ".join(unresolved))
        print("Refusing to create vehicle, training, timing, or safety results.")
        return 2
    print("Configuration gate passed. Connect the audited safety core to the validated simulator here.")
    return 0


if __name__ == "__main__":
    sys.exit(main())


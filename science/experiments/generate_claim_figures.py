"""Generate claim figures only from provenance-verified, schema-complete logs."""
from pathlib import Path
import csv
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "results" / "per_step.csv"
SCHEMA = ROOT / "docs" / "log_schema.csv"
EPISODES = ROOT / "results" / "per_episode.csv"
EPISODE_SCHEMA = ROOT / "docs" / "episode_log_schema.csv"


def main() -> int:
    gate = subprocess.run([sys.executable, str(ROOT / "experiments" / "check_result_provenance.py")])
    if gate.returncode:
        print("[EXPERIMENT REQUIRED] no claim figure was generated")
        return gate.returncode
    for log, schema in ((LOG, SCHEMA), (EPISODES, EPISODE_SCHEMA)):
        required = {row["column"] for row in csv.DictReader(schema.open(encoding="utf-8")) if row["required"] == "yes"}
        with log.open(encoding="utf-8", newline="") as stream:
            columns = set(next(csv.reader(stream)))
        missing = required.difference(columns)
        if missing:
            print(f"[DATA REQUIRED] {log.name} lacks: " + ", ".join(sorted(missing)))
            return 2
    print("Schema/provenance passed for both step and episode logs. Generate Figs. 1--8 and Predictive P1--P4 only from these verified inputs.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

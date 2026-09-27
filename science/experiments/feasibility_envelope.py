"""Deterministic analytical pre-validation for Proposition 2.

The sweep is deliberately normalized and synthetic. It tests only whether adding
bounded auxiliary braking enlarges the instantaneous demand/temperature region;
it is not a vehicle validation or an RL result.
"""
from pathlib import Path
import csv

ROOT = Path(__file__).resolve().parents[1]
CSV = Path(__file__).with_name("feasibility_envelope.csv")
SUMMARY = Path(__file__).with_name("feasibility_envelope_summary.txt")

temperatures = [100.0 + 2.5 * i for i in range(161)]
demands = [20.0 + i for i in range(161)]
auxiliary_cap = 80.0
rows = []
for t in temperatures:
    cap = max(15.0, 140.0 * (1.0 - 0.0025 * (t - 100.0)))
    for d in demands:
        rows.append((t, d, cap, d <= cap, d <= cap + auxiliary_cap))

with CSV.open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["temperature_degC", "demand_kN", "friction_cap_kN", "friction_only_feasible", "dual_brake_feasible"])
    for t, d, cap, f_ok, d_ok in rows:
        writer.writerow([f"{t:.3f}", f"{d:.3f}", f"{cap:.3f}", int(f_ok), int(d_ok)])

f_fraction = sum(int(row[3]) for row in rows) / len(rows)
d_fraction = sum(int(row[4]) for row in rows) / len(rows)
with SUMMARY.open("w", encoding="utf-8") as f:
    f.write(f"grid_points={len(rows)}\n")
    f.write(f"friction_only_feasible_fraction={f_fraction:.6f}\n")
    f.write(f"dual_brake_feasible_fraction={d_fraction:.6f}\n")
    f.write(f"absolute_fraction_gain={d_fraction-f_fraction:.6f}\n")
    f.write("scope=normalized instantaneous analytical sweep; not vehicle or RL evidence\n")

print(SUMMARY.read_text(encoding="utf-8"))

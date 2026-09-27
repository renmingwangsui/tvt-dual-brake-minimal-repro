"""Validate the required machine-readable symbol audit."""
from pathlib import Path
import csv

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "docs" / "symbol_audit.csv"
EXPECTED = [
    "symbol",
    "definition",
    "unit",
    "first use",
    "state/control/parameter",
    "paper equation",
    "code variable",
    "parameter source",
    "hard guarantee yes/no",
]
with PATH.open(encoding="utf-8", newline="") as stream:
    reader = csv.DictReader(stream)
    assert reader.fieldnames == EXPECTED, reader.fieldnames
    rows = list(reader)
assert len(rows) >= 50
for number, row in enumerate(rows, start=2):
    missing = [field for field in EXPECTED if not row[field].strip()]
    assert not missing, f"row {number} lacks {missing}"
symbols = {row["symbol"] for row in rows}
for required in ("h_i^c", "h_i^T", "h_i^F", "h_i^A", "M_i", "H_QP", "H_pred", "N_rec"):
    assert required in symbols, required
print(f"PASS: symbol audit has {len(rows)} complete entries and resolves H_QP/H_pred/N_rec")


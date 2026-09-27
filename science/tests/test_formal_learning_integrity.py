#!/usr/bin/env python3
"""Phase 2C-3 protocol and completed-campaign integrity gates."""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from experiments.learning.formal_pipeline import (
    EXPECTED_PROTOCOL_HASH,
    PROTOCOL_PATH,
    RESULT_ROOT,
    file_hash,
    load_protocol,
    verify_integrity,
)


def main() -> int:
    protocol = load_protocol()
    assert file_hash(PROTOCOL_PATH) == EXPECTED_PROTOCOL_HASH
    assert len(protocol["training_seeds"]) == 10
    assert len(protocol["evaluation_scenarios"]) == 7
    assert protocol["early_stopping_rule"] == "none"
    assert protocol["reward_definition"]["ppo_likelihood_action"] == "u_RL"
    assert protocol["reward_definition"]["transition_action"] == "u_final"
    assert "jerk-slack" in protocol["forward_qp"]
    result = verify_integrity(RESULT_ROOT, require_results=False)
    if result["status"] == "NOT_RUN":
        print(f"PASS: frozen Phase 2C-3 protocol preflight; results not run protocol_sha256={file_hash(PROTOCOL_PATH)}")
    else:
        assert result["run_count"] == 20
        assert result["failed_runs"] == 0 and result["invalidated_runs"] == 20
        assert result["pairwise_initialization_match"] and result["forward_controller_hash_match"]
        manifest = json.loads((RESULT_ROOT / "result_manifest.json").read_text(encoding="utf-8"))
        assert len(manifest["figures"]) == 6 and len(manifest["tables"]) == 5
        print(f"PASS: Phase 2C-3 immutable campaign integrity runs={result['run_count']} protocol_sha256={file_hash(PROTOCOL_PATH)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

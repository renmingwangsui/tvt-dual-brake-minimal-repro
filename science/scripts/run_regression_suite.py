"""Portable fail-fast entrypoint for all accepted regression groups."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]

REFERENCE_TESTS = (
    "tests/check_manuscript_integrity.py",
    "tests/check_predictive_traceability.py",
    "tests/check_symbol_audit.py",
    "tests/check_traceability.py",
    "tests/test_actor_parameterization.py",
    "tests/test_equation_regression.py",
    "tests/test_low_speed_g_t0.py",
    "tests/test_predictive_objective.py",
    "tests/test_predictive_reserve.py",
    "tests/test_revision_gaps.py",
    "tests/test_symbolic_derivations.py",
    "tests/test_units_and_limits.py",
)

PHASE_2A_TESTS = (
    "tests/test_simulator_actuators.py",
    "tests/test_simulator_dynamics.py",
    "tests/test_simulator_integration.py",
    "tests/test_simulator_road.py",
    "tests/test_simulator_thermal.py",
)

PHASE_2B_TESTS = (
    "tests/test_phase2b_bottleneck.py",
    "tests/test_phase2b_closed_loop.py",
    "tests/test_phase2b_environment.py",
    "tests/test_phase2b_network.py",
    "tests/test_phase2b_safety_integration.py",
)

PHASE_2C1_TESTS = (
    "tests/test_mappo_buffer_gae.py",
    "tests/test_mappo_checkpoint_eval.py",
    "tests/test_mappo_distribution_actor.py",
    "tests/test_mappo_smoke.py",
    "tests/test_mappo_trainer.py",
)

PHASE_2C2_TESTS = (
    "tests/test_phase2c2_diffqp_math.py",
    "tests/test_phase2c2_matched_training.py",
)

PHASE_2C3_TESTS = (
    "tests/test_formal_learning_integrity.py",
)

PHASE_2M_TESTS = (
    "tests/test_model_simulation_pipeline.py",
)

PHASE_2M_LIT_TESTS = (
    "tests/test_literature_calibration.py",
)

PHASE_2M_FOLLOWUP_TESTS = (
    "tests/test_phase2m_followup.py",
)

GROUPS = (
    ("REFERENCE", REFERENCE_TESTS),
    ("PHASE_2A", PHASE_2A_TESTS),
    ("PHASE_2B", PHASE_2B_TESTS),
    ("PHASE_2C1", PHASE_2C1_TESTS),
    ("PHASE_2C2", PHASE_2C2_TESTS),
    ("PHASE_2C3", PHASE_2C3_TESTS),
    ("PHASE_2M", PHASE_2M_TESTS),
    ("PHASE_2M_LIT", PHASE_2M_LIT_TESTS),
    ("PHASE_2M_FOLLOWUP", PHASE_2M_FOLLOWUP_TESTS),
)


def main() -> int:
    expected = {name for _group, tests in GROUPS for name in tests}
    discovered = {path.relative_to(ROOT).as_posix() for path in (ROOT / "tests").glob("*.py")}
    if expected != discovered:
        print("Regression registry does not exactly match tests/*.py", file=sys.stderr)
        print(f"unregistered={sorted(discovered - expected)}", file=sys.stderr)
        print(f"missing={sorted(expected - discovered)}", file=sys.stderr)
        return 2

    environment = os.environ.copy()
    environment.setdefault("PYTHONUTF8", "1")
    passed = 0
    for group, tests in GROUPS:
        for relative_name in tests:
            print(f"RUN [{group}] {relative_name}", flush=True)
            completed = subprocess.run(
                [sys.executable, str(ROOT / relative_name)],
                cwd=ROOT,
                env=environment,
                check=False,
            )
            if completed.returncode != 0:
                print(
                    f"FAIL [{group}] {relative_name} exit={completed.returncode}",
                    file=sys.stderr,
                )
                return completed.returncode or 1
            passed += 1
    print(f"PASS: {passed} scripts (reference=12, Phase2A=5, Phase2B=5, Phase2C1=5, Phase2C2=2, Phase2C3=1, Phase2M=1, Phase2M-LIT=1, Phase2M-FOLLOWUP=1)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

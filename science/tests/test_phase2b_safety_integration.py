"""Integrated hard-QP, predictive backup, supervisor, causal order and g_T0 tests."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from controllers.integrated_safety_controller import IntegratedSafetyController  # noqa: E402
from envs.heavy_platoon_env import HeavyPlatoonEnv  # noqa: E402
from envs.scenario import build_synthetic_debug_scenario  # noqa: E402
from safety.supervisor import SupervisorMode, SupervisorThresholds  # noqa: E402


env = HeavyPlatoonEnv(build_synthetic_debug_scenario(1))
result = env.step().controller_results[0]
assert result.qp_status == "optimal"
assert result.two_pass.provisional_prediction.first_control == result.two_pass.provisional_command
assert result.two_pass.final_prediction.backup_control_widths_N[0] == (0.0, 0.0)
assert any(max(widths) > 0.0 for widths in result.two_pass.final_prediction.backup_control_widths_N[1:])
required_rows = {
    "collision_hocbf",
    "thermal_hocbf",
    "friction_command_envelope",
    "auxiliary_command_envelope",
    "fade_cbf",
    "realized_auxiliary_cbf",
}
assert required_rows.issubset(result.hard_residuals)
assert result.residuals_valid
order = result.callback_order
assert order.index("predictive_tube") < order.index("supervisor_after_prediction")
assert order.index("validate_final_hard_rows") < order.index("queue_verified_action")

# Force an anticipatory supervisor action through registered thresholds; changed action must reverify.
changed_env = HeavyPlatoonEnv(build_synthetic_debug_scenario(1, "constant_descent"))
base_config = changed_env.controllers[0].config
changed_env.controllers[0] = IntegratedSafetyController(replace(
    base_config,
    supervisor_thresholds=SupervisorThresholds(1.0, 2.0, 0.001, 0.01, 2),
))
changed = changed_env.step().controller_results[0]
assert changed.supervisor_decision.mode is SupervisorMode.ANTICIPATORY
assert changed.two_pass.prediction_reverified
assert "reverify_changed_final_action" in changed.callback_order
assert changed.callback_order.index("reverify_changed_final_action") < changed.callback_order.index("validate_final_hard_rows")

# Exact existing supervisor implementation exercises all four modes.
controller = IntegratedSafetyController(base_config)
assert controller.evaluate_supervisor(0.2, 0.2, True, False).mode is SupervisorMode.NORMAL
assert controller.evaluate_supervisor(0.2, 0.05, True, False).mode is SupervisorMode.ANTICIPATORY
assert controller.evaluate_supervisor(0.01, 0.2, True, False).mode is SupervisorMode.CERTIFIED_BACKUP
assert controller.evaluate_supervisor(-0.1, -0.2, False, True).mode is SupervisorMode.MINIMAL_RISK
controller.previous_mode = SupervisorMode.ANTICIPATORY
first_recovery = controller.evaluate_supervisor(0.2, 0.2, True, False)
assert first_recovery.mode is SupervisorMode.ANTICIPATORY and first_recovery.recovery_counter == 1
controller.recovery_counter = first_recovery.recovery_counter
second_recovery = controller.evaluate_supervisor(0.2, 0.2, True, False)
assert second_recovery.mode is SupervisorMode.NORMAL

low = HeavyPlatoonEnv(build_synthetic_debug_scenario(1, "low_speed"))
low_result = low.step().logs[0]
assert low_result.g_T0_K is not None
assert any(name.startswith("thermal_zoh") for name in low_result.hard_row_residuals)

print("PASS: Phase 2B QP/certificates/backup/supervisor/order/reverification/residuals/g_T0")

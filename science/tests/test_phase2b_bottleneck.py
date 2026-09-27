"""Central diagnostic versus communication-available recursive bottleneck semantics."""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from envs.heavy_platoon_env import HeavyPlatoonEnv  # noqa: E402
from envs.scenario import build_synthetic_debug_scenario  # noqa: E402
from network.v2v_channel import ChannelConfig  # noqa: E402


zero = HeavyPlatoonEnv(build_synthetic_debug_scenario(3, channel_config=ChannelConfig()))
zero_result = zero.step()
assert abs(zero_result.head_local_rho_up - zero_result.centralized_fleet_rho_H_cert) < 1e-12
assert zero_result.logs[0].centralized_equals_local_asserted
assert zero_result.logs[0].critical_vehicle_id in {1, 2, 3}

delayed_config = ChannelConfig(delay_model="fixed", fixed_delay_s=0.2)
delayed = HeavyPlatoonEnv(build_synthetic_debug_scenario(3, "communication_delay_loss", delayed_config))
delayed_result = delayed.step()
assert delayed_result.head_local_rho_up != delayed_result.centralized_fleet_rho_H_cert
assert not any(record.centralized_equals_local_asserted for record in delayed_result.logs)

loss_config = ChannelConfig(forced_loss_sequences=frozenset({0}))
lost = HeavyPlatoonEnv(build_synthetic_debug_scenario(3, "communication_delay_loss", loss_config))
lost_result = lost.step()
assert lost_result.head_local_rho_up != lost_result.centralized_fleet_rho_H_cert
assert not any(record.centralized_equals_local_asserted for record in lost_result.logs)

print("PASS: zero-delay recursive minimum equivalence and delayed/lost non-equivalence semantics")

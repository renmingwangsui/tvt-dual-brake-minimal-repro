"""Six N=3 DEBUG scenarios and N=5/10/20/40 scalability smoke execution."""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from envs.heavy_platoon_env import HeavyPlatoonEnv, Phase2BLogRecord  # noqa: E402
from envs.scenario import build_synthetic_debug_scenario  # noqa: E402
from network.v2v_channel import ChannelConfig  # noqa: E402


scenarios = (
    ("flat_steady", ChannelConfig()),
    ("constant_descent", ChannelConfig()),
    ("leader_braking", ChannelConfig()),
    ("hot_brake", ChannelConfig()),
    ("auxiliary_limitation", ChannelConfig()),
    (
        "communication_delay_loss",
        ChannelConfig(delay_model="fixed", fixed_delay_s=0.15, packet_loss_probability=0.2, random_seed=11),
    ),
)
episode_count = 0
for name, channel in scenarios:
    env = HeavyPlatoonEnv(build_synthetic_debug_scenario(3, name, channel))
    episode = env.run(2)
    episode_count += 1
    assert len(episode) == 2 and len(env.logs) == 6
    assert all(not record.paper_eligible and record.data_provenance == "synthetic_debug" for record in env.logs)
    assert all(record.qp_status in {"optimal", "empty_hard_polytope"} for record in env.logs)
    assert all("collision_hocbf" in record.hard_row_residuals for record in env.logs)

for n in (5, 10, 20, 40):
    env = HeavyPlatoonEnv(build_synthetic_debug_scenario(n, "flat_steady"))
    result = env.step()
    assert len(result.states) == n and len(result.logs) == n

try:
    Phase2BLogRecord(
        0.0, 1, build_synthetic_debug_scenario(1).initial_states[0], object(),
        (0.0, 0.0), (0.0, 0.0), (0.0, 0.0), {}, {},
        0.0, 0.0, 0.0, 0.0, None, 0.0, 0.0, 0.0,
        1, 1, "none", 0.0, True, "normal_actor", False, "optimal", True, True,
        paper_eligible=True,
    )
except ValueError:
    pass
else:
    raise AssertionError("DEBUG log cannot become paper eligible")

print(f"PASS: {episode_count} N=3 closed-loop scenarios and N=5/10/20/40 smoke execution")

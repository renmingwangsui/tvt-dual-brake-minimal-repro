"""Position-coordinate road profile and interpolation-boundary tests."""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from envs.road_profile import ConstantGrade, InterpolatedRoadProfile, PiecewiseLinearGrade  # noqa: E402


flat = ConstantGrade(0.0)
downhill = ConstantGrade(-0.05)
uphill = ConstantGrade(0.05)
for position in (-100.0, 0.0, 123.4):
    assert flat.grade_rad(position) == 0.0
    assert downhill.grade_rad(position) < 0.0
    assert uphill.grade_rad(position) > 0.0

profile = PiecewiseLinearGrade(
    positions_m=(0.0, 100.0, 200.0, 300.0),
    grades_rad=(0.0, -0.04, -0.04, 0.02),
)
assert profile.grade_rad(0.0) == 0.0
assert abs(profile.grade_rad(50.0) - (-0.02)) < 1e-15
assert profile.grade_rad(100.0) == -0.04
assert profile.grade_rad(200.0) == -0.04
assert abs(profile.grade_rad(250.0) - (-0.01)) < 1e-15
assert profile.grade_rad(300.0) == 0.02
assert profile.grade_rad(-1.0) == 0.0
assert profile.grade_rad(301.0) == 0.02

sampled = InterpolatedRoadProfile.from_samples((0.0, 10.0), (0.01, 0.03), "error")
assert abs(sampled.grade_rad(5.0) - 0.02) < 1e-15
for outside in (-0.1, 10.1):
    try:
        sampled.grade_rad(outside)
    except ValueError:
        pass
    else:
        raise AssertionError("error boundary policy must reject extrapolation")

print("PASS: constant, piecewise-linear and sampled road profiles including boundaries")

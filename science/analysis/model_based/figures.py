"""Dependency-free SVG diagnostics for Phase 2M DEBUG runs."""
from __future__ import annotations

from html import escape
import json
from pathlib import Path
from typing import Any


def _read(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _svg(title: str, subtitle: str, series: list[tuple[str, list[float]]], destination: Path) -> None:
    width, height = 960, 520
    left, top, plot_w, plot_h = 90, 100, 800, 330
    all_values = [value for _label, values in series for value in values]
    low, high = (min(all_values), max(all_values)) if all_values else (0.0, 1.0)
    if high <= low:
        high = low + 1.0
    colors = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9")
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="34" text-anchor="middle" font-family="sans-serif" font-size="22" font-weight="bold">{escape(title)}</text>',
        f'<text x="{width/2}" y="62" text-anchor="middle" font-family="sans-serif" font-size="14" fill="#b2182b">SYNTHETIC DEBUG — NOT PAPER ELIGIBLE</text>',
        f'<text x="{width/2}" y="82" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#555">{escape(subtitle)}</text>',
        f'<line x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}" stroke="#222"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" stroke="#222"/>',
    ]
    for tick in range(6):
        y = top + plot_h - tick * plot_h / 5
        value = low + tick * (high - low) / 5
        parts += [f'<line x1="{left}" y1="{y:.1f}" x2="{left+plot_w}" y2="{y:.1f}" stroke="#ddd"/>',
                  f'<text x="{left-8}" y="{y+4:.1f}" text-anchor="end" font-family="monospace" font-size="11">{value:.3g}</text>']
    max_len = max((len(values) for _label, values in series), default=1)
    for index, (label, values) in enumerate(series):
        color = colors[index % len(colors)]
        points = []
        for x_index, value in enumerate(values):
            x = left + (x_index / max(1, max_len - 1)) * plot_w
            y = top + (high - value) / (high - low) * plot_h
            points.append(f"{x:.1f},{y:.1f}")
        parts.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="2"/>')
        legend_y = 455 + (index // 3) * 20
        legend_x = 100 + (index % 3) * 280
        parts.append(f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x+24}" y2="{legend_y}" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<text x="{legend_x+30}" y="{legend_y+4}" font-family="sans-serif" font-size="12">{escape(label)}</text>')
    parts.append('</svg>')
    destination.write_text("\n".join(parts) + "\n", encoding="utf-8", newline="\n")


def generate_debug_figures(output: Path, figures: Path) -> list[Path]:
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("mode") != "DEBUG_SYNTHETIC" or manifest.get("paper_eligible") is not False:
        raise ValueError("DEBUG figure generator requires an explicitly ineligible DEBUG manifest")
    figures.mkdir(parents=True, exist_ok=True)
    specifications = [
        ("m1_feasibility_geometry.svg", "M1 Feasibility Geometry", "reserve over real-unit parameter-grid cases", "M1", "temperature_K", "rho"),
        ("m2_long_descent_time_history.svg", "M2 Long-Descent Comparison", "temperature histories by nominal allocation strategy", "M2", "method", "temperature_K"),
        ("m3_hot_brake_thermal_conflict.svg", "M3 Hot-Brake Thermal Conflict", "temperature histories by initial temperature", "M3", "initial_temperature_K", "rho"),
        ("m4_predictive_warning.svg", "M4 Predictive Certificate", "instantaneous and horizon reserves", "M4", "method", "rho_H_cert"),
        ("m5_horizon_sensitivity.svg", "M5 Horizon Sensitivity", "certified reserve by prediction horizon", "M5", "H_pred", "rho_H_cert"),
        ("m7_communication_bottleneck.svg", "M7 Communication Study", "local recursive bottleneck under channel cases", "M7", "channel_case", "rho_up_local"),
        ("m9_string_stability.svg", "M9 Empirical String Response", "numerical G2 by platoon size/disturbance; no theorem", "M9", "disturbance", "G2_i"),
        ("m10_runtime.svg", "M10 Runtime Scaling", "raw wall-clock controller samples; no hardware guarantee", "M10", "platoon_size", "total_controller_s"),
    ]
    generated: list[Path] = []
    for filename, title, subtitle, experiment, group_key, value_key in specifications:
        rows = _read(output / f"{experiment.lower()}_records.jsonl")
        groups: dict[str, list[float]] = {}
        if experiment == "M4":
            groups = {
                "instantaneous rho": [float(row["rho"]) for row in rows],
                "predictive rho_H_cert": [float(row["rho_H_cert"]) for row in rows],
            }
        else:
            for row in rows:
                value = row.get(value_key)
                if value is not None:
                    groups.setdefault(str(row[group_key]), []).append(float(value))
        destination = figures / filename
        _svg(title, subtitle, list(groups.items())[:12], destination)
        generated.append(destination)
    index = {
        "mode": "DEBUG_SYNTHETIC", "paper_eligible": False,
        "watermark": "SYNTHETIC DEBUG — NOT PAPER ELIGIBLE",
        "figures": [path.name for path in generated],
    }
    (figures / "figure_manifest.json").write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8", newline="\n")
    return generated

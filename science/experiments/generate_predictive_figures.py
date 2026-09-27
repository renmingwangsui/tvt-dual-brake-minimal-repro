"""Generate Predictive P1--P8 exclusively from provenance-verified real logs."""
from __future__ import annotations

import csv
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STEP_LOG = ROOT / "results" / "per_step.csv"
EPISODE_LOG = ROOT / "results" / "per_episode.csv"
OUT = ROOT / "results" / "figures"

STEP_REQUIRED = {
    "time_s",
    "episode_id",
    "vehicle_id",
    "rho_instantaneous",
    "rho_cert_H",
    "rho_upstream_local_H",
    "rho_fleet_centralized_H",
    "critical_vehicle_id",
    "critical_prediction_step",
    "predictive_trigger",
    "certificate_loss",
    "critical_limit_type",
    "tube_width_total",
    "prediction_horizon",
    "uncertainty_scale",
    "end_to_end_ms",
}
EPISODE_REQUIRED = {
    "reserve_variant",
    "certificate_loss_episode",
    "backup_occupancy",
    "early_warning_time_s",
    "min_h_T_K",
    "mission_time_s",
}


def read_rows(path: Path, required: set[str]) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path.name} lacks {sorted(missing)}")
        return list(reader)


def main() -> int:
    gate = subprocess.run(
        [sys.executable, str(ROOT / "experiments" / "generate_claim_figures.py")],
        check=False,
    )
    if gate.returncode:
        print("[EXPERIMENT REQUIRED] predictive figures were not generated")
        return gate.returncode

    import matplotlib.pyplot as plt

    steps = read_rows(STEP_LOG, STEP_REQUIRED)
    episodes = read_rows(EPISODE_LOG, EPISODE_REQUIRED)
    OUT.mkdir(parents=True, exist_ok=True)

    chosen_episode = min(int(row["episode_id"]) for row in steps)
    trace = [row for row in steps if int(row["episode_id"]) == chosen_episode]
    trace.sort(key=lambda row: (float(row["time_s"]), int(row["vehicle_id"])))

    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    by_vehicle: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in trace:
        by_vehicle[int(row["vehicle_id"])].append(row)
    for vehicle, rows in sorted(by_vehicle.items()):
        ax.plot([float(r["time_s"]) for r in rows], [float(r["rho_instantaneous"]) for r in rows], alpha=0.45, label=f"rho vehicle {vehicle}")
        ax.plot([float(r["time_s"]) for r in rows], [float(r["rho_cert_H"]) for r in rows], linewidth=1.8, label=f"rho cert vehicle {vehicle}")
    trigger_times = [float(r["time_s"]) for r in trace if int(r["predictive_trigger"])]
    if trigger_times:
        ax.axvline(min(trigger_times), color="tab:orange", linestyle="--", label="predictive trigger")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set(xlabel="time (s)", ylabel="complete certificate rho", title="P1: instantaneous versus predictive certificate")
    ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / "P1_instantaneous_vs_predictive_rho.pdf")
    plt.close(fig)

    times = sorted({float(row["time_s"]) for row in trace})
    vehicles = sorted(by_vehicle)
    value = {(float(r["time_s"]), int(r["vehicle_id"])): float(r["rho_cert_H"]) for r in trace}
    heat = [[value.get((time, vehicle), float("nan")) for time in times] for vehicle in vehicles]
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    image = ax.imshow(heat, aspect="auto", origin="lower", extent=[min(times), max(times), min(vehicles), max(vehicles)])
    fig.colorbar(image, ax=ax, label=r"$\rho_i^{H,\mathrm{cert}}$")
    ax.set(xlabel="time (s)", ylabel="vehicle id", title="Predictive P2: vehicle-time reserve heatmap")
    fig.tight_layout()
    fig.savefig(OUT / "P2_predictive_rho_heatmap.pdf")
    plt.close(fig)

    unique_time: dict[float, dict[str, str]] = {}
    for row in trace:
        unique_time.setdefault(float(row["time_s"]), row)
    ordered = [unique_time[time] for time in sorted(unique_time)]
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    ax.plot([float(r["time_s"]) for r in ordered], [float(r["rho_fleet_centralized_H"]) for r in ordered], label="centralized fleet certificate")
    ax.plot([float(r["time_s"]) for r in ordered], [float(r["rho_upstream_local_H"]) for r in ordered], linestyle="--", label="available upstream certificate")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax2 = ax.twinx()
    ax2.step([float(r["time_s"]) for r in ordered], [int(r["critical_vehicle_id"]) for r in ordered], where="post", color="tab:red", label="bottleneck vehicle")
    ax.set(xlabel="time (s)", ylabel="fleet certificate", title="P3: fleet bottleneck certificate")
    ax2.set_ylabel("critical vehicle id")
    fig.tight_layout()
    fig.savefig(OUT / "P3_fleet_bottleneck_trace.pdf")
    plt.close(fig)

    variants = ["instantaneous_only", "predictive_vehicle", "predictive_vehicle_plus_fleet"]
    metrics = ["certificate_loss_episode", "backup_occupancy", "early_warning_time_s", "min_h_T_K", "mission_time_s"]
    means: dict[str, list[float]] = {}
    for metric in metrics:
        means[metric] = []
        for variant in variants:
            values = [float(row[metric]) for row in episodes if row["reserve_variant"] == variant and row[metric] != ""]
            if not values:
                raise ValueError(f"no registered values for {variant}/{metric}")
            means[metric].append(sum(values) / len(values))
    fig, axes = plt.subplots(1, len(metrics), figsize=(12.0, 3.0))
    for ax, metric in zip(axes, metrics):
        ax.bar(range(len(variants)), means[metric])
        ax.set_xticks(range(len(variants)), ["instant", "predict", "predict+fleet"], rotation=35, ha="right")
        ax.set_title(metric.replace("_", " "), fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "P6_predictive_ablation.pdf")
    plt.close(fig)

    # P4: physical attribution at the critical vehicle/step.
    labels = [r["critical_limit_type"] for r in ordered]
    categories = sorted(set(labels))
    counts = [labels.count(label) for label in categories]
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    ax.bar(categories, counts)
    ax.set(xlabel="limiting hard row", ylabel="critical samples", title="P4: physical bottleneck attribution")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout(); fig.savefig(OUT / "P4_limit_attribution.pdf"); plt.close(fig)

    # P5: trigger and later realized boundary on the same certificate trace.
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    ax.plot([float(r["time_s"]) for r in ordered], [float(r["rho_upstream_local_H"]) for r in ordered], label="available predictive certificate")
    ax.axhline(0.0, color="black", linewidth=0.8)
    for r in ordered:
        if int(r["predictive_trigger"]): ax.axvline(float(r["time_s"]), color="tab:orange", alpha=0.2)
        if int(r["certificate_loss"]): ax.axvline(float(r["time_s"]), color="tab:red", alpha=0.2)
    ax.set(xlabel="time (s)", ylabel="rho", title="P5: predictive trigger versus later boundary")
    fig.tight_layout(); fig.savefig(OUT / "P5_trigger_vs_boundary.pdf"); plt.close(fig)

    # P7: horizon/uncertainty trade-off from registered runs.
    groups: dict[tuple[int, float], list[dict[str, str]]] = defaultdict(list)
    for r in steps: groups[(int(r["prediction_horizon"]), float(r["uncertainty_scale"]))].append(r)
    keys = sorted(groups)
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.2))
    axes[0].plot(range(len(keys)), [sum(float(r["tube_width_total"]) for r in groups[k]) / len(groups[k]) for k in keys], "o-")
    axes[1].plot(range(len(keys)), [min(float(r["rho_cert_H"]) for r in groups[k]) for k in keys], "o-")
    axes[2].plot(range(len(keys)), [sum(float(r["end_to_end_ms"]) for r in groups[k]) / len(groups[k]) for k in keys], "o-")
    for ax, title in zip(axes, ("tube width", "minimum rho^H", "runtime ms")): ax.set_title(title); ax.set_xticks(range(len(keys)), [str(k) for k in keys], rotation=45, ha="right")
    fig.tight_layout(); fig.savefig(OUT / "P7_horizon_uncertainty_tradeoff.pdf"); plt.close(fig)

    # P8: empirical runtime distribution; no synthetic samples are allowed.
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    ax.hist([float(r["end_to_end_ms"]) for r in steps], bins=40)
    ax.set(xlabel="end-to-end runtime (ms)", ylabel="count", title="P8: runtime distribution")
    fig.tight_layout(); fig.savefig(OUT / "P8_runtime_distributions.pdf"); plt.close(fig)
    print(f"Generated Predictive P1--P8 from verified logs in {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

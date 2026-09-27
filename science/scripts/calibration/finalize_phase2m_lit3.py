"""Reproduce the Phase 2M-LIT-3 literature closure and frozen config.

This script only maps published data, digitizes two published figures, and
derives model-boundary quantities.  It does not execute M1--M10 or create any
paper result.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "model_parameters"
REGISTRY = DATA / "parameter_provenance.yaml"
SOURCES = DATA / "source_extracts.json"
DIGITIZATION = DATA / "source_points" / "phase2m_lit3_digitization.json"
RESIDUAL = DATA / "derived" / "thermal_model_discrepancy.json"
DERIVED = DATA / "derived" / "phase2m_lit3_parameters.json"
CONFIG = ROOT / "configs" / "model_simulation" / "literature_calibrated_v1.yaml"
FADE_IMAGE = DATA / "source_points" / "ntsb_2002_figure4_page7_crop_300dpi.png"
RETARDER_IMAGE = DATA / "source_points" / "li2026_figure5_retarder_map.png"

SCRIPT = "scripts/calibration/finalize_phase2m_lit3.py"
DIGITIZATION_REL = "data/model_parameters/source_points/phase2m_lit3_digitization.json"
DERIVED_REL = "data/model_parameters/derived/phase2m_lit3_parameters.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def linear(value: float, pixel0: float, value0: float, pixel1: float, value1: float) -> float:
    return value0 + (value - pixel0) * (value1 - value0) / (pixel1 - pixel0)


def fade_digitization() -> dict[str, Any]:
    # Coordinates are relative to the stored 1080 x 930 crop.  Samples avoid
    # vertical grid lines.  y is the midpoint of the manually audited dark
    # boundary-pixel cluster at each x coordinate.
    xs = [105.0, 235.0, 366.0, 500.0, 632.0, 764.0, 898.0, 1003.0]
    quality_y = [321.0, 316.5, 288.5, 288.5, 291.5, 330.5, 387.0, 464.5]
    inferior_y = [360.0, 369.0, 409.0, 458.5, 533.0, 600.5, 647.5, 721.0]
    calibration = {
        "temperature_axis": {"pixel_100F": [91.0, 769.0], "pixel_800F": [1017.0, 769.0]},
        "friction_axis": {"pixel_mu_0": [91.0, 769.0], "pixel_mu_0_6": [91.0, 127.0]},
        "pixel_uncertainty": {"x_px": 2.0, "y_px": 2.0},
    }

    def make_points(ys: list[float]) -> list[dict[str, float]]:
        points = []
        for x, y in zip(xs, ys):
            temp_f = linear(x, 91.0, 100.0, 1017.0, 800.0)
            mu = linear(y, 769.0, 0.0, 127.0, 0.6)
            points.append({
                "pixel_x": x,
                "pixel_y": y,
                "temperature_degF": temp_f,
                "temperature_K": (temp_f - 32.0) * 5.0 / 9.0 + 273.15,
                "coefficient_of_friction": mu,
            })
        return points

    quality = make_points(quality_y)
    inferior = make_points(inferior_y)
    mu_ref = inferior[0]["coefficient_of_friction"]
    running = 1.0
    conservative = []
    for point in inferior:
        running = min(running, point["coefficient_of_friction"] / mu_ref)
        conservative.append({
            "temperature_K": point["temperature_K"],
            "temperature_degF": point["temperature_degF"],
            "mu": point["coefficient_of_friction"],
            "phi": running,
        })
    assert all(a["phi"] >= b["phi"] for a, b in zip(conservative, conservative[1:]))
    return {
        "label": "DIGITIZED_REPRESENTATIVE_HEAVY_TRUCK_S_CAM_FADE",
        "source": {
            "title": "Vehicle Dynamics Simulation Study",
            "authors": "Lawrence E. Jackson",
            "date": "2002-08-16",
            "identifier": "NTSB HWY-01-M-H-25",
            "url": "https" + "://data.ntsb.gov/Docket/Document/docBLOB?FileExtension=.PDF&FileName=Simulation+Study-Master.PDF&ID=40165030",
            "pdf_page": 7,
            "printed_page": 4,
            "figure": 4,
            "figure_title": "Fade Resistance",
            "image_file": FADE_IMAGE.relative_to(ROOT).as_posix(),
            "image_sha256": sha256(FADE_IMAGE),
        },
        "axis_calibration": calibration,
        "raw_digitized_points": {"quality_lining": quality, "inferior_lining": inferior},
        "selected_conservative_curve": conservative,
        "reference_temperature_K": conservative[0]["temperature_K"],
        "reference_mu": mu_ref,
        "scope_note": "Representative published heavy-truck lining data; not claimed material-identical to Reference_Truck.",
    }


def critical_temperature(fade: dict[str, Any]) -> dict[str, float | str]:
    # Pre-registered before any M1--M10 inspection: the hard thermal threshold
    # is the first point where the conservative available friction falls to 80%.
    fraction = 0.80
    curve = fade["selected_conservative_curve"]
    for left, right in zip(curve, curve[1:]):
        if left["phi"] >= fraction >= right["phi"]:
            weight = (fraction - left["phi"]) / (right["phi"] - left["phi"])
            temp_k = left["temperature_K"] + weight * (right["temperature_K"] - left["temperature_K"])
            return {
                "criterion": "first temperature where conservative phi(T) <= 0.80",
                "safety_fraction": fraction,
                "temperature_K": temp_k,
                "temperature_degC": temp_k - 273.15,
                "interpolation": "piecewise linear between adjacent digitized inferior-lining points",
            }
    raise RuntimeError("fade curve does not cross the pre-registered safety fraction")


def retarder_digitization() -> dict[str, Any]:
    # Figure calibration: plot rectangle x=230..1677 maps 0..1800 rpm;
    # y=909..45 maps 0..4000 N m.  The blue 300-kPa trace is the maximum
    # measured pressure-speed-torque curve; centers are audited marker/trace
    # coordinates, including endpoints.
    raw = [
        (230.0, 909.0), (391.0, 889.0), (552.0, 730.0), (633.0, 515.0),
        (713.0, 300.0), (873.0, 120.0), (1034.0, 76.0), (1194.0, 91.0),
        (1355.0, 132.0), (1516.0, 192.0), (1677.0, 265.0),
    ]
    i0 = 5.7
    radius_m = 0.492
    efficiency = 0.97
    continuous_power_w = 300_000.0
    peak_power_w = 450_000.0
    points = []
    for x, y in raw:
        rpm = linear(x, 230.0, 0.0, 1677.0, 1800.0)
        torque = linear(y, 909.0, 0.0, 45.0, 4000.0)
        omega = rpm * 2.0 * math.pi / 60.0
        continuous_torque = torque if omega == 0.0 else min(torque, continuous_power_w / omega)
        peak_torque = torque if omega == 0.0 else min(torque, peak_power_w / omega)
        speed_mps = omega * radius_m / i0
        points.append({
            "pixel_x": x, "pixel_y": y,
            "driveshaft_speed_rpm": rpm,
            "measured_300kPa_torque_Nm": torque,
            "continuous_torque_envelope_Nm": continuous_torque,
            "peak_torque_envelope_Nm": peak_torque,
            "vehicle_speed_mps": speed_mps,
            "continuous_longitudinal_force_N": continuous_torque * i0 * efficiency / radius_m,
            "peak_longitudinal_force_N": peak_torque * i0 * efficiency / radius_m,
        })
    return {
        "label": "DIGITIZED_MEASURED_HYDRAULIC_RETARDER_ENVELOPE",
        "source": {
            "title": "An Energy-Efficient Constant-Speed Downhill Control Approach for Heavy-Duty Electric Trucks with Hydraulic Retarders",
            "authors": "Xuebo Li; Yanli Feng; Shiwei Xu; Yixi Zhang",
            "year": 2026,
            "doi": "10.3390/machines14070814",
            "pdf_page": 9,
            "figure": 5,
            "image_file": RETARDER_IMAGE.relative_to(ROOT).as_posix(),
            "image_sha256": sha256(RETARDER_IMAGE),
        },
        "axis_calibration": {
            "speed_axis": {"pixel_0rpm": [230.0, 909.0], "pixel_1800rpm": [1677.0, 909.0]},
            "torque_axis": {"pixel_0Nm": [230.0, 909.0], "pixel_4000Nm": [230.0, 45.0]},
            "pixel_uncertainty": {"x_px": 2.0, "y_px": 2.0},
        },
        "selected_trace": "300 kPa measured pressure-speed-torque curve",
        "power_limits_W": {"continuous": continuous_power_w, "peak": peak_power_w},
        "drivetrain_mapping": {
            "equations": [
                "omega_ds = v*i0/r_w",
                "F_ret = T_ret* i0 * eta_fd / r_w",
                "T_cont = min(T_300kPa, P_cont/omega_ds)",
            ],
            "final_drive_ratio": i0,
            "final_drive_source": "Li et al. (2026), Table 2",
            "wheel_rolling_radius_m": radius_m,
            "final_drive_efficiency": efficiency,
            "geometry_efficiency_source": "Donkers et al. (2020), Energies 13, 2434, Table 2 and Eq. (25)",
            "scope_note": "Representative cross-source heavy-duty geometry; retarder curve is expressed at driveshaft speed.",
        },
        "raw_digitized_points": points,
    }


def thermal_discrepancy(tcrit_k: float, selected_k_w_per_k: float) -> dict[str, Any]:
    # Match the NHTSA ten-wheel-end aggregate.  Yan and Xu provide a per-drum
    # convection+radiation model.  Two front and eight rear S-cam assemblies
    # are aggregated to the same ten-brake system boundary.
    ambient_k = 298.15
    area_m2 = 0.45
    emissivity = 0.65
    sigma = 5.67e-8
    temperatures_c = [25.0, 50.0, 75.0, 100.0, 125.0, 150.0, 175.0, tcrit_k - 273.15]
    speeds_kmh = [50.0, 55.0, 60.0, 65.0, 70.0]
    samples = []
    for speed_kmh in speeds_kmh:
        speed_mps = speed_kmh / 3.6
        h_front = 0.92 + 0.7 * speed_mps * math.exp(-speed_mps / 328.0)
        h_rear = 0.92 + 0.3 * speed_mps * math.exp(-speed_mps / 328.0)
        h_weighted = (2.0 * h_front + 8.0 * h_rear) / 10.0
        for temperature_c in temperatures_c:
            temperature_k = temperature_c + 273.15
            selected_cooling = selected_k_w_per_k * (temperature_k - ambient_k)
            independent_cooling = 10.0 * area_m2 * (
                h_weighted * (temperature_k - ambient_k)
                + emissivity * sigma * (temperature_k**4 - ambient_k**4)
            )
            residual = selected_cooling - independent_cooling
            samples.append({
                "speed_kmh": speed_kmh,
                "temperature_degC": temperature_c,
                "selected_nhtsa_cooling_W": selected_cooling,
                "independent_yan_xu_cooling_W": independent_cooling,
                "equivalent_q_w_W": residual,
            })
    bound = max(abs(row["equivalent_q_w_W"]) for row in samples)
    return {
        "label": "DERIVED_MODEL_DISCREPANCY_BOUND",
        "nominal_q_w_W": 0.0,
        "comparison_domain": {
            "speed_kmh": [50.0, 70.0],
            "temperature_K": [ambient_k, tcrit_k],
            "grid_speeds_kmh": speeds_kmh,
            "grid_temperatures_degC": temperatures_c,
            "system_boundary": "ten wheel-end S-cam assemblies represented by one aggregate temperature",
        },
        "selected_model": {
            "source": "NHTSA DOT HS 811 367 Appendix C, Eqs. (C2)-(C4)",
            "equation": "Qdot_cool = k_NHTSA*(T-Ta)",
            "k_W_per_K": selected_k_w_per_k,
            "nominal_residual": "q_w=0 because Eq. (C2) contains no additive residual source",
        },
        "independent_model": {
            "source": "Yan and Xu (2018), DOI 10.1155/2018/4587673, Eqs. (9)-(12), Table 1",
            "equations": [
                "h_c=0.92+a*v*exp(-v/328), v in m/s",
                "Qdot=A'[h_c(T-Ta)+epsilon*sigma(T^4-Ta^4)]",
            ],
            "parameters": {"A_prime_m2": area_m2, "epsilon": emissivity, "sigma_W_m2_K4": sigma, "a_front": 0.7, "a_rear": 0.3},
        },
        "residual_definition": "q_w = Qdot_cool,NHTSA - Qdot_cool,YanXu",
        "samples": samples,
        "maximum_absolute_residual_W": bound,
        "uncertainty_semantics": "deterministic model-discrepancy envelope on the registered grid; not an experimental confidence interval",
        "derivation_script": SCRIPT,
    }


def update_sources() -> dict[str, Any]:
    document = json.loads(SOURCES.read_text(encoding="utf-8"))
    document["accessed_on"] = "2026-09-18"
    document["purpose"] = "Traceable source excerpts for Phase 2M-LIT-3 final literature closure; no paper results."
    sources = document["sources"]
    sources.update({
        "NTSB2002": {
            "title": "Vehicle Dynamics Simulation Study", "authors": "Lawrence E. Jackson", "year": 2002,
            "identifier": "NTSB HWY-01-M-H-25", "url": "https" + "://data.ntsb.gov/Docket/Document/docBLOB?FileExtension=.PDF&FileName=Simulation+Study-Master.PDF&ID=40165030",
            "source_quality": "A/C: official NTSB docket study reproducing published heavy-truck lining fade curves",
            "extracts": {"fade_curve": {"value": "quality and inferior lining coefficient of friction versus drum temperature", "location": "PDF p. 7, printed p. 4, Figure 4", "use": "representative heavy-truck S-cam curve; not material-identical"}},
        },
        "LI2026": {
            "title": "An Energy-Efficient Constant-Speed Downhill Control Approach for Heavy-Duty Electric Trucks with Hydraulic Retarders", "authors": "Xuebo Li; Yanli Feng; Shiwei Xu; Yixi Zhang", "year": 2026,
            "identifier": "DOI:10.3390/machines14070814", "url": "https" + "://doi.org/10.3390/machines14070814",
            "source_quality": "D: peer-reviewed heavy-duty electric-truck model with bench-derived retarder map and DiL validation",
            "extracts": {
                "retarder_map": {"value": "measured pressure-speed-torque data at 120/180/240/300 kPa", "location": "p. 9, Figure 5 and Eqs. (12)-(14)"},
                "power_limits_W": {"value": {"continuous": 300000.0, "peak": 450000.0}, "location": "p. 9, Figure 5 discussion"},
                "final_drive_ratio": {"value": 5.7, "location": "p. 5, Table 2"},
                "first_order_retarder_tau_s": {"value": 0.25, "location": "p. 12, offline optimization model", "note": "published simplified-link time constant; not a universal measured actuator constant"},
            },
        },
        "DONKERS2020": {
            "title": "Electric Powertrain Topology Analysis and Design for Heavy-Duty Trucks", "authors": "Ad Donkers et al.", "year": 2020,
            "identifier": "DOI:10.3390/en13102434", "url": "https" + "://doi.org/10.3390/en13102434",
            "source_quality": "D: peer-reviewed heavy-duty-truck powertrain design model",
            "extracts": {
                "wheel_radius_m": {"value": 0.492, "location": "Table 2"},
                "final_drive_efficiency": {"value": 0.97, "location": "Section 3.2, Eq. (25)"},
            },
        },
        "YAN2018": {
            "title": "Prediction Model for Brake-Drum Temperature of Large Trucks on Consecutive Mountain Downgrade Routes Based on Energy Conservation Law", "authors": "Menghua Yan; Jinliang Xu", "year": 2018,
            "identifier": "DOI:10.1155/2018/4587673", "url": "https" + "://doi.org/10.1155/2018/4587673",
            "source_quality": "D: peer-reviewed independent large-truck brake-drum energy model",
            "extracts": {
                "ambient_temperature_degC": {"value": 25.0, "location": "Table 1"},
                "thermal_model": {"value": "convection plus Stefan-Boltzmann radiation", "location": "Eqs. (9)-(12)"},
                "operating_speed_kmh": {"value": [50.0, 55.0, 60.0, 65.0, 70.0], "location": "Table 1"},
                "drum_area_m2": {"value": 0.45, "location": "Table 1"},
                "emissivity": {"value": 0.65, "location": "Table 1"},
            },
        },
    })
    dump(SOURCES, document)
    return document


def completed_record(record: dict[str, Any], **updates: Any) -> None:
    record.update(updates)
    record["resolved"] = True
    record["paper_eligible"] = True


def update_registry(fade: dict[str, Any], tcrit: dict[str, Any], retarder: dict[str, Any], discrepancy: dict[str, Any]) -> dict[str, Any]:
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    # Command slew is neither a plant state nor an independently supported
    # physical limit.  The two records are retired from the required set; the
    # ZOH command can jump at a sample while b and r remain continuous through
    # their published first-order dynamics.
    retired = {"friction_command_slew", "auxiliary_command_slew"}
    registry["parameters"] = [r for r in registry["parameters"] if r["parameter_id"] not in retired]
    registry["required_parameter_count"] = 19
    registry["phase"] = "2M-LIT-3"
    registry["status"] = "READY_FOR_PAPER_CANDIDATE_MODEL_SIMULATION"
    registry["retired_required_constraints"] = {
        "friction_command_slew": "removed: actuator lag is retained; no physical command-slew evidence",
        "auxiliary_command_slew": "removed: actuator lag is retained; no physical command-slew evidence",
    }
    by_id = {item["parameter_id"]: item for item in registry["parameters"]}
    for item in registry["parameters"]:
        if item["resolved"]:
            item["paper_eligible"] = True

    completed_record(by_id["auxiliary_actuator_lag"],
        nominal_value=0.25, uncertainty_lower=0.25, uncertainty_upper=0.25,
        uncertainty_status="published_first_order_model_mapping_not_CI", provenance_status="DERIVED",
        source_title="An Energy-Efficient Constant-Speed Downhill Control Approach for Heavy-Duty Electric Trucks with Hydraulic Retarders",
        authors="Xuebo Li; Yanli Feng; Shiwei Xu; Yixi Zhang", year=2026,
        **{"DOI / manufacturer document ID": "DOI:10.3390/machines14070814", "page/table/figure/equation": "p. 12, simplified first-order pneumatic-control link"},
        original_value={"published_first_order_time_constant_s": 0.25}, original_unit="s",
        conversion_to_SI="No unit conversion; mapped to repository tau_a. Under the stated 2% four-time-constant convention the implied settling time is 1.0 s.",
        derivation_script=SCRIPT, digitization_file=None, identification_script=None, cross_source=False,
        physical_scope="published hydraulic-retarder pneumatic command-to-torque first-order approximation",
        validation_scope="representative model mapping, not a universal measured time constant", source_id="LI2026",
        notes="The source explicitly calls 0.25 s a simplified-link time constant. Total closed-loop settling time is not equated to tau_a.")

    continuous_forces = [p["continuous_longitudinal_force_N"] for p in retarder["raw_digitized_points"]]
    completed_record(by_id["auxiliary_speed_gear_power_envelope"],
        nominal_value={"map_file": DIGITIZATION_REL, "maximum_continuous_force_N": max(continuous_forces)},
        uncertainty_lower=0.0, uncertainty_upper=max(continuous_forces), uncertainty_status="digitization_and_cross_source_model_mapping_not_CI",
        provenance_status="DIGITIZED", source_title="Measured hydraulic-retarder map plus documented heavy-duty drivetrain geometry",
        authors="Xuebo Li; Yanli Feng; Shiwei Xu; Yixi Zhang; Ad Donkers et al.", year="2026; 2020",
        **{"DOI / manufacturer document ID": "DOI:10.3390/machines14070814; DOI:10.3390/en13102434", "page/table/figure/equation": "Li Fig. 5/Table 2; Donkers Table 2/Eq. (25)"},
        original_value={"continuous_power_W": 300000.0, "peak_power_W": 450000.0, "final_drive_ratio": 5.7, "wheel_radius_m": 0.492, "final_drive_efficiency": 0.97},
        original_unit="W,1,m", conversion_to_SI="F=T_ds*i0*eta_fd/r_w; omega_ds=v*i0/r_w; continuous and peak power caps applied before force conversion",
        derivation_script=SCRIPT, digitization_file=DIGITIZATION_REL, identification_script=None, cross_source=True,
        physical_scope="representative hydraulic retarder at transmission output/driveshaft with a heavy-duty final drive",
        validation_scope="0-1800 rpm digitized map; continuous 300 kW for long descent; 450 kW peak retained separately", source_id="LI2026+DONKERS2020",
        notes="The installation is a traceable representative cross-source mapping, not a claim that the sources describe one identical vehicle.")

    completed_record(by_id["braking_to_thermal_conversion"],
        nominal_value=1.0, uncertainty_lower=1.0, uncertainty_upper=1.0,
        uncertainty_status="exact_model_boundary_mapping_not_CI", provenance_status="DERIVED",
        source_title="Study of Heavy Truck S-Cam, Enhanced S-Cam, and Air Disc Brake Models Using NADS",
        authors="M. Kamel Salaani; Gary J. Heydinger; Paul A. Grygier; Chris Schwarz; Tim Brown", year=2010,
        **{"DOI / manufacturer document ID": "DOT HS 811 367", "page/table/figure/equation": "Appendix C, pp. 67-68, Eqs. (C2)-(C4)"},
        original_value={"thermal_input": "4.628*Trq*omega without an additional partition coefficient"}, original_unit="ft lbf rad/s",
        conversion_to_SI="eta=1 when b*v, C and k use the same aggregate brake-system boundary",
        derivation_script=SCRIPT, digitization_file=None, identification_script=None, cross_source=False,
        physical_scope="same aggregate brake mechanical-power and thermal-state boundary as NHTSA Appendix C",
        validation_scope="model mapping only", source_id="NHTSA2010",
        notes="eta=1 is not described as a universal measured physical efficiency.")

    completed_record(by_id["critical_brake_temperature"],
        nominal_value=tcrit["temperature_K"], uncertainty_lower=tcrit["temperature_K"], uncertainty_upper=tcrit["temperature_K"],
        uncertainty_status="pre_registered_80_percent_capability_criterion_not_CI", provenance_status="DERIVED",
        source_title="Vehicle Dynamics Simulation Study", authors="Lawrence E. Jackson", year=2002,
        **{"DOI / manufacturer document ID": "NTSB HWY-01-M-H-25", "page/table/figure/equation": "PDF p. 7, printed p. 4, Figure 4"},
        original_value={"criterion": tcrit["criterion"], "temperature_degC": tcrit["temperature_degC"]}, original_unit="degF curve -> K",
        conversion_to_SI="piecewise-linear crossing of phi=0.80; K=(degF-32)*5/9+273.15",
        derivation_script=SCRIPT, digitization_file=DIGITIZATION_REL, identification_script=None, cross_source=False,
        physical_scope="representative inferior heavy-truck lining fade curve",
        validation_scope="pre-registered before M1-M10; threshold sensitivity is reserved for later", source_id="NTSB2002",
        notes="This is a declared capability criterion, not a material failure temperature.")

    phi_values = [p["phi"] for p in fade["selected_conservative_curve"]]
    completed_record(by_id["thermal_fade_curve"],
        nominal_value={"label": fade["label"], "file": DIGITIZATION_REL}, uncertainty_lower=min(phi_values), uncertainty_upper=max(phi_values),
        uncertainty_status="digitization_pixel_bound_not_CI", provenance_status="DIGITIZED",
        source_title="Vehicle Dynamics Simulation Study", authors="Lawrence E. Jackson", year=2002,
        **{"DOI / manufacturer document ID": "NTSB HWY-01-M-H-25", "page/table/figure/equation": "PDF p. 7, printed p. 4, Figure 4"},
        original_value="inferior-lining coefficient of friction versus drum temperature", original_unit="degF,1",
        conversion_to_SI="T_K=(T_F-32)*5/9+273.15; phi(T)=mu(T)/mu(T_ref); cumulative minimum enforces conservative monotonicity",
        derivation_script=SCRIPT, digitization_file=DIGITIZATION_REL, identification_script=None, cross_source=False,
        physical_scope="representative heavy-truck S-cam lining; not Reference_Truck material identity",
        validation_scope="digitized 110.6-789.4 degF source domain", source_id="NTSB2002",
        notes=fade["label"])

    completed_record(by_id["ambient_operating_envelope"],
        nominal_value=298.15, uncertainty_lower=298.15, uncertainty_upper=298.15,
        uncertainty_status="literature_defined_nominal_baseline_not_CI", provenance_status="DIRECT",
        source_title="Prediction Model for Brake-Drum Temperature of Large Trucks on Consecutive Mountain Downgrade Routes Based on Energy Conservation Law",
        authors="Menghua Yan; Jinliang Xu", year=2018,
        **{"DOI / manufacturer document ID": "DOI:10.1155/2018/4587673", "page/table/figure/equation": "Table 1"},
        original_value=25.0, original_unit="degC", conversion_to_SI="K=degC+273.15",
        derivation_script=None, digitization_file=None, identification_script=None, cross_source=False,
        physical_scope="nominal large-truck downhill thermal benchmark ambient",
        validation_scope="25 degC baseline only; hot/cold ambient are later OOD sensitivity cases", source_id="YAN2018",
        notes="A statistical ambient confidence interval is not asserted.")

    bound = discrepancy["maximum_absolute_residual_W"]
    completed_record(by_id["residual_heat_bound"],
        nominal_value=0.0, uncertainty_lower=-bound, uncertainty_upper=bound,
        uncertainty_status="DERIVED_MODEL_DISCREPANCY_BOUND_not_experimental_CI", provenance_status="DERIVED",
        source_title="NHTSA Appendix-C lumped model compared with Yan-Xu large-truck brake-drum model",
        authors="M. Kamel Salaani et al.; Menghua Yan; Jinliang Xu", year="2010; 2018",
        **{"DOI / manufacturer document ID": "DOT HS 811 367; DOI:10.1155/2018/4587673", "page/table/figure/equation": "NHTSA Eqs. (C2)-(C4); Yan-Xu Eqs. (9)-(12), Table 1"},
        original_value={"nominal_q_w_W": 0.0, "maximum_absolute_residual_W": bound}, original_unit="W",
        conversion_to_SI="q_w=Qdot_cool,NHTSA-Qdot_cool,YanXu over the registered grid",
        derivation_script=SCRIPT, digitization_file=None, identification_script=None, cross_source=True,
        physical_scope="ten-wheel-end aggregate temperature state over 50-70 km/h and 25 degC to Tcrit",
        validation_scope="deterministic model discrepancy envelope; not an experimental confidence interval", source_id="NHTSA2010+YAN2018",
        notes="Nominal q_w=0 follows the selected NHTSA base equation; the nonzero robust bound covers the independent-model discrepancy.")

    assert len(registry["parameters"]) == 19
    assert all(item["resolved"] and item["provenance_status"] != "ASSUMED" for item in registry["parameters"])
    dump(REGISTRY, registry)
    return registry


def write_config(registry: dict[str, Any], sources: dict[str, Any], derived: dict[str, Any]) -> None:
    key_files = [
        ROOT / "src" / "safety_core.py",
        ROOT / "src" / "safety" / "conflict_reserve.py",
        ROOT / "src" / "controllers" / "integrated_safety_controller.py",
        ROOT / "manuscript" / "sections" / "04_dynamics_and_safety.tex",
        ROOT / "manuscript" / "sections" / "06_analysis.tex",
    ]
    file_hashes = {path.relative_to(ROOT).as_posix(): sha256(path) for path in key_files}
    snapshot = hashlib.sha256(json.dumps(file_hashes, sort_keys=True).encode()).hexdigest()
    parameter_hashes = {
        item["parameter_id"]: hashlib.sha256(json.dumps(item, sort_keys=True).encode()).hexdigest()
        for item in registry["parameters"]
    }
    config = {
        "schema_version": 2,
        "phase": "2M-LIT-3",
        "mode": "LITERATURE_CALIBRATED",
        "configuration_frozen": True,
        "paper_eligible_requested": False,
        "status": "READY_FOR_PAPER_CANDIDATE_MODEL_SIMULATION",
        "parameter_registry": REGISTRY.relative_to(ROOT).as_posix(),
        "provenance_sha256": sha256(REGISTRY),
        "source_extracts_sha256": sha256(SOURCES),
        "parameter_hashes": parameter_hashes,
        "repository_state_id": snapshot,
        "repository_file_hashes": file_hashes,
        "constraint_partition": {
            "friction_command_slew_hard_constraint": False,
            "auxiliary_command_slew_hard_constraint": False,
            "friction_actuator_lag_retained": True,
            "auxiliary_actuator_lag_retained": True,
            "command_hold": "zero-order hold between controller samples",
        },
        "physical_model": derived,
        "paper_eligible_results_generated": False,
        "prohibited_in_this_phase": "final M1-M10 execution",
    }
    dump(CONFIG, config)


def main() -> int:
    for path in (FADE_IMAGE, RETARDER_IMAGE, REGISTRY, SOURCES):
        if not path.is_file():
            raise FileNotFoundError(path)
    fade = fade_digitization()
    tcrit = critical_temperature(fade)
    retarder = retarder_digitization()
    digitization = {"schema_version": 1, "paper_result": False, "fade": fade, "retarder": retarder}
    dump(DIGITIZATION, digitization)

    old_derived = json.loads((DATA / "derived" / "reference_truck_parameters.json").read_text(encoding="utf-8"))
    selected_k = old_derived["derivations"]["cooling_coefficient_w_per_k"]["nominal_at_50_kmh"]
    discrepancy = thermal_discrepancy(float(tcrit["temperature_K"]), selected_k)
    dump(RESIDUAL, discrepancy)
    discrepancy["artifact_sha256"] = sha256(RESIDUAL)
    derived = {
        "artifact_type": "Phase 2M-LIT-3 literature and model-mapping closure",
        "phase": "2M-LIT-3",
        "paper_eligible_result": False,
        "fade_curve": {"file": DIGITIZATION_REL, "label": fade["label"]},
        "critical_temperature": tcrit,
        "braking_to_thermal_eta": {"value": 1.0, "classification": "DERIVED MODEL-MAPPING"},
        "ambient_temperature_K": 298.15,
        "thermal_model_discrepancy": {"file": RESIDUAL.relative_to(ROOT).as_posix(), "maximum_absolute_residual_W": discrepancy["maximum_absolute_residual_W"], "sha256": discrepancy["artifact_sha256"]},
        "auxiliary_lag": {"tau_a_s": 0.25, "mapping": "published first-order retarder pneumatic-control approximation", "implied_2_percent_settling_s": 1.0},
        "auxiliary_envelope": {"file": DIGITIZATION_REL, "continuous_power_W": 300000.0, "peak_power_W": 450000.0},
        "slew_decisions": {"friction": "removed from physical hard constraints", "auxiliary": "removed from physical hard constraints"},
    }
    dump(DERIVED, derived)
    sources = update_sources()
    registry = update_registry(fade, tcrit, retarder, discrepancy)
    write_config(registry, sources, derived)
    print(CONFIG.relative_to(ROOT).as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

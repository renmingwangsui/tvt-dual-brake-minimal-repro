"""Fail-closed provenance checks for Phase 2M model-simulation artifacts."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class EligibilityDecision:
    eligible: bool
    reasons: tuple[str, ...]


def evaluate_paper_eligibility(
    config: dict[str, Any],
    registry: dict[str, Any],
    manifest: dict[str, Any] | None = None,
) -> EligibilityDecision:
    """Return eligibility only when every affirmative prerequisite is recorded.

    Missing fields are failures.  This deliberately prevents a DEBUG artifact, a
    template, or a partially sourced physical model from being promoted by
    changing a single flag.
    """
    reasons: list[str] = []
    if config.get("mode") != "LITERATURE_CALIBRATED":
        reasons.append("mode_is_not_literature_calibrated")
    if config.get("configuration_frozen") is not True:
        reasons.append("configuration_not_frozen")
    if config.get("paper_eligible_requested") is not True:
        reasons.append("paper_eligibility_not_requested")

    parameters = registry.get("parameters")
    if not isinstance(parameters, list) or not parameters:
        reasons.append("parameter_registry_missing_or_empty")
    else:
        for item in parameters:
            name = str(item.get("name", "unnamed"))
            if item.get("source_type") not in {"peer_reviewed_literature", "verified_measurement"}:
                reasons.append(f"{name}:unacceptable_source_type")
            if not item.get("source") or not item.get("source_location"):
                reasons.append(f"{name}:source_traceability_incomplete")
            if item.get("calibration_status") != "accepted":
                reasons.append(f"{name}:calibration_not_accepted")
            if item.get("paper_eligible") is not True:
                reasons.append(f"{name}:parameter_not_paper_eligible")

    if manifest is None:
        reasons.append("result_manifest_missing")
    else:
        if manifest.get("mode") != "LITERATURE_CALIBRATED":
            reasons.append("manifest_mode_mismatch")
        if manifest.get("paper_eligible") is not True:
            reasons.append("manifest_not_paper_eligible")
        if manifest.get("config_hash") != config.get("config_hash"):
            reasons.append("manifest_config_hash_unverified")
        if manifest.get("source_provenance_hash") != registry.get("registry_hash"):
            reasons.append("manifest_provenance_hash_unverified")
        if manifest.get("validation_status") != "PASS":
            reasons.append("manifest_validation_not_passed")
    return EligibilityDecision(not reasons, tuple(reasons))


def require_paper_eligible(
    config: dict[str, Any], registry: dict[str, Any], manifest: dict[str, Any]
) -> None:
    decision = evaluate_paper_eligibility(config, registry, manifest)
    if not decision.eligible:
        raise ValueError("publication gate rejected artifact: " + ", ".join(decision.reasons))

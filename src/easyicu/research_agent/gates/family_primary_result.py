"""Reconcile causal/survival headline contracts with registered result data.

The Planner owns the scientific contract in ``FamilyPrimaryResultRequirement``;
this gate owns the narrow execution check.  It reads the one exact registered
table named by that contract, never a chart artist or a free-text summary.
"""

from __future__ import annotations

from ..canonical_json import sha256_file as _sha256_file

import csv
import json
import math
import re
from pathlib import Path
from typing import Any, List, Mapping

from pydantic import ValidationError

from ..contracts.family_primary import FamilyPrimaryResultRequirement
from ..contracts.runtime import ValidationFinding
from ..schema import (
    SURVIVAL_ANALYSIS_RECEIPT_PRODUCT,
    AnalysisPlan,
    AnalysisStep,
    ResearchContext,
    SurvivalAnalysisReceipt,
)
from ..contracts.survival import (
    SURVIVAL_PH_DIAGNOSTIC_PRODUCT,
    SURVIVAL_PRIMARY_OWNER,
)
from ..contracts.survival_execution import SURVIVAL_PRIMARY_ANALYSIS_KIND
from ..contracts.time_units import canonical_time_unit


_EFFECT_COLUMNS = (
    "effect_estimate",
    "estimate",
    "point_estimate",
    "adjusted_effect",
    "risk_difference",
    "risk_ratio",
    "odds_ratio",
    "hazard_ratio",
)


def _normalise(value: Any) -> str:
    token = re.sub(r"[^a-z0-9]+", "_", str(value or "").lower()).strip("_")
    return {"hr": "hazard_ratio", "or": "odds_ratio", "rr": "risk_ratio"}.get(
        token, token
    )


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _finding(step: AnalysisStep, issue: str, **detail: Any) -> ValidationFinding:
    return ValidationFinding(
        validator="family_primary_result_contract",
        severity="error",
        message=(
            f"Step {step.step_id} did not reconcile its causal/survival primary "
            f"result contract: {issue}."
        ),
        detail={"issue": issue, "step_id": step.step_id, **detail},
    )


def _materialised_path(
    *,
    out_dir: Path,
    raw_path: Any,
    expected_suffix: str,
) -> Path | None:
    """Return one registered child path without allowing output-dir escape."""

    if not isinstance(raw_path, str) or not raw_path.strip():
        return None
    try:
        candidate = (Path(out_dir) / raw_path).resolve()
        candidate.relative_to(Path(out_dir).resolve())
    except ValueError:
        return None
    if not candidate.is_file() or candidate.suffix.lower() != expected_suffix:
        return None
    return candidate


def _survival_receipt_findings(
    *,
    step: AnalysisStep,
    requirement: FamilyPrimaryResultRequirement,
    context: ResearchContext,
    step_summary: Mapping[str, Any],
    output_files: Mapping[str, Any],
    out_dir: Path,
) -> List[ValidationFinding]:
    """Reconcile execution-owned survival design with plan and endpoint."""

    if (
        step_summary.get("deterministic_standard_analysis")
        != SURVIVAL_PRIMARY_ANALYSIS_KIND
        or step_summary.get("receipt_issuer") != SURVIVAL_PRIMARY_OWNER
    ):
        return [_finding(step, "survival_primary_not_host_executed")]

    endpoint = context.endpoint
    if endpoint is None or endpoint.kind != "time_to_event":
        return [_finding(step, "survival_endpoint_not_declared")]
    if (
        requirement.time_origin != endpoint.time_origin
        or requirement.time_column != endpoint.time_column
        or requirement.event_column != endpoint.event_column
    ):
        return [
            _finding(
                step,
                "survival_requirement_endpoint_mismatch",
                requirement_time_origin=requirement.time_origin,
                requirement_time_column=requirement.time_column,
                requirement_event_column=requirement.event_column,
                endpoint_time_origin=endpoint.time_origin,
                endpoint_time_column=endpoint.time_column,
                endpoint_event_column=endpoint.event_column,
            )
        ]
    time_descriptor = context.variable(str(requirement.time_column))
    authoritative_time_unit = canonical_time_unit(
        getattr(time_descriptor, "unit", None)
    )
    if authoritative_time_unit is None:
        return [_finding(step, "survival_time_unit_authority_missing")]
    if authoritative_time_unit != requirement.time_unit:
        return [
            _finding(
                step,
                "survival_time_unit_authority_mismatch",
                authoritative_unit=authoritative_time_unit,
                requirement_unit=requirement.time_unit,
            )
        ]

    raw_path = output_files.get(SURVIVAL_ANALYSIS_RECEIPT_PRODUCT)
    if not isinstance(raw_path, str) or not raw_path.strip():
        return [
            _finding(
                step,
                "survival_execution_receipt_unregistered",
                expected_receipt_product=SURVIVAL_ANALYSIS_RECEIPT_PRODUCT,
            )
        ]
    receipt_path = _materialised_path(
        out_dir=out_dir,
        raw_path=raw_path,
        expected_suffix=".json",
    )
    if receipt_path is None:
        return [
            _finding(
                step,
                "survival_execution_receipt_not_a_materialised_json",
                path=str(raw_path),
            )
        ]
    try:
        receipt = SurvivalAnalysisReceipt.model_validate_json(
            receipt_path.read_text(encoding="utf-8")
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValidationError) as exc:
        return [
            _finding(
                step,
                "survival_execution_receipt_invalid",
                error=type(exc).__name__,
            )
        ]

    result_raw_path = output_files.get(requirement.expected_result_product)
    result_path = _materialised_path(
        out_dir=out_dir,
        raw_path=result_raw_path,
        expected_suffix=".csv",
    )
    ph_raw_path = output_files.get(SURVIVAL_PH_DIAGNOSTIC_PRODUCT)
    ph_path = _materialised_path(
        out_dir=out_dir,
        raw_path=ph_raw_path,
        expected_suffix=".csv",
    )
    if result_path is None or ph_path is None:
        return [
            _finding(
                step,
                "survival_host_evidence_unregistered",
                result_path=result_raw_path,
                ph_diagnostic_path=ph_raw_path,
            )
        ]

    expected_values = {
        "result_product": requirement.expected_result_product,
        "input_product": requirement.input_product,
        "exposure_source": requirement.exposure_source,
        "outcome": requirement.outcome,
        "effect_scale": requirement.effect_scale,
        "analysis_population": requirement.population,
        "time_origin": requirement.time_origin,
        "time_column": requirement.time_column,
        "time_unit": requirement.time_unit,
        "event_column": requirement.event_column,
        "event_value": requirement.event_value,
        "censor_value": 0,
        "event_definition": requirement.event_definition,
        "censoring_strategy": requirement.censoring_strategy,
        "competing_risk_strategy": requirement.competing_risk_strategy,
        "time_horizon": requirement.time_horizon,
        "time_horizon_value": requirement.time_horizon_value,
        "estimator": requirement.estimator,
        "effect_measure": requirement.effect_measure,
        "covariates": list(requirement.covariates or ()),
        "model_terms": list(requirement.model_terms or ()),
        "ph_diagnostic_product": SURVIVAL_PH_DIAGNOSTIC_PRODUCT,
        "time_unit_authority": "research_context_concept_descriptor",
        "proportional_hazards_alpha": requirement.proportional_hazards_alpha,
        "proportional_hazards_policy": requirement.proportional_hazards_policy,
    }
    mismatches = {
        field: {"expected": expected, "reported": getattr(receipt, field)}
        for field, expected in expected_values.items()
        if getattr(receipt, field) != expected
    }
    if requirement.proportional_hazards_diagnostic is not None and (
        receipt.proportional_hazards_diagnostic
        != requirement.proportional_hazards_diagnostic
    ):
        mismatches["proportional_hazards_diagnostic"] = {
            "expected": requirement.proportional_hazards_diagnostic,
            "reported": receipt.proportional_hazards_diagnostic,
        }
    for field in ("input_evidence_id", "input_sha256"):
        if getattr(receipt, field) != step_summary.get(field):
            mismatches[field] = {
                "expected": step_summary.get(field),
                "reported": getattr(receipt, field),
            }
    for field in (
        "design_columns",
        "exposure_design_column",
        "proportional_hazards_status",
        "paper_authorization_allowed",
    ):
        if getattr(receipt, field) != step_summary.get(field):
            mismatches[field] = {
                "expected": step_summary.get(field),
                "reported": getattr(receipt, field),
            }
    if mismatches:
        return [
            _finding(
                step,
                "survival_execution_receipt_contract_mismatch",
                mismatches=mismatches,
            )
        ]
    digest_mismatches = {}
    if receipt.result_sha256 != _sha256_file(result_path):
        digest_mismatches["result_sha256"] = receipt.result_sha256
    if receipt.ph_diagnostic_sha256 != _sha256_file(ph_path):
        digest_mismatches["ph_diagnostic_sha256"] = receipt.ph_diagnostic_sha256
    if digest_mismatches:
        return [
            _finding(
                step,
                "survival_host_receipt_binding_mismatch",
                mismatches=digest_mismatches,
            )
        ]
    if not receipt.paper_authorization_allowed:
        return [
            _finding(
                step,
                "survival_ph_policy_blocks_paper_authorization",
                proportional_hazards_p_value=receipt.proportional_hazards_p_value,
                proportional_hazards_alpha=receipt.proportional_hazards_alpha,
                proportional_hazards_policy=receipt.proportional_hazards_policy,
                proportional_hazards_status=receipt.proportional_hazards_status,
            )
        ]
    return []


def family_primary_result_reconciliation_findings(
    *,
    step: AnalysisStep,
    plan: AnalysisPlan,
    context: ResearchContext,
    step_summary: Mapping[str, Any],
    out_dir: Path,
) -> List[ValidationFinding]:
    """Require the planned causal/survival headline to exist as source data."""

    requirement = step.family_primary_result_requirement
    if requirement is None:
        return []
    if requirement.analysis_family != plan.analysis_type:
        return [
            _finding(
                step,
                "analysis_family_mismatch",
                planned_family=plan.analysis_type,
                requirement_family=requirement.analysis_family,
            )
        ]
    if (
        requirement.exposure_source != str(context.primary_exposure or "").strip()
        or requirement.outcome != str(context.target_outcome or "").strip()
    ):
        return [
            _finding(
                step,
                "research_context_binding_mismatch",
                expected_exposure=context.primary_exposure,
                expected_outcome=context.target_outcome,
                reported_exposure=requirement.exposure_source,
                reported_outcome=requirement.outcome,
            )
        ]
    output_files = step_summary.get("output_files")
    if not isinstance(output_files, Mapping):
        return [_finding(step, "registered_result_table_missing")]
    raw_path = output_files.get(requirement.expected_result_product)
    if not isinstance(raw_path, str) or not raw_path.strip():
        return [
            _finding(
                step,
                "expected_result_product_unregistered",
                expected_result_product=requirement.expected_result_product,
            )
        ]
    result_path = _materialised_path(
        out_dir=out_dir,
        raw_path=raw_path,
        expected_suffix=".csv",
    )
    if result_path is None:
        try:
            candidate = (Path(out_dir) / raw_path).resolve()
            candidate.relative_to(Path(out_dir).resolve())
        except ValueError:
            return [_finding(step, "result_table_path_escapes_step_output")]
        return [
            _finding(
                step,
                "result_table_not_a_materialised_csv",
                path=str(raw_path),
            )
        ]
    try:
        with result_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            rows = [
                {_normalise(column): value for column, value in row.items()}
                for row in reader
            ]
            columns = {_normalise(column) for column in (reader.fieldnames or [])}
    except (OSError, UnicodeError, csv.Error) as exc:
        return [_finding(step, "result_table_unreadable", error=type(exc).__name__)]
    required_columns = {"exposure_source", "outcome", "effect_scale"}
    missing_columns = sorted(required_columns - columns)
    if missing_columns:
        return [
            _finding(
                step, "result_table_missing_contract_columns", fields=missing_columns
            )
        ]
    if not rows:
        return [_finding(step, "result_table_has_no_rows")]
    matching = [
        row
        for row in rows
        if _normalise(row.get("exposure_source"))
        == _normalise(requirement.exposure_source)
        and _normalise(row.get("outcome")) == _normalise(requirement.outcome)
        and _normalise(row.get("effect_scale")) == _normalise(requirement.effect_scale)
    ]
    if not matching:
        return [
            _finding(
                step,
                "result_table_has_no_matching_primary_effect",
                exposure_source=requirement.exposure_source,
                outcome=requirement.outcome,
                effect_scale=requirement.effect_scale,
            )
        ]
    for row in matching:
        has_effect = any(_finite(row.get(column)) for column in _EFFECT_COLUMNS)
        has_interval = _finite(row.get("ci_low")) and _finite(row.get("ci_high"))
        has_standard_error = _finite(row.get("standard_error"))
        if has_effect and (has_interval or has_standard_error):
            if requirement.analysis_family == "survival":
                return _survival_receipt_findings(
                    step=step,
                    requirement=requirement,
                    context=context,
                    step_summary=step_summary,
                    output_files=output_files,
                    out_dir=out_dir,
                )
            return []
    return [
        _finding(
            step,
            "matching_primary_effect_lacks_uncertainty",
            required_effect_columns=list(_EFFECT_COLUMNS),
            accepted_uncertainty="ci_low+ci_high or standard_error",
        )
    ]


__all__ = ["family_primary_result_reconciliation_findings"]

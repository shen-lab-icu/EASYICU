"""Reconcile causal/survival headline contracts with registered result data.

The Planner owns the scientific contract in ``FamilyPrimaryResultRequirement``;
this gate owns the narrow execution check.  It reads the one exact registered
table named by that contract, never a chart artist or a free-text summary.
"""

from __future__ import annotations

import csv
import math
import re
from pathlib import Path
from typing import Any, List, Mapping

from ..contracts.runtime import ValidationFinding
from ..schema import AnalysisPlan, AnalysisStep, ResearchContext


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
    try:
        result_path = (Path(out_dir) / raw_path).resolve()
        result_path.relative_to(Path(out_dir).resolve())
    except ValueError:
        return [_finding(step, "result_table_path_escapes_step_output")]
    if not result_path.is_file() or result_path.suffix.lower() != ".csv":
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
        return [_finding(step, "result_table_missing_contract_columns", fields=missing_columns)]
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

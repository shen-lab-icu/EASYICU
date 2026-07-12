"""Typed execution contract for plan-locked robustness results.

This module verifies execution evidence only. Specification declarations and
cohort-membership replay remain owned by the execute orchestrator.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from .schema import ResearchContext


ROBUSTNESS_RESULT_REQUIRED_TEXT_FIELDS = (
    "status",
    "model_id",
    "outcome_concept_id",
    "model_family",
    "effect_scale",
    "comparison",
    "coefficient_term",
    "analysis_set",
    "baseline_missing_policy",
    "fit_status",
    "interval_method",
)
ROBUSTNESS_RESULT_REQUIRED_BOOLEAN_FIELDS = (
    "converged",
    "penalized",
    "reportable",
)
ROBUSTNESS_RESULT_REQUIRED_FIELDS = (
    "spec_id",
    "axis",
    *ROBUSTNESS_RESULT_REQUIRED_TEXT_FIELDS,
    *ROBUSTNESS_RESULT_REQUIRED_BOOLEAN_FIELDS,
    "n",
    "point_estimate",
    "ci_low",
    "ci_high",
)
ROBUSTNESS_MODEL_RESULT_REQUIRED_COLUMNS = frozenset(
    {
        "model_id",
        "outcome",
        "model_family",
        "effect_scale",
        "n",
        "fit_status",
        "converged",
        "penalized",
        "interval_method",
    }
)
ROBUSTNESS_MODEL_RESULT_IDENTIFIER_COLUMNS = frozenset(
    {"definition_id", "spec_id"}
)
ROBUSTNESS_COEFFICIENT_RESULT_REQUIRED_COLUMNS = frozenset(
    {
        "model_id",
        "outcome",
        "term",
        "term_role",
        "effect_scale",
        "estimate",
        "ci_low",
        "ci_high",
    }
)
ROBUSTNESS_COHORT_MEMBERSHIP_ALIASES = {
    "universe_n": ("universe_n",),
    "variant_membership_n": (
        "variant_membership_n",
        "retained_n",
        "cohort_n",
        "membership_n",
    ),
    "inflow_n": (
        "inflow_n",
        "entered_n",
        "entering_relative_to_primary_n",
        "enter_n",
    ),
    "outflow_n": (
        "outflow_n",
        "left_primary_n",
        "leaving_relative_to_primary_n",
        "leave_n",
    ),
    "overlap_n": ("overlap_n", "overlap_with_primary_n"),
}

ROBUSTNESS_EXECUTION_CONTRACT_GUIDANCE = (
    "CANONICAL ROBUSTNESS EXECUTION CONTRACT: write exactly one "
    "step_summary['robustness_rows'] row for every locked spec_id. Every row "
    "must contain these keys: "
    + ", ".join(ROBUSTNESS_RESULT_REQUIRED_FIELDS)
    + ". n is the analytic fitted-model N, never a cohort-membership or "
    "retained-row count. ci_low and ci_high may both be null only for a "
    "penalized, non-reportable point-only fit with "
    "interval_method='unavailable'. For an outcome-axis spec, "
    "applied_outcome_override must exactly equal the locked outcome_override. "
    "For a missing-axis spec, missing_strategy must exactly equal the locked "
    "missing_override.strategy. For a cohort-axis spec, report universe_n, "
    "variant_membership_n (retained_n is an accepted alias), inflow_n, "
    "outflow_n, and overlap_n; all values must match deterministic replay on "
    "EASYICU_UNIVERSE_PARQUET. Also emit one unique schema-identifiable model "
    "result CSV with columns "
    + ", ".join(sorted(ROBUSTNESS_MODEL_RESULT_REQUIRED_COLUMNS))
    + " plus definition_id or spec_id, and one unique coefficient CSV with "
    "columns "
    + ", ".join(sorted(ROBUSTNESS_COEFFICIENT_RESULT_REQUIRED_COLUMNS))
    + ". The robustness row, model row, and exposure coefficient row must "
    "agree exactly on identifiers, outcome, model family, effect scale, term, "
    "analytic n, point estimate, interval, fit status, and penalty/convergence "
    "metadata. Specification or membership declarations are not execution "
    "evidence."
)


def _nonnegative_integral_value(value: Any) -> Optional[int]:
    if value is None or isinstance(value, bool):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not numeric.is_integer() or numeric < 0:
        return None
    return int(numeric)


def _normalise_result_token(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower()).strip(
        "_"
    )


def _finite_result_number(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _result_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    token = str(value if value is not None else "").strip().lower()
    if token in {"1", "true", "yes"}:
        return True
    if token in {"0", "false", "no"}:
        return False
    return None


def _unique_structured_csv(
    *,
    out_dir: Path,
    required_columns: set[str],
    required_any_columns: Optional[set[str]] = None,
) -> Tuple[Optional[Any], List[str]]:
    """Load one unambiguous result table by schema, never by case tokens."""

    try:
        import pandas as pd  # type: ignore
    except Exception as exc:
        return None, [f"pandas_unavailable:{exc}"]
    candidates: List[Tuple[Path, Any]] = []
    root = Path(out_dir).resolve()
    for path in sorted(root.glob("*.csv")):
        try:
            resolved = path.resolve(strict=True)
            resolved.relative_to(root)
            frame = pd.read_csv(resolved)
        except Exception:
            continue
        columns = set(str(column) for column in frame.columns)
        if required_columns.issubset(columns) and (
            not required_any_columns or bool(required_any_columns.intersection(columns))
        ):
            candidates.append((resolved, frame))
    if len(candidates) != 1:
        return None, [
            "structured_table_ambiguous"
            if candidates
            else "structured_table_missing",
            *[path.name for path, _frame in candidates],
        ]
    return candidates[0][1], []


def _executed_robustness_result_issues(
    *,
    locked_by_id: Mapping[str, Dict[str, Any]],
    step_summary: Mapping[str, Any],
    out_dir: Path,
    context: Optional[ResearchContext],
) -> List[Dict[str, Any]]:
    """Bind each locked variant to one fitted, typed, cross-checked result.

    Specification and membership tables prove what was requested and which
    rows moved.  They do not prove that an estimand was executed.  Only the
    machine-readable ``robustness_rows`` headline, reconciled to a unique model
    table and coefficient table, can establish that third fact.
    """

    raw_rows = step_summary.get("robustness_rows")
    if not isinstance(raw_rows, list):
        raw_rows = []
    rows_by_id: Dict[str, List[Dict[str, Any]]] = {}
    for raw_row in raw_rows:
        if not isinstance(raw_row, dict):
            continue
        spec_id = str(raw_row.get("spec_id") or "").strip()
        if spec_id:
            rows_by_id.setdefault(spec_id, []).append(dict(raw_row))

    model_frame, model_table_errors = _unique_structured_csv(
        out_dir=out_dir,
        required_columns=set(ROBUSTNESS_MODEL_RESULT_REQUIRED_COLUMNS),
        required_any_columns=set(ROBUSTNESS_MODEL_RESULT_IDENTIFIER_COLUMNS),
    )
    coefficient_frame, coefficient_table_errors = _unique_structured_csv(
        out_dir=out_dir,
        required_columns=set(ROBUSTNESS_COEFFICIENT_RESULT_REQUIRED_COLUMNS),
    )
    issues: List[Dict[str, Any]] = []
    if model_table_errors:
        issues.append(
            {"issue": "model_result_table_unavailable", "detail": model_table_errors}
        )
    if coefficient_table_errors:
        issues.append(
            {
                "issue": "coefficient_result_table_unavailable",
                "detail": coefficient_table_errors,
            }
        )

    primary_outcome = str(
        (context.target_outcome if context is not None else None) or ""
    ).strip()

    def _number_agrees(observed: Any, expected: Any) -> bool:
        left = _finite_result_number(observed)
        right = _finite_result_number(expected)
        if left is None or right is None:
            return left is None and right is None
        return math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-12)

    for spec_id, spec in locked_by_id.items():
        candidates = rows_by_id.get(spec_id, [])
        if len(candidates) != 1:
            issues.append(
                {
                    "spec_id": spec_id,
                    "issue": "executed_result_row_count",
                    "expected": 1,
                    "observed": len(candidates),
                }
            )
            continue
        row = candidates[0]
        axis = _normalise_result_token(spec.get("axis"))
        if _normalise_result_token(row.get("axis")) != axis:
            issues.append(
                {
                    "spec_id": spec_id,
                    "issue": "executed_axis_mismatch",
                    "expected": axis,
                    "observed": row.get("axis"),
                }
            )

        missing_fields = [
            field
            for field in ROBUSTNESS_RESULT_REQUIRED_TEXT_FIELDS
            if not str(row.get(field) or "").strip()
        ]
        for field in ROBUSTNESS_RESULT_REQUIRED_BOOLEAN_FIELDS:
            if _result_bool(row.get(field)) is None:
                missing_fields.append(field)
        for field in ("ci_low", "ci_high"):
            if field not in row:
                missing_fields.append(field)
        n_value = _nonnegative_integral_value(row.get("n"))
        point = _finite_result_number(row.get("point_estimate"))
        if n_value is None or n_value <= 0:
            missing_fields.append("positive_n")
        if point is None:
            missing_fields.append("finite_point_estimate")
        if missing_fields:
            issues.append(
                {
                    "spec_id": spec_id,
                    "issue": "executed_result_fields_missing_or_invalid",
                    "fields": sorted(set(missing_fields)),
                }
            )
        if _normalise_result_token(row.get("status")) not in {
            "analyzed",
            "executed",
            "estimated",
            "ok",
        }:
            issues.append(
                {
                    "spec_id": spec_id,
                    "issue": "executed_result_status_invalid",
                    "observed": row.get("status"),
                }
            )

        outcome_override = spec.get("outcome_override")
        expected_outcome = primary_outcome
        if axis == "outcome" and isinstance(outcome_override, dict):
            expected_outcome = str(
                outcome_override.get("concept_id")
                or outcome_override.get("column")
                or outcome_override.get("target")
                or ""
            ).strip()
            applied = row.get("applied_outcome_override")
            if not isinstance(applied, dict) or json.dumps(
                applied, sort_keys=True, separators=(",", ":"), ensure_ascii=False
            ) != json.dumps(
                outcome_override,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ):
                issues.append(
                    {
                        "spec_id": spec_id,
                        "issue": "outcome_override_not_applied_exactly",
                        "expected": outcome_override,
                        "observed": applied,
                    }
                )
        if expected_outcome and _normalise_result_token(
            row.get("outcome_concept_id")
        ) != _normalise_result_token(expected_outcome):
            issues.append(
                {
                    "spec_id": spec_id,
                    "issue": "executed_outcome_mismatch",
                    "expected": expected_outcome,
                    "observed": row.get("outcome_concept_id"),
                }
            )

        missing_override = spec.get("missing_override")
        if axis == "missing" and isinstance(missing_override, dict):
            expected_strategy = _normalise_result_token(
                missing_override.get("strategy")
            )
            if _normalise_result_token(row.get("missing_strategy")) != expected_strategy:
                issues.append(
                    {
                        "spec_id": spec_id,
                        "issue": "missing_strategy_mismatch",
                        "expected": expected_strategy,
                        "observed": row.get("missing_strategy"),
                    }
                )
            analysis_set = _normalise_result_token(row.get("analysis_set"))
            policy = _normalise_result_token(row.get("baseline_missing_policy"))
            if expected_strategy == "complete_case" and analysis_set != "complete_case":
                issues.append(
                    {
                        "spec_id": spec_id,
                        "issue": "complete_case_model_not_used",
                        "observed_analysis_set": analysis_set,
                    }
                )
            if expected_strategy == "missing_indicator" and (
                analysis_set != "source_aware" or policy != "explicit_missing_category"
            ):
                issues.append(
                    {
                        "spec_id": spec_id,
                        "issue": "missing_indicator_model_not_used",
                        "observed_analysis_set": analysis_set,
                        "observed_policy": policy,
                    }
                )

        converged = _result_bool(row.get("converged"))
        penalized = _result_bool(row.get("penalized"))
        reportable = _result_bool(row.get("reportable"))
        low = _finite_result_number(row.get("ci_low"))
        high = _finite_result_number(row.get("ci_high"))
        interval_method = _normalise_result_token(row.get("interval_method"))
        if converged is not True or _normalise_result_token(row.get("fit_status")) != "fitted":
            issues.append(
                {"spec_id": spec_id, "issue": "executed_model_not_fitted_converged"}
            )
        finite_interval = low is not None and high is not None and low <= high
        point_only = low is None and high is None
        if reportable is True and not finite_interval:
            issues.append(
                {"spec_id": spec_id, "issue": "reportable_result_requires_finite_ci"}
            )
        if point_only and not (
            penalized is True and interval_method == "unavailable" and reportable is False
        ):
            issues.append(
                {
                    "spec_id": spec_id,
                    "issue": "point_only_result_must_be_penalized_nonreportable",
                }
            )
        if (low is None) != (high is None):
            issues.append(
                {"spec_id": spec_id, "issue": "partial_confidence_interval"}
            )

        if model_frame is None or coefficient_frame is None:
            continue
        model_id = str(row.get("model_id") or "")
        model_rows = model_frame[
            model_frame["model_id"].astype(str).eq(model_id)
        ].copy()
        for identifier_column in ("definition_id", "spec_id"):
            if identifier_column not in model_rows.columns:
                continue
            model_rows = model_rows[
                model_rows[identifier_column].astype(str).eq(spec_id)
            ]
        if len(model_rows) != 1:
            issues.append(
                {
                    "spec_id": spec_id,
                    "issue": "model_contract_row_count",
                    "model_id": model_id,
                    "observed": int(len(model_rows)),
                }
            )
            continue
        model_row = model_rows.iloc[0]
        comparisons = {
            "outcome_concept_id": model_row.get("outcome"),
            "model_family": model_row.get("model_family"),
            "effect_scale": model_row.get("effect_scale"),
            "analysis_set": model_row.get("analysis_set"),
            "baseline_missing_policy": model_row.get("baseline_missing_policy"),
            "fit_status": model_row.get("fit_status"),
            "interval_method": model_row.get("interval_method"),
        }
        for field, expected in comparisons.items():
            if _normalise_result_token(row.get(field)) != _normalise_result_token(
                expected
            ):
                issues.append(
                    {
                        "spec_id": spec_id,
                        "issue": "model_contract_field_mismatch",
                        "field": field,
                        "expected": expected,
                        "observed": row.get(field),
                    }
                )
        for field in ("converged", "penalized"):
            if _result_bool(row.get(field)) != _result_bool(model_row.get(field)):
                issues.append(
                    {
                        "spec_id": spec_id,
                        "issue": "model_contract_field_mismatch",
                        "field": field,
                    }
                )
        if n_value is not None and n_value != _nonnegative_integral_value(
            model_row.get("n")
        ):
            issues.append(
                {"spec_id": spec_id, "issue": "model_n_mismatch"}
            )
        for field, model_fields in (
            ("point_estimate", ("point_estimate", "stage3_effect", "estimate")),
            ("ci_low", ("ci_low", "stage3_ci_low")),
            ("ci_high", ("ci_high", "stage3_ci_high")),
        ):
            expected = next(
                (
                    model_row.get(candidate)
                    for candidate in model_fields
                    if candidate in model_rows.columns
                ),
                None,
            )
            if not _number_agrees(row.get(field), expected):
                issues.append(
                    {
                        "spec_id": spec_id,
                        "issue": "model_result_value_mismatch",
                        "field": field,
                        "expected": expected,
                        "observed": row.get(field),
                    }
                )

        coefficient_term = str(row.get("coefficient_term") or "")
        coefficient_rows = coefficient_frame[
            coefficient_frame["model_id"].astype(str).eq(model_id)
            & coefficient_frame["term"].astype(str).eq(coefficient_term)
        ].copy()
        if len(coefficient_rows) != 1:
            issues.append(
                {
                    "spec_id": spec_id,
                    "issue": "coefficient_term_row_count",
                    "model_id": model_id,
                    "coefficient_term": coefficient_term,
                    "observed": int(len(coefficient_rows)),
                }
            )
        else:
            coefficient_row = coefficient_rows.iloc[0]
            if _normalise_result_token(coefficient_row.get("term_role")) != "exposure":
                issues.append(
                    {"spec_id": spec_id, "issue": "coefficient_term_not_exposure"}
                )
            for field, coefficient_field in (
                ("outcome_concept_id", "outcome"),
                ("effect_scale", "effect_scale"),
            ):
                if _normalise_result_token(row.get(field)) != _normalise_result_token(
                    coefficient_row.get(coefficient_field)
                ):
                    issues.append(
                        {
                            "spec_id": spec_id,
                            "issue": "coefficient_contract_field_mismatch",
                            "field": field,
                        }
                    )
            for field in ("point_estimate", "ci_low", "ci_high"):
                coefficient_field = "estimate" if field == "point_estimate" else field
                if not _number_agrees(row.get(field), coefficient_row.get(coefficient_field)):
                    issues.append(
                        {
                            "spec_id": spec_id,
                            "issue": "coefficient_result_value_mismatch",
                            "field": field,
                        }
                    )
        if axis == "missing" and _normalise_result_token(
            (spec.get("missing_override") or {}).get("strategy")
        ) == "missing_indicator":
            model_terms = coefficient_frame[
                coefficient_frame["model_id"].astype(str).eq(model_id)
            ]["term"].astype(str)
            if not model_terms.str.lower().str.contains("missing", regex=False).any():
                issues.append(
                    {"spec_id": spec_id, "issue": "missing_indicator_term_absent"}
                )

    return issues


__all__ = [
    "ROBUSTNESS_COHORT_MEMBERSHIP_ALIASES",
    "ROBUSTNESS_EXECUTION_CONTRACT_GUIDANCE",
    "ROBUSTNESS_RESULT_REQUIRED_FIELDS",
    "_executed_robustness_result_issues",
]

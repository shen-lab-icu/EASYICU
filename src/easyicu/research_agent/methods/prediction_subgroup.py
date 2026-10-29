"""Descriptive subgroup performance of one frozen binary probability column.

AUROC, Brier and joint logistic calibration reuse prediction_validation.
For a single-class cell only the defined Brier score/rates remain available.
No model, sensitive category, threshold, reference group or split is selected.
"""

from __future__ import annotations

from itertools import product
from typing import Any, Mapping

import numpy as np
import pandas as pd
from pydantic import ValidationError
from sklearn.metrics import brier_score_loss

from ..canonical_json import canonical_sha256
from ..contracts.closed_levels import typed_level_key
from ..contracts.prediction_subgroup import (
    PredictionSubgroupError, PredictionSubgroupFinding, PredictionSubgroupPerformance, PredictionSubgroupResult,
    PredictionSubgroupSpec, prediction_subgroup_spec_sha256,
)
from ..contracts.prediction_validation import PredictionValidationSpec
from .prediction_validation import evaluate_binary_predictions


def _refuse(code: str, message: str) -> None:
    raise PredictionSubgroupError("prediction_subgroup_" + code, message)


def _missing(series: pd.Series) -> pd.Series:
    return series.isna() | series.map(lambda value: isinstance(value, str) and not value.strip())


def subgroup_level_key(value: Any) -> tuple[str, str]:
    """Normalize scalar storage wrappers, retaining typed category identity."""
    if pd.isna(value):
        return ("missing", "")
    if isinstance(value, np.generic):
        value = value.item()
    return typed_level_key(value)


def _validated_evaluation(frame: pd.DataFrame, spec: PredictionSubgroupSpec) -> pd.DataFrame:
    required = {
        spec.unit_id_column, spec.subject_id_column, spec.split_column,
        spec.outcome_column, spec.probability_column, *(axis.column for axis in spec.axes),
    }
    if not frame.columns.is_unique or required - set(frame.columns):
        _refuse("columns_invalid", "the prediction table requires unique declared columns")
    if frame.empty:
        _refuse("empty_input", "the prediction table contains no rows")
    for column in (spec.unit_id_column, spec.subject_id_column):
        if bool(_missing(frame[column]).any()):
            _refuse("identity_missing", "every row requires both declared identities")
    if bool(frame[spec.unit_id_column].duplicated().any()):
        _refuse("duplicate_unit", "prediction unit identifiers must be unique")
    splits = frame[spec.split_column]
    if bool(_missing(splits).any()) or any(not isinstance(value, str) or value != value.strip() for value in splits):
        _refuse("split_invalid", "split labels must be canonical nonempty strings")
    if bool((splits.groupby(frame[spec.subject_id_column]).nunique() > 1).any()):
        _refuse("subject_split_leakage", "a subject cannot occur in multiple splits")
    evaluation = frame.loc[splits.eq(spec.evaluation_split)].copy()
    if evaluation.empty:
        _refuse("evaluation_split_missing", "the declared split contains no rows")
    if spec.analysis_unit == "subject" and bool(evaluation[spec.subject_id_column].duplicated().any()):
        _refuse("subject_unit_not_unique", "subject analysis requires one row per subject")
    for column, role in ((spec.outcome_column, "outcome"), (spec.probability_column, "probability")):
        raw = evaluation[column]
        if pd.api.types.is_bool_dtype(raw.dtype) or not pd.api.types.is_numeric_dtype(raw.dtype):
            _refuse(role + "_invalid", f"{role} requires a numeric non-boolean dtype")
        observed = raw.dropna()
        invalid = ~observed.isin((0, 1)) if role == "outcome" else (
            ~np.isfinite(observed) | observed.lt(0) | observed.gt(1)
        )
        if bool(invalid.any()):
            _refuse(role + "_invalid", f"observed {role} values are outside the declared binary probability contract")
        if spec.missing_value_policy == "error" and bool(raw.isna().any()):
            _refuse(role + "_missing", f"missing {role} requires an explicit complete-case declaration")
    for axis in spec.axes:
        allowed = {subgroup_level_key(level) for level in axis.levels}
        if any(subgroup_level_key(value) not in allowed for value in evaluation[axis.column]):
            _refuse("undeclared_level", "an evaluation subgroup value is outside the closed category roster")
    return evaluation


def _cell(
    frame: pd.DataFrame, spec: PredictionSubgroupSpec, *, scope: str, group_values: dict[str, Any],
) -> PredictionSubgroupPerformance:
    complete = frame[spec.outcome_column].notna() & frame[spec.probability_column].notna()
    evaluated = frame.loc[complete]
    n = len(evaluated)
    outcomes = evaluated[spec.outcome_column].to_numpy(dtype=int)
    probabilities = evaluated[spec.probability_column].to_numpy(dtype=float)
    events = int(outcomes.sum())
    subjects = evaluated[spec.subject_id_column].value_counts()
    small = n < spec.minimum_group_n or min(events, n - events) < spec.minimum_class_n
    reasons = []
    if not n:
        reasons.append("empty_after_declared_complete_case" if len(frame) else "empty_declared_group")
    elif small:
        reasons.append("below_declared_sample_counts")
    values: dict[str, Any] = {}
    if n:
        values.update(event_rate=float(outcomes.mean()), mean_predicted_probability=float(probabilities.mean()))
    if n and events and n - events:
        result = evaluate_binary_predictions(evaluated, PredictionValidationSpec(
            unit_id_column=spec.unit_id_column, subject_id_column=spec.subject_id_column,
            split_column=spec.split_column, outcome_column=spec.outcome_column,
            probability_column=spec.probability_column, evaluation_split=spec.evaluation_split,
            analysis_unit=spec.analysis_unit, thresholds=(0.5,),
            calibration_bins=spec.calibration_bins, calibration_logit_epsilon=spec.calibration_logit_epsilon,
        )).summary
        values.update({name: getattr(result, name) for name in (
            "auroc", "brier_score", "calibration_status", "calibration_intercept",
            "calibration_slope", "clipped_probability_n",
        )})
        values["auroc_status"] = "estimated"
        if result.calibration_status != "estimated":
            reasons.append(result.calibration_status)
    else:
        status = "not_estimable_single_class" if n else "not_estimable_empty"
        values.update(auroc_status=status, calibration_status=status)
        if n:
            # The base owner correctly refuses single-class AUROC/calibration.
            # sklearn still defines the binary Brier score for these cells.
            values["brier_score"] = float(brier_score_loss(outcomes, probabilities, pos_label=1))
            reasons.append(status)
    if bool((subjects > 1).any()):
        reasons.append("repeated_subjects_point_estimates_only")
    return PredictionSubgroupPerformance(
        scope=scope, group_values=group_values, input_n=len(frame), evaluation_n=n,
        excluded_n=len(frame) - n, missing_outcome_n=int(frame[spec.outcome_column].isna().sum()),
        missing_probability_n=int(frame[spec.probability_column].isna().sum()),
        event_n=events, non_event_n=n - events, subject_n=len(subjects),
        repeated_subject_n=int((subjects > 1).sum()),
        sample_status="empty" if not n else "small_sample" if small else "adequate_for_declared_counts",
        reason_codes=tuple(reasons), **values,
    )


def evaluate_prediction_subgroups(
    frame: pd.DataFrame, spec: PredictionSubgroupSpec | Mapping[str, Any],
) -> PredictionSubgroupResult:
    """Evaluate all prespecified cells, retaining empty and non-estimable cells."""
    parsed = PredictionSubgroupSpec.model_validate(spec)
    evaluation = _validated_evaluation(frame, parsed)
    rows = [_cell(evaluation, parsed, scope="overall", group_values={})]
    keys = {axis.column: evaluation[axis.column].map(subgroup_level_key) for axis in parsed.axes}
    for axis in parsed.axes:
        for level in axis.levels:
            mask = keys[axis.column].map(lambda key: key == subgroup_level_key(level))
            rows.append(_cell(evaluation.loc[mask], parsed, scope="marginal", group_values={axis.column: level}))
    if parsed.include_intersections and len(parsed.axes) > 1:
        for levels in product(*(axis.levels for axis in parsed.axes)):
            values = dict(zip((axis.column for axis in parsed.axes), levels))
            mask = pd.Series(True, index=evaluation.index)
            for column, level in values.items():
                mask &= keys[column].map(lambda key: key == subgroup_level_key(level))
            rows.append(_cell(evaluation.loc[mask], parsed, scope="intersection", group_values=values))
    return PredictionSubgroupResult(
        contract_sha256=prediction_subgroup_spec_sha256(parsed), evaluation_split=parsed.evaluation_split,
        analysis_unit=parsed.analysis_unit, input_n=len(frame), evaluation_input_n=len(evaluation),
        evaluation_unit_ids_sha256=canonical_sha256([
            list(subgroup_level_key(value)) for value in evaluation[parsed.unit_id_column]
        ]), rows=tuple(rows),
    )


def prediction_subgroup_result_findings(
    *, frame: pd.DataFrame, spec: PredictionSubgroupSpec | Mapping[str, Any],
    candidate: PredictionSubgroupResult | Mapping[str, Any],
) -> tuple[PredictionSubgroupFinding, ...]:
    """Recompute a candidate, without issuing an authority seal or paper grant."""
    payload = candidate.model_dump(mode="python") if isinstance(candidate, PredictionSubgroupResult) else candidate
    try:
        parsed = PredictionSubgroupResult.model_validate(payload)
    except ValidationError:
        return (PredictionSubgroupFinding(
            reason_code="prediction_subgroup_result_schema_invalid",
            message="candidate subgroup performance is not schema-valid",
        ),)
    expected = evaluate_prediction_subgroups(frame, spec)
    if canonical_sha256(parsed.model_dump(mode="json")) != canonical_sha256(expected.model_dump(mode="json")):
        return (PredictionSubgroupFinding(
            reason_code="prediction_subgroup_result_mismatch",
            message="candidate subgroup performance differs from declared-input recomputation",
        ),)
    return ()


def bind_subgroup_membership(
    predictions: pd.DataFrame, membership: pd.DataFrame, spec: PredictionSubgroupSpec,
) -> pd.DataFrame:
    """Connect identities explicitly; never infer a join from row positions.

    The existing prediction producer encodes a unique source identity as
    ``type:repr``. That encoding is allowed only when explicitly declared and
    the source key is unique; its positional duplicate-key fallback cannot be
    reconstructed safely here and is refused.
    """
    needed = {spec.subgroup_unit_id_column, *(axis.column for axis in spec.axes)}
    if (not membership.columns.is_unique or needed - set(membership.columns)
            or spec.unit_id_column not in predictions or not predictions.columns.is_unique):
        raise PredictionSubgroupError("prediction_subgroup_membership_columns_invalid", "membership and prediction tables require their declared unique columns")
    source = membership[spec.subgroup_unit_id_column]
    if bool(source.isna().any()) or bool(source.map(lambda value: isinstance(value, str) and not value.strip()).any()):
        raise PredictionSubgroupError("prediction_subgroup_membership_identity_missing", "membership requires a nonmissing row key")
    if spec.subgroup_key_encoding == "prediction_typed_scalar":
        source_keys = source.map(lambda value: f"{type(value).__name__}:{value!r}")
        score_keys = predictions[spec.unit_id_column]
    else:
        source_keys = source.map(subgroup_level_key)
        score_keys = predictions[spec.unit_id_column].map(subgroup_level_key)
    if bool(source_keys.duplicated().any()):
        raise PredictionSubgroupError("prediction_subgroup_membership_duplicate_unit", "membership requires a unique key without positional disambiguation")
    positions = {key: position for position, key in enumerate(source_keys)}
    if any(key not in positions for key in score_keys):
        raise PredictionSubgroupError("prediction_subgroup_membership_unmatched", "every prediction row must match exactly one declared membership row")
    selected = membership.iloc[[positions[key] for key in score_keys]]
    result = predictions.copy()
    for axis in spec.axes:
        if axis.column in result:
            raise PredictionSubgroupError("prediction_subgroup_axis_collision", "the score artifact already contains a declared membership column")
        result[axis.column] = selected[axis.column].to_numpy()
    return result


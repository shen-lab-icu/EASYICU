"""Deterministic validation of already-produced binary risk probabilities.

This experimental kernel owns metric mechanics only.  It does not train or
select a model, choose a cohort, outcome, split, threshold, or analysis unit.
Those coordinates must arrive in a digest-bound ``PredictionValidationSpec``.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import brier_score_loss, roc_auc_score
from statsmodels.tools.sm_exceptions import (
    PerfectSeparationError,
    PerfectSeparationWarning,
)

from ..contracts.prediction_validation import (
    CalibrationStatus,
    PredictionCalibrationBin,
    PredictionThresholdMetric,
    PredictionValidationError,
    PredictionValidationReason,
    PredictionValidationResult,
    PredictionValidationSpec,
    PredictionValidationSummary,
    prediction_validation_spec_sha256,
)


def _raise(
    reason_code: PredictionValidationReason,
    message: str,
    **detail: Any,
) -> None:
    raise PredictionValidationError(reason_code, message, **detail)


def _missing_identity_mask(series: pd.Series) -> pd.Series:
    return series.isna() | series.map(
        lambda value: isinstance(value, str) and not value.strip()
    )


def _evaluation_arrays(
    frame: pd.DataFrame,
    spec: PredictionValidationSpec,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    required = {
        spec.unit_id_column,
        spec.subject_id_column,
        spec.split_column,
        spec.outcome_column,
        spec.probability_column,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        _raise(
            PredictionValidationReason.MISSING_COLUMNS,
            "prediction table is missing declared columns",
            missing_columns=missing,
        )
    if frame.empty:
        _raise(
            PredictionValidationReason.EMPTY_INPUT,
            "prediction table contains no rows",
        )

    for column in (spec.unit_id_column, spec.subject_id_column):
        missing_identity = _missing_identity_mask(frame[column])
        if bool(missing_identity.any()):
            _raise(
                PredictionValidationReason.IDENTITY_MISSING,
                "prediction table contains a missing identity",
                column=column,
                row_count=int(missing_identity.sum()),
            )
    duplicate_units = frame[spec.unit_id_column].duplicated(keep=False)
    if bool(duplicate_units.any()):
        _raise(
            PredictionValidationReason.DUPLICATE_UNIT,
            "unit identifiers must be globally unique",
            column=spec.unit_id_column,
            row_count=int(duplicate_units.sum()),
        )

    raw_splits = frame[spec.split_column]
    invalid_splits = raw_splits.isna() | raw_splits.map(
        lambda value: not str(value).strip() if value is not None else True
    )
    if bool(invalid_splits.any()):
        _raise(
            PredictionValidationReason.SPLIT_MISSING,
            "every row requires a non-empty split label",
            column=spec.split_column,
            row_count=int(invalid_splits.sum()),
        )
    splits = raw_splits.map(lambda value: str(value).strip())
    subject_split_count = splits.groupby(
        frame[spec.subject_id_column], sort=False
    ).nunique()
    leaking_subjects = subject_split_count[subject_split_count > 1]
    if not leaking_subjects.empty:
        _raise(
            PredictionValidationReason.SUBJECT_SPLIT_LEAKAGE,
            "a subject occurs in more than one split",
            subject_count=int(len(leaking_subjects)),
        )

    evaluation_mask = splits.eq(spec.evaluation_split)
    if not bool(evaluation_mask.any()):
        _raise(
            PredictionValidationReason.EVALUATION_SPLIT_MISSING,
            "the declared evaluation split contains no rows",
            evaluation_split=spec.evaluation_split,
            available_splits=sorted(set(splits)),
        )
    evaluation = frame.loc[evaluation_mask].copy()
    repeated_subjects = evaluation[spec.subject_id_column].duplicated(keep=False)
    if spec.analysis_unit == "subject" and bool(repeated_subjects.any()):
        _raise(
            PredictionValidationReason.SUBJECT_UNIT_NOT_UNIQUE,
            "subject-level evaluation requires exactly one row per subject",
            row_count=int(repeated_subjects.sum()),
        )

    raw_outcomes = evaluation[spec.outcome_column]
    if pd.api.types.is_bool_dtype(
        raw_outcomes.dtype
    ) or not pd.api.types.is_numeric_dtype(raw_outcomes.dtype):
        _raise(
            PredictionValidationReason.OUTCOME_INVALID,
            "evaluation outcomes must use a numeric, non-boolean dtype",
            column=spec.outcome_column,
            observed_dtype=str(raw_outcomes.dtype),
        )
    outcomes = pd.to_numeric(raw_outcomes, errors="coerce")
    invalid_outcomes = outcomes.isna() | ~outcomes.isin((0, 1))
    if bool(invalid_outcomes.any()):
        _raise(
            PredictionValidationReason.OUTCOME_INVALID,
            "evaluation outcomes must be observed binary values",
            column=spec.outcome_column,
            row_count=int(invalid_outcomes.sum()),
        )
    raw_probabilities = evaluation[spec.probability_column]
    if pd.api.types.is_bool_dtype(
        raw_probabilities.dtype
    ) or not pd.api.types.is_numeric_dtype(raw_probabilities.dtype):
        _raise(
            PredictionValidationReason.PROBABILITY_INVALID,
            "evaluation probabilities must use a numeric, non-boolean dtype",
            column=spec.probability_column,
            observed_dtype=str(raw_probabilities.dtype),
        )
    probabilities = pd.to_numeric(raw_probabilities, errors="coerce")
    invalid_probabilities = (
        probabilities.isna()
        | ~np.isfinite(probabilities)
        | probabilities.lt(0.0)
        | probabilities.gt(1.0)
    )
    if bool(invalid_probabilities.any()):
        _raise(
            PredictionValidationReason.PROBABILITY_INVALID,
            "evaluation probabilities must be finite values in [0, 1]",
            column=spec.probability_column,
            row_count=int(invalid_probabilities.sum()),
        )

    outcome_values = outcomes.astype(int).to_numpy()
    probability_values = probabilities.astype(float).to_numpy()
    if len(np.unique(outcome_values)) != 2:
        _raise(
            PredictionValidationReason.SINGLE_CLASS,
            "evaluation requires both outcome classes",
            observed_classes=sorted(set(outcome_values.tolist())),
        )
    return evaluation, outcome_values, probability_values


def _perfectly_separated(outcomes: np.ndarray, logits: np.ndarray) -> bool:
    event_logits = logits[outcomes == 1]
    non_event_logits = logits[outcomes == 0]
    return bool(
        np.min(event_logits) > np.max(non_event_logits)
        or np.max(event_logits) < np.min(non_event_logits)
    )


def _calibration_model(
    outcomes: np.ndarray,
    probabilities: np.ndarray,
    *,
    epsilon: float,
) -> tuple[CalibrationStatus, float | None, float | None, int]:
    clipped = np.clip(probabilities, epsilon, 1.0 - epsilon)
    clipped_n = int(np.count_nonzero(clipped != probabilities))
    logits = np.log(clipped / (1.0 - clipped))
    if np.unique(logits).size < 2:
        return "not_estimable_constant_probability", None, None, clipped_n
    if _perfectly_separated(outcomes, logits):
        return "not_estimable_perfect_separation", None, None, clipped_n

    design = sm.add_constant(logits, has_constant="add")
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", PerfectSeparationWarning)
            fitted = sm.GLM(
                outcomes,
                design,
                family=sm.families.Binomial(),
            ).fit(maxiter=100, tol=1e-10)
    except (PerfectSeparationError, np.linalg.LinAlgError):
        return "not_estimable_nonconvergence", None, None, clipped_n
    if any(issubclass(item.category, PerfectSeparationWarning) for item in caught):
        return "not_estimable_perfect_separation", None, None, clipped_n
    parameters = np.asarray(fitted.params, dtype=float)
    if (
        not bool(fitted.converged)
        or parameters.shape != (2,)
        or not np.isfinite(parameters).all()
    ):
        return "not_estimable_nonconvergence", None, None, clipped_n
    return "estimated", float(parameters[0]), float(parameters[1]), clipped_n


def _calibration_bins(
    outcomes: np.ndarray,
    probabilities: np.ndarray,
    *,
    requested_bins: int,
) -> tuple[PredictionCalibrationBin, ...]:
    quantile_count = min(int(requested_bins), len(probabilities))
    edges = np.unique(
        np.quantile(
            probabilities,
            np.linspace(0.0, 1.0, quantile_count + 1),
            method="linear",
        )
    )
    labels = (
        np.zeros(len(probabilities), dtype=int)
        if len(edges) == 1
        else np.searchsorted(edges[1:-1], probabilities, side="right")
    )
    rows: list[PredictionCalibrationBin] = []
    for output_index, label in enumerate(sorted(set(labels.tolist())), start=1):
        selected = labels == label
        selected_outcomes = outcomes[selected]
        selected_probabilities = probabilities[selected]
        rows.append(
            PredictionCalibrationBin(
                bin_index=output_index,
                n=int(selected.sum()),
                event_n=int(selected_outcomes.sum()),
                mean_predicted_probability=float(selected_probabilities.mean()),
                observed_event_rate=float(selected_outcomes.mean()),
                minimum_predicted_probability=float(selected_probabilities.min()),
                maximum_predicted_probability=float(selected_probabilities.max()),
            )
        )
    return tuple(rows)


def _safe_ratio(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else float(numerator / denominator)


def _threshold_metrics(
    outcomes: np.ndarray,
    probabilities: np.ndarray,
    *,
    thresholds: tuple[float, ...],
) -> tuple[PredictionThresholdMetric, ...]:
    rows: list[PredictionThresholdMetric] = []
    for threshold in thresholds:
        positive = probabilities >= threshold
        negative = ~positive
        events = outcomes == 1
        non_events = ~events
        true_positive = int(np.count_nonzero(positive & events))
        false_positive = int(np.count_nonzero(positive & non_events))
        true_negative = int(np.count_nonzero(negative & non_events))
        false_negative = int(np.count_nonzero(negative & events))
        rows.append(
            PredictionThresholdMetric(
                threshold=threshold,
                n=int(len(outcomes)),
                predicted_positive_n=int(np.count_nonzero(positive)),
                true_positive_n=true_positive,
                false_positive_n=false_positive,
                true_negative_n=true_negative,
                false_negative_n=false_negative,
                sensitivity=float(true_positive / (true_positive + false_negative)),
                specificity=float(true_negative / (true_negative + false_positive)),
                positive_predictive_value=_safe_ratio(
                    true_positive, true_positive + false_positive
                ),
                negative_predictive_value=_safe_ratio(
                    true_negative, true_negative + false_negative
                ),
            )
        )
    return tuple(rows)


def evaluate_binary_predictions(
    frame: pd.DataFrame,
    spec: PredictionValidationSpec | Mapping[str, Any],
) -> PredictionValidationResult:
    """Evaluate one predeclared split without training or choosing anything."""

    parsed_spec = PredictionValidationSpec.model_validate(spec)
    evaluation, outcomes, probabilities = _evaluation_arrays(frame, parsed_spec)
    calibration_status, intercept, slope, clipped_n = _calibration_model(
        outcomes,
        probabilities,
        epsilon=parsed_spec.calibration_logit_epsilon,
    )
    subject_counts = evaluation[parsed_spec.subject_id_column].value_counts()
    repeated_subject_n = int((subject_counts > 1).sum())
    event_n = int(outcomes.sum())
    evaluation_n = int(len(outcomes))
    summary = PredictionValidationSummary(
        evaluation_split=parsed_spec.evaluation_split,
        analysis_unit=parsed_spec.analysis_unit,
        input_n=int(len(frame)),
        evaluation_n=evaluation_n,
        event_n=event_n,
        non_event_n=evaluation_n - event_n,
        evaluation_subject_n=int(subject_counts.size),
        repeated_subject_n=repeated_subject_n,
        clipped_probability_n=clipped_n,
        event_rate=float(outcomes.mean()),
        mean_predicted_probability=float(probabilities.mean()),
        auroc=float(roc_auc_score(outcomes, probabilities)),
        brier_score=float(brier_score_loss(outcomes, probabilities)),
        calibration_status=calibration_status,
        calibration_intercept=intercept,
        calibration_slope=slope,
    )
    return PredictionValidationResult(
        contract_sha256=prediction_validation_spec_sha256(parsed_spec),
        summary=summary,
        calibration_bins=_calibration_bins(
            outcomes,
            probabilities,
            requested_bins=parsed_spec.calibration_bins,
        ),
        threshold_metrics=_threshold_metrics(
            outcomes,
            probabilities,
            thresholds=parsed_spec.thresholds,
        ),
    )


__all__ = ["evaluate_binary_predictions"]

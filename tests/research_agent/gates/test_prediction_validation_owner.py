"""Synthetic oracle and fail-closed checks for prediction-validation owner."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from pydantic import ValidationError

from easyicu.research_agent.contracts.prediction_validation import (
    PredictionValidationError,
    PredictionValidationReason,
    PredictionValidationSpec,
    prediction_validation_result_sha256,
    prediction_validation_spec_sha256,
)
from easyicu.research_agent.execution.runners.prediction_validation_executor import (
    prediction_validation_result_findings,
    run_prediction_validation,
)


def _spec(**updates: object) -> PredictionValidationSpec:
    payload: dict[str, object] = {
        "unit_id_column": "row_id",
        "subject_id_column": "subject_id",
        "split_column": "split",
        "outcome_column": "outcome",
        "probability_column": "probability",
        "evaluation_split": "test",
        "analysis_unit": "subject",
        "thresholds": (0.5, 0.8),
        "calibration_bins": 3,
    }
    payload.update(updates)
    return PredictionValidationSpec.model_validate(payload)


def _oracle_frame() -> pd.DataFrame:
    probabilities = [0.2] * 5 + [0.5] * 4 + [0.8] * 5
    outcomes = [1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0]
    return pd.DataFrame(
        {
            "row_id": range(16),
            "subject_id": [f"subject-{index}" for index in range(16)],
            "split": ["train"] * 2 + ["test"] * 14,
            # Non-evaluation rows need no label or prediction.  They remain in
            # the frame so patient-level split integrity can be checked.
            "outcome": [None, None, *outcomes],
            "probability": [None, None, *probabilities],
        }
    )


def _assert_reason(
    frame: pd.DataFrame,
    reason: PredictionValidationReason,
    *,
    spec: PredictionValidationSpec | None = None,
) -> None:
    with pytest.raises(PredictionValidationError) as caught:
        run_prediction_validation(frame, spec or _spec())
    assert caught.value.reason_code is reason


def test_synthetic_oracle_recovers_declared_metrics_and_calibration() -> None:
    result = run_prediction_validation(_oracle_frame(), _spec())

    assert result.summary.input_n == 16
    assert result.summary.evaluation_n == 14
    assert result.summary.event_n == result.summary.non_event_n == 7
    assert result.summary.event_rate == pytest.approx(0.5)
    assert result.summary.mean_predicted_probability == pytest.approx(0.5)
    assert result.summary.auroc == pytest.approx(38 / 49)
    assert result.summary.brier_score == pytest.approx(2.6 / 14)
    assert result.summary.calibration_status == "estimated"
    assert result.summary.calibration_intercept == pytest.approx(0.0, abs=1e-10)
    assert result.summary.calibration_slope == pytest.approx(1.0, abs=1e-10)


def test_calibration_bins_and_threshold_counts_keep_denominators() -> None:
    result = run_prediction_validation(_oracle_frame(), _spec())

    assert [row.n for row in result.calibration_bins] == [5, 4, 5]
    assert [row.event_n for row in result.calibration_bins] == [1, 2, 4]
    assert [row.observed_event_rate for row in result.calibration_bins] == [
        0.2,
        0.5,
        0.8,
    ]
    assert sum(row.n for row in result.calibration_bins) == 14
    at_half, at_point_eight = result.threshold_metrics
    assert (
        at_half.true_positive_n,
        at_half.false_positive_n,
        at_half.true_negative_n,
        at_half.false_negative_n,
    ) == (6, 3, 4, 1)
    assert at_half.positive_predictive_value == pytest.approx(2 / 3)
    assert at_half.negative_predictive_value == pytest.approx(4 / 5)
    assert at_point_eight.sensitivity == pytest.approx(4 / 7)
    assert at_point_eight.specificity == pytest.approx(6 / 7)


def test_threshold_metrics_disclose_zero_denominators() -> None:
    result = run_prediction_validation(
        _oracle_frame(),
        _spec(thresholds=(0.1, 0.9)),
    )

    all_positive, all_negative = result.threshold_metrics
    assert all_positive.negative_predictive_value is None
    assert all_negative.positive_predictive_value is None


def test_result_validator_accepts_recomputation_and_rejects_tampering() -> None:
    frame = _oracle_frame()
    spec = _spec()
    result = run_prediction_validation(frame, spec)

    assert prediction_validation_result_findings(frame, spec, result) == ()
    tampered = result.model_dump(mode="json")
    tampered["summary"]["brier_score"] += 0.01
    findings = prediction_validation_result_findings(frame, spec, tampered)

    assert len(findings) == 1
    assert findings[0].reason_code is PredictionValidationReason.RESULT_MISMATCH
    assert findings[0].detail["path"] == "$.summary.brier_score"
    assert (
        findings[0].detail["expected_result_sha256"]
        != findings[0].detail["observed_result_sha256"]
    )


def test_result_validator_rejects_invalid_result_schema() -> None:
    candidate = run_prediction_validation(_oracle_frame(), _spec()).model_dump(
        mode="json"
    )
    del candidate["summary"]["event_n"]

    findings = prediction_validation_result_findings(
        _oracle_frame(), _spec(), candidate
    )

    assert len(findings) == 1
    assert findings[0].reason_code is PredictionValidationReason.RESULT_SCHEMA_INVALID


@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf"), float("-inf")])
def test_result_validator_rejects_nonfinite_optional_metrics(
    nonfinite: float,
) -> None:
    candidate = run_prediction_validation(_oracle_frame(), _spec()).model_dump(
        mode="python"
    )
    candidate["threshold_metrics"][0]["positive_predictive_value"] = nonfinite

    findings = prediction_validation_result_findings(
        _oracle_frame(), _spec(), candidate
    )

    assert len(findings) == 1
    assert findings[0].reason_code is PredictionValidationReason.RESULT_SCHEMA_INVALID


def test_subject_split_leakage_fails_before_metric_computation() -> None:
    frame = _oracle_frame()
    frame.loc[0, "subject_id"] = frame.loc[2, "subject_id"]

    _assert_reason(frame, PredictionValidationReason.SUBJECT_SPLIT_LEAKAGE)


def test_subject_and_encounter_analysis_units_have_distinct_contracts() -> None:
    frame = _oracle_frame()
    frame.loc[3, "subject_id"] = frame.loc[2, "subject_id"]

    _assert_reason(frame, PredictionValidationReason.SUBJECT_UNIT_NOT_UNIQUE)
    encounter_result = run_prediction_validation(
        frame,
        _spec(analysis_unit="encounter"),
    )

    assert encounter_result.summary.evaluation_n == 14
    assert encounter_result.summary.evaluation_subject_n == 13
    assert encounter_result.summary.repeated_subject_n == 1


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        ("missing_column", PredictionValidationReason.MISSING_COLUMNS),
        ("empty", PredictionValidationReason.EMPTY_INPUT),
        ("missing_identity", PredictionValidationReason.IDENTITY_MISSING),
        ("duplicate_unit", PredictionValidationReason.DUPLICATE_UNIT),
        ("missing_split", PredictionValidationReason.SPLIT_MISSING),
        (
            "missing_evaluation_split",
            PredictionValidationReason.EVALUATION_SPLIT_MISSING,
        ),
        ("invalid_outcome", PredictionValidationReason.OUTCOME_INVALID),
        ("invalid_probability", PredictionValidationReason.PROBABILITY_INVALID),
        ("single_class", PredictionValidationReason.SINGLE_CLASS),
    ],
)
def test_invalid_inputs_fail_with_owner_diagnostic(
    mutation: str,
    reason: PredictionValidationReason,
) -> None:
    frame = _oracle_frame()
    if mutation == "missing_column":
        frame = frame.drop(columns="probability")
    elif mutation == "empty":
        frame = frame.iloc[0:0]
    elif mutation == "missing_identity":
        frame.loc[2, "subject_id"] = None
    elif mutation == "duplicate_unit":
        frame.loc[2, "row_id"] = frame.loc[3, "row_id"]
    elif mutation == "missing_split":
        frame.loc[2, "split"] = " "
    elif mutation == "missing_evaluation_split":
        frame["split"] = "train"
    elif mutation == "invalid_outcome":
        frame.loc[2, "outcome"] = 2
    elif mutation == "invalid_probability":
        frame.loc[2, "probability"] = 1.1
    elif mutation == "single_class":
        frame.loc[frame["split"] == "test", "outcome"] = 0
    else:  # pragma: no cover - the parameter table is closed above
        raise AssertionError(mutation)

    _assert_reason(frame, reason)


@pytest.mark.parametrize(
    ("outcomes", "probabilities", "reason"),
    [
        (
            [False, True, False, True],
            [0.1, 0.4, 0.6, 0.9],
            PredictionValidationReason.OUTCOME_INVALID,
        ),
        (
            ["0", "1", "0", "1"],
            [0.1, 0.4, 0.6, 0.9],
            PredictionValidationReason.OUTCOME_INVALID,
        ),
        (
            [0, 1, 0, 1],
            [False, True, False, True],
            PredictionValidationReason.PROBABILITY_INVALID,
        ),
        (
            [0, 1, 0, 1],
            ["0.1", "0.4", "0.6", "0.9"],
            PredictionValidationReason.PROBABILITY_INVALID,
        ),
    ],
)
def test_numeric_owner_rejects_implicit_type_coercion(
    outcomes: list[object],
    probabilities: list[object],
    reason: PredictionValidationReason,
) -> None:
    frame = pd.DataFrame(
        {
            "row_id": range(4),
            "subject_id": [f"subject-{index}" for index in range(4)],
            "split": ["test"] * 4,
            "outcome": outcomes,
            "probability": probabilities,
        }
    )

    _assert_reason(frame, reason)


@pytest.mark.parametrize(
    ("probabilities", "expected_status", "expected_clipped_n"),
    [
        (
            [0.5, 0.5, 0.5, 0.5],
            "not_estimable_constant_probability",
            0,
        ),
        (
            [0.0, 0.1, 0.9, 1.0],
            "not_estimable_perfect_separation",
            2,
        ),
    ],
)
def test_non_estimable_calibration_is_explicit_and_never_fabricated(
    probabilities: list[float],
    expected_status: str,
    expected_clipped_n: int,
) -> None:
    frame = pd.DataFrame(
        {
            "row_id": range(4),
            "subject_id": [f"subject-{index}" for index in range(4)],
            "split": ["test"] * 4,
            "outcome": [0, 0, 1, 1],
            "probability": probabilities,
        }
    )

    result = run_prediction_validation(frame, _spec())

    assert result.summary.calibration_status == expected_status
    assert result.summary.calibration_intercept is None
    assert result.summary.calibration_slope is None
    assert result.summary.clipped_probability_n == expected_clipped_n


def test_spec_digest_binds_thresholds_and_spec_is_immutable() -> None:
    first = _spec(thresholds=(0.5,))
    second = _spec(thresholds=(0.6,))

    assert prediction_validation_spec_sha256(
        first
    ) != prediction_validation_spec_sha256(second)
    with pytest.raises(ValidationError):
        first.evaluation_split = "validation"  # type: ignore[misc]
    invalid_spec = first.model_copy(update={"thresholds": (2.0,)})
    with pytest.raises(ValidationError):
        prediction_validation_spec_sha256(invalid_spec)


def test_result_digest_revalidates_model_copies() -> None:
    result = run_prediction_validation(_oracle_frame(), _spec())
    invalid_result = result.model_copy(
        update={"summary": result.summary.model_copy(update={"auroc": 2.0})}
    )

    with pytest.raises(ValidationError):
        prediction_validation_result_sha256(invalid_result)


def test_experimental_runner_is_not_wired_into_production_selection() -> None:
    root = Path(__file__).resolve().parents[2]
    selection = (
        root / "src/easyicu/research_agent/execution/runners/selection.py"
    ).read_text(encoding="utf-8")
    runner = (
        root
        / "src/easyicu/research_agent/execution/runners/prediction_validation_executor.py"
    ).read_text(encoding="utf-8")

    assert "prediction_validation" not in selection
    assert "EvidenceStore" not in runner

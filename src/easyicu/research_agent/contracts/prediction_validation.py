"""Typed, non-authoritative contract for binary prediction validation.

The contract describes an already-produced probability table.  It does not
choose a model, threshold, cohort, outcome, or split, and it grants no paper
authority.  The experimental host runner resolves this declaration against a
data frame and returns a result bound to the declaration digest.
"""

from __future__ import annotations

import math
from enum import Enum
from typing import Any, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..canonical_json import canonical_sha256


class PredictionValidationReason(str, Enum):
    """Stable owner diagnostics for prediction-validation failures."""

    MISSING_COLUMNS = "prediction_validation_missing_columns"
    EMPTY_INPUT = "prediction_validation_empty_input"
    IDENTITY_MISSING = "prediction_validation_identity_missing"
    DUPLICATE_UNIT = "prediction_validation_duplicate_unit"
    SPLIT_MISSING = "prediction_validation_split_missing"
    SUBJECT_SPLIT_LEAKAGE = "prediction_validation_subject_split_leakage"
    EVALUATION_SPLIT_MISSING = "prediction_validation_evaluation_split_missing"
    SUBJECT_UNIT_NOT_UNIQUE = "prediction_validation_subject_unit_not_unique"
    OUTCOME_INVALID = "prediction_validation_outcome_invalid"
    PROBABILITY_INVALID = "prediction_validation_probability_invalid"
    SINGLE_CLASS = "prediction_validation_single_class"
    RESULT_SCHEMA_INVALID = "prediction_validation_result_schema_invalid"
    RESULT_MISMATCH = "prediction_validation_result_mismatch"
    SOURCE_ARTIFACT_INVALID = "prediction_validation_source_artifact_invalid"
    SOURCE_DIGEST_INVALID = "prediction_validation_source_digest_invalid"
    SOURCE_DIGEST_MISMATCH = "prediction_validation_source_digest_mismatch"
    SOURCE_READ_FAILED = "prediction_validation_source_read_failed"
    RECEIPT_SCHEMA_INVALID = "prediction_validation_receipt_schema_invalid"
    RECEIPT_MISMATCH = "prediction_validation_receipt_mismatch"


CalibrationStatus = Literal[
    "estimated",
    "not_estimable_constant_probability",
    "not_estimable_perfect_separation",
    "not_estimable_nonconvergence",
]


class PredictionValidationError(ValueError):
    """Typed refusal raised by the prediction-validation owner."""

    def __init__(
        self,
        reason_code: PredictionValidationReason,
        message: str,
        **detail: Any,
    ) -> None:
        self.reason_code = reason_code
        self.detail = dict(detail)
        super().__init__(f"{reason_code.value}: {message}")


class PredictionValidationSpec(BaseModel):
    """Predeclared evaluation coordinates for one binary risk model."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["easyicu.prediction_validation/1"] = (
        "easyicu.prediction_validation/1"
    )
    unit_id_column: str
    subject_id_column: str
    split_column: str
    outcome_column: str
    probability_column: str
    evaluation_split: str
    analysis_unit: Literal["subject", "encounter"]
    thresholds: tuple[float, ...] = Field(min_length=1, max_length=20)
    calibration_bins: int = Field(default=10, ge=2, le=20)
    calibration_logit_epsilon: float = Field(default=1e-6, gt=0.0, lt=0.5)
    calibration_method: Literal["logistic_recalibration_intercept_and_slope"] = (
        "logistic_recalibration_intercept_and_slope"
    )
    calibration_curve_strategy: Literal["quantile"] = "quantile"
    threshold_rule: Literal["probability_greater_than_or_equal"] = (
        "probability_greater_than_or_equal"
    )

    @field_validator(
        "unit_id_column",
        "subject_id_column",
        "split_column",
        "outcome_column",
        "probability_column",
        "evaluation_split",
    )
    @classmethod
    def _nonempty_coordinate(cls, value: str) -> str:
        cleaned = str(value or "").strip()
        if not cleaned:
            raise ValueError("prediction-validation coordinates must be non-empty")
        return cleaned

    @field_validator("thresholds")
    @classmethod
    def _closed_thresholds(cls, values: tuple[float, ...]) -> tuple[float, ...]:
        parsed = tuple(float(value) for value in values)
        if any(not math.isfinite(value) or not 0.0 < value < 1.0 for value in parsed):
            raise ValueError("thresholds must be finite and strictly inside (0, 1)")
        if len(parsed) != len(set(parsed)):
            raise ValueError("thresholds must be unique")
        return tuple(sorted(parsed))

    @model_validator(mode="after")
    def _distinct_data_coordinates(self) -> "PredictionValidationSpec":
        identity_columns = {self.unit_id_column, self.subject_id_column}
        value_columns = {
            self.split_column,
            self.outcome_column,
            self.probability_column,
        }
        if len(value_columns) != 3 or identity_columns & value_columns:
            raise ValueError(
                "split, outcome, and probability columns must be distinct from "
                "each other and from identity columns"
            )
        return self


def prediction_validation_spec_sha256(
    spec: PredictionValidationSpec | Mapping[str, Any],
) -> str:
    """Return the canonical digest of the exact evaluation declaration."""

    parsed = PredictionValidationSpec.model_validate(spec)
    return canonical_sha256(parsed.model_dump(mode="json"))


class PredictionValidationSummary(BaseModel):
    """Scalar results and dependence disclosure for one evaluation split."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    evaluation_split: str
    analysis_unit: Literal["subject", "encounter"]
    input_n: int = Field(ge=1)
    evaluation_n: int = Field(ge=1)
    event_n: int = Field(ge=1)
    non_event_n: int = Field(ge=1)
    evaluation_subject_n: int = Field(ge=1)
    repeated_subject_n: int = Field(ge=0)
    clipped_probability_n: int = Field(ge=0)
    event_rate: float = Field(ge=0.0, le=1.0)
    mean_predicted_probability: float = Field(ge=0.0, le=1.0)
    auroc: float = Field(ge=0.0, le=1.0)
    brier_score: float = Field(ge=0.0, le=1.0)
    calibration_status: CalibrationStatus
    calibration_intercept: float | None = None
    calibration_slope: float | None = None

    @model_validator(mode="after")
    def _calibration_values_match_status(self) -> "PredictionValidationSummary":
        values = (self.calibration_intercept, self.calibration_slope)
        if self.calibration_status == "estimated":
            if any(value is None or not math.isfinite(value) for value in values):
                raise ValueError(
                    "estimated calibration requires finite intercept and slope"
                )
        elif any(value is not None for value in values):
            raise ValueError(
                "non-estimable calibration must not publish intercept or slope"
            )
        if self.event_n + self.non_event_n != self.evaluation_n:
            raise ValueError("event and non-event counts must sum to evaluation_n")
        if self.evaluation_subject_n > self.evaluation_n:
            raise ValueError("evaluation_subject_n cannot exceed evaluation_n")
        return self


class PredictionCalibrationBin(BaseModel):
    """One denominator-bearing quantile calibration bin."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    bin_index: int = Field(ge=1)
    n: int = Field(ge=1)
    event_n: int = Field(ge=0)
    mean_predicted_probability: float = Field(ge=0.0, le=1.0)
    observed_event_rate: float = Field(ge=0.0, le=1.0)
    minimum_predicted_probability: float = Field(ge=0.0, le=1.0)
    maximum_predicted_probability: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _counts_and_bounds_reconcile(self) -> "PredictionCalibrationBin":
        if self.event_n > self.n:
            raise ValueError("calibration-bin event_n cannot exceed n")
        if self.minimum_predicted_probability > self.maximum_predicted_probability:
            raise ValueError("calibration-bin probability bounds are reversed")
        return self


class PredictionThresholdMetric(BaseModel):
    """Confusion counts and operating characteristics at one threshold."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    threshold: float = Field(gt=0.0, lt=1.0)
    n: int = Field(ge=1)
    predicted_positive_n: int = Field(ge=0)
    true_positive_n: int = Field(ge=0)
    false_positive_n: int = Field(ge=0)
    true_negative_n: int = Field(ge=0)
    false_negative_n: int = Field(ge=0)
    sensitivity: float = Field(ge=0.0, le=1.0)
    specificity: float = Field(ge=0.0, le=1.0)
    positive_predictive_value: float | None
    negative_predictive_value: float | None

    @model_validator(mode="after")
    def _confusion_counts_reconcile(self) -> "PredictionThresholdMetric":
        if (
            self.true_positive_n
            + self.false_positive_n
            + self.true_negative_n
            + self.false_negative_n
            != self.n
        ):
            raise ValueError("threshold confusion counts must sum to n")
        if self.predicted_positive_n != self.true_positive_n + self.false_positive_n:
            raise ValueError("predicted_positive_n does not match confusion counts")
        return self


class PredictionValidationResult(BaseModel):
    """Versioned result emitted by the experimental deterministic owner."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["easyicu.prediction_validation_result/1"] = (
        "easyicu.prediction_validation_result/1"
    )
    contract_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    summary: PredictionValidationSummary
    calibration_bins: tuple[PredictionCalibrationBin, ...] = Field(min_length=1)
    threshold_metrics: tuple[PredictionThresholdMetric, ...] = Field(min_length=1)


def prediction_validation_result_sha256(
    result: PredictionValidationResult | Mapping[str, Any],
) -> str:
    """Return the canonical digest of one normalized result."""

    parsed = PredictionValidationResult.model_validate(result)
    return canonical_sha256(parsed.model_dump(mode="json"))


class PredictionValidationSourceBinding(BaseModel):
    """Portable identity of the exact CSV bytes consumed by the owner."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["easyicu.prediction_validation_source/1"] = (
        "easyicu.prediction_validation_source/1"
    )
    source_format: Literal["csv_utf8"] = "csv_utf8"
    parser: Literal["pandas.read_csv"] = "pandas.read_csv"
    parser_version: str
    source_artifact_name: str
    source_artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_artifact_size_bytes: int = Field(ge=1)
    source_row_count: int = Field(ge=1)
    source_columns: tuple[str, ...] = Field(min_length=1)

    @field_validator("parser_version")
    @classmethod
    def _nonempty_parser_version(cls, value: str) -> str:
        parsed = str(value or "").strip()
        if not parsed:
            raise ValueError("parser_version must be non-empty")
        return parsed

    @field_validator("source_artifact_name")
    @classmethod
    def _portable_csv_name(cls, value: str) -> str:
        parsed = str(value or "").strip()
        if (
            not parsed
            or "/" in parsed
            or "\\" in parsed
            or not parsed.lower().endswith(".csv")
        ):
            raise ValueError("source_artifact_name must be one portable CSV basename")
        return parsed

    @field_validator("source_columns")
    @classmethod
    def _closed_source_columns(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        parsed = tuple(str(value or "").strip() for value in values)
        if any(not value for value in parsed) or len(parsed) != len(set(parsed)):
            raise ValueError("source_columns must be non-empty and unique")
        return parsed


class PredictionValidationReceipt(BaseModel):
    """Digest-bound source, declaration and deterministic result bundle."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["easyicu.prediction_validation_receipt/1"] = (
        "easyicu.prediction_validation_receipt/1"
    )
    issuer: Literal["easyicu.prediction_validation/1"] = (
        "easyicu.prediction_validation/1"
    )
    execution_mode: Literal["experimental_deterministic"] = "experimental_deterministic"
    paper_authorization: Literal[False] = False
    source: PredictionValidationSourceBinding
    contract_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    result_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    result: PredictionValidationResult

    @model_validator(mode="after")
    def _bindings_reconcile(self) -> "PredictionValidationReceipt":
        if self.contract_sha256 != self.result.contract_sha256:
            raise ValueError("receipt contract digest does not match its result")
        if self.result_sha256 != prediction_validation_result_sha256(self.result):
            raise ValueError("receipt result digest does not match its result")
        if self.source.source_row_count != self.result.summary.input_n:
            raise ValueError("receipt source row count does not match its result")
        return self


def prediction_validation_receipt_sha256(
    receipt: PredictionValidationReceipt | Mapping[str, Any],
) -> str:
    """Return the canonical digest of one normalized provenance receipt."""

    parsed = PredictionValidationReceipt.model_validate(receipt)
    return canonical_sha256(parsed.model_dump(mode="json"))


class PredictionValidationFinding(BaseModel):
    """Structured result-validation finding owned by this boundary."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    validator: Literal["prediction_validation_contract"] = (
        "prediction_validation_contract"
    )
    reason_code: PredictionValidationReason
    message: str
    detail: dict[str, Any] = Field(default_factory=dict)


__all__ = [
    "CalibrationStatus",
    "PredictionCalibrationBin",
    "PredictionThresholdMetric",
    "PredictionValidationError",
    "PredictionValidationFinding",
    "PredictionValidationReason",
    "PredictionValidationReceipt",
    "PredictionValidationResult",
    "PredictionValidationSourceBinding",
    "PredictionValidationSpec",
    "PredictionValidationSummary",
    "prediction_validation_receipt_sha256",
    "prediction_validation_result_sha256",
    "prediction_validation_spec_sha256",
]

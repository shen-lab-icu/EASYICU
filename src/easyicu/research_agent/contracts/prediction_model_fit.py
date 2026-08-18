"""Typed contract for one experimental, train-only binary model fit.

The declaration fixes every currently supported preprocessing and estimator
choice.  It does not select a cohort, split subjects, tune a model, register an
EvidenceStore artifact, or grant paper authority.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..canonical_json import canonical_json_bytes, canonical_sha256, sha256_bytes


class PredictionModelFitReason(str, Enum):
    """Stable owner diagnostics for prediction-model fit failures."""

    SPEC_INVALID = "prediction_model_fit_spec_invalid"
    SOURCE_INPUT_INVALID = "prediction_model_fit_source_input_invalid"
    SOURCE_IDENTITY_MISMATCH = "prediction_model_fit_source_identity_mismatch"
    MISSING_COLUMNS = "prediction_model_fit_missing_columns"
    EMPTY_INPUT = "prediction_model_fit_empty_input"
    IDENTITY_INVALID = "prediction_model_fit_identity_invalid"
    DUPLICATE_UNIT = "prediction_model_fit_duplicate_unit"
    SPLIT_INVALID = "prediction_model_fit_split_invalid"
    SUBJECT_SPLIT_LEAKAGE = "prediction_model_fit_subject_split_leakage"
    SUBJECT_UNIT_NOT_UNIQUE = "prediction_model_fit_subject_unit_not_unique"
    TRAINING_SPLIT_MISSING = "prediction_model_fit_training_split_missing"
    EVALUATION_SPLIT_MISSING = "prediction_model_fit_evaluation_split_missing"
    OUTCOME_INVALID = "prediction_model_fit_outcome_invalid"
    TRAINING_SINGLE_CLASS = "prediction_model_fit_training_single_class"
    FEATURE_NONNUMERIC = "prediction_model_fit_feature_nonnumeric"
    FEATURE_NONFINITE = "prediction_model_fit_feature_nonfinite"
    FEATURE_ALL_MISSING_TRAIN = "prediction_model_fit_feature_all_missing_train"
    FIT_FAILED = "prediction_model_fit_failed"
    PREDICTION_INVALID = "prediction_model_fit_prediction_invalid"
    RUNTIME_IDENTITY_INVALID = "prediction_model_fit_runtime_identity_invalid"
    BUNDLE_INVALID = "prediction_model_fit_bundle_invalid"
    RECOMPUTATION_MISMATCH = "prediction_model_fit_recomputation_mismatch"


class PredictionModelFitError(ValueError):
    """Typed refusal raised by the prediction-model fit owner."""

    owner = "easyicu.prediction_model_fit_owner"
    phase = "prediction_model_fit"

    def __init__(
        self,
        reason_code: PredictionModelFitReason,
        message: str,
        **detail: Any,
    ) -> None:
        self.reason_code = reason_code
        self.detail = dict(detail)
        super().__init__(f"{reason_code.value}: {message}")


class PredictionModelFitSpec(BaseModel):
    """Fully fixed coordinates for the narrow v1 binary fit owner."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    schema_version: Literal["easyicu.prediction_model_fit/1"] = (
        "easyicu.prediction_model_fit/1"
    )
    model_identifier: str
    unit_id_column: str
    subject_id_column: str
    split_column: str
    outcome_column: str
    probability_column: str
    feature_columns: tuple[str, ...] = Field(min_length=1, max_length=100)
    training_split: str
    evaluation_split: str
    analysis_unit: Literal["subject"] = "subject"
    preprocessing: Literal["numeric_median_then_standardize"] = (
        "numeric_median_then_standardize"
    )
    estimator: Literal["logistic_regression_l2"] = "logistic_regression_l2"
    solver: Literal["lbfgs"] = "lbfgs"
    regularization_c: float = Field(default=1.0, gt=0.0)
    fit_intercept: Literal[True] = True
    class_weight: Literal[None] = None
    max_iter: int = Field(default=1000, ge=10, le=100_000)
    tolerance: float = Field(default=1e-8, gt=0.0, le=1.0)

    @field_validator(
        "model_identifier",
        "unit_id_column",
        "subject_id_column",
        "split_column",
        "outcome_column",
        "probability_column",
        "training_split",
        "evaluation_split",
    )
    @classmethod
    def _canonical_nonempty_text(cls, value: str) -> str:
        if not isinstance(value, str) or not value or value != value.strip():
            raise ValueError("model-fit coordinates must be canonical non-empty text")
        return value

    @field_validator("feature_columns")
    @classmethod
    def _canonical_feature_columns(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        parsed = tuple(values)
        if any(
            not isinstance(value, str) or not value or value != value.strip()
            for value in parsed
        ):
            raise ValueError("feature columns must be canonical non-empty text")
        if len(parsed) != len(set(parsed)):
            raise ValueError("feature columns must be unique")
        return parsed

    @model_validator(mode="after")
    def _distinct_coordinates(self) -> "PredictionModelFitSpec":
        coordinates = {
            self.unit_id_column,
            self.subject_id_column,
            self.split_column,
            self.outcome_column,
            self.probability_column,
        }
        if len(coordinates) != 5:
            raise ValueError("identity, split, outcome, and probability columns differ")
        overlap = sorted(coordinates & set(self.feature_columns))
        if overlap:
            raise ValueError(f"feature columns overlap model coordinates: {overlap}")
        if self.training_split == self.evaluation_split:
            raise ValueError("training and evaluation split labels must differ")
        return self


class PredictionPackageVersion(BaseModel):
    """One package version captured by the deterministic producer."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    distribution: str = Field(min_length=1)
    version: str = Field(min_length=1)


class PredictionPreprocessingArtifact(BaseModel):
    """Exact train-fitted median and standardization state."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    strategy: Literal["numeric_median_then_standardize"]
    fit_scope: Literal["training_subjects_only"]
    feature_columns: tuple[str, ...] = Field(min_length=1)
    medians: tuple[float, ...] = Field(min_length=1)
    means: tuple[float, ...] = Field(min_length=1)
    scales: tuple[float, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _aligned_finite_state(self) -> "PredictionPreprocessingArtifact":
        width = len(self.feature_columns)
        if not all(
            len(values) == width for values in (self.medians, self.means, self.scales)
        ):
            raise ValueError("preprocessing state must align with feature columns")
        numeric = (*self.medians, *self.means, *self.scales)
        if any(not math.isfinite(value) for value in numeric):
            raise ValueError("preprocessing state must be finite")
        if any(value <= 0.0 for value in self.scales):
            raise ValueError("preprocessing scales must be positive")
        return self


class PredictionLogisticArtifact(BaseModel):
    """Exact train-fitted L2 logistic regression state."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    estimator: Literal["logistic_regression_l2"]
    solver: Literal["lbfgs"]
    fit_scope: Literal["training_subjects_only"]
    feature_columns: tuple[str, ...] = Field(min_length=1)
    classes: tuple[Literal[0], Literal[1]] = (0, 1)
    coefficients: tuple[float, ...] = Field(min_length=1)
    intercept: float
    regularization_c: float = Field(gt=0.0)
    fit_intercept: Literal[True]
    class_weight: Literal[None]
    max_iter: int = Field(ge=10)
    tolerance: float = Field(gt=0.0)
    n_iter: int = Field(ge=1)

    @model_validator(mode="after")
    def _aligned_finite_state(self) -> "PredictionLogisticArtifact":
        if len(self.coefficients) != len(self.feature_columns):
            raise ValueError("model coefficients must align with feature columns")
        if any(
            not math.isfinite(value) for value in (*self.coefficients, self.intercept)
        ):
            raise ValueError("model coefficients and intercept must be finite")
        if self.n_iter > self.max_iter:
            raise ValueError("model iteration count exceeds its declared maximum")
        return self


class PredictionModelArtifact(BaseModel):
    """Auditable model JSON emitted by the experimental owner."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["easyicu.prediction_model_artifact/1"] = (
        "easyicu.prediction_model_artifact/1"
    )
    issuer: Literal["easyicu.prediction_model_fit_owner"]
    model_identifier: str = Field(min_length=1)
    contract_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_input_receipt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    training_subjects_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    evaluation_subjects_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    split_assignment_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    analysis_unit: Literal["subject"]
    preprocessing: PredictionPreprocessingArtifact
    estimator: PredictionLogisticArtifact
    authority_scope: Literal["analysis_only"]
    paper_authorization_allowed: Literal[False]

    @model_validator(mode="after")
    def _same_feature_roster(self) -> "PredictionModelArtifact":
        if self.preprocessing.feature_columns != self.estimator.feature_columns:
            raise ValueError("preprocessing and estimator feature rosters differ")
        return self


class PredictionModelFitReceipt(BaseModel):
    """Self-digesting binding over source, model, and prediction bytes."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        allow_inf_nan=False,
    )

    schema_version: Literal["easyicu.prediction_model_fit_receipt/1"]
    issuer: Literal["easyicu.prediction_model_fit_owner"]
    execution_mode: Literal["experimental"]
    authority_scope: Literal["analysis_only"]
    paper_authorization_allowed: Literal[False]
    source_input_receipt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_loaded_frame_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_row_identity_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    contract_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    training_split: str = Field(min_length=1)
    evaluation_split: str = Field(min_length=1)
    input_n: int = Field(ge=2)
    training_n: int = Field(ge=2)
    evaluation_n: int = Field(ge=1)
    training_event_n: int = Field(ge=1)
    evaluation_event_n: int = Field(ge=0)
    training_subjects_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    evaluation_subjects_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    split_assignment_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    preprocessing_fit_scope: Literal["training_subjects_only"]
    model_fit_scope: Literal["training_subjects_only"]
    model_artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    model_artifact_size_bytes: int = Field(ge=1)
    prediction_table_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    prediction_table_size_bytes: int = Field(ge=1)
    prediction_table_row_count: int = Field(ge=2)
    prediction_table_columns: tuple[str, ...] = Field(min_length=5, max_length=5)
    package_versions: tuple[PredictionPackageVersion, ...] = Field(min_length=1)
    receipt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def _reconcile_and_verify(self) -> "PredictionModelFitReceipt":
        if self.training_split == self.evaluation_split:
            raise ValueError("training and evaluation split labels must differ")
        if self.training_n + self.evaluation_n != self.input_n:
            raise ValueError("training and evaluation counts must sum to input_n")
        if self.prediction_table_row_count != self.input_n:
            raise ValueError("prediction-table rows must equal input_n")
        if self.training_event_n >= self.training_n:
            raise ValueError("training data must contain both outcome classes")
        if self.evaluation_event_n > self.evaluation_n:
            raise ValueError("evaluation event count exceeds evaluation_n")
        if len(set(self.prediction_table_columns)) != 5:
            raise ValueError("prediction-table columns must be distinct")
        package_names = tuple(item.distribution for item in self.package_versions)
        if package_names != tuple(sorted(package_names)) or len(package_names) != len(
            set(package_names)
        ):
            raise ValueError("package versions must be unique and sorted")
        if prediction_model_fit_receipt_sha256(self) != self.receipt_sha256:
            raise ValueError("prediction-model fit receipt SHA-256 mismatch")
        return self


def prediction_model_fit_spec_sha256(
    spec: PredictionModelFitSpec | Mapping[str, Any],
) -> str:
    """Return the canonical digest of the exact fit declaration."""

    parsed = (
        spec
        if isinstance(spec, PredictionModelFitSpec)
        else PredictionModelFitSpec.model_validate(spec)
    )
    return canonical_sha256(parsed.model_dump(mode="json"))


def prediction_model_artifact_bytes(
    artifact: PredictionModelArtifact | Mapping[str, Any],
) -> bytes:
    """Serialize one model artifact as canonical UTF-8 JSON."""

    parsed = (
        artifact
        if isinstance(artifact, PredictionModelArtifact)
        else PredictionModelArtifact.model_validate(artifact)
    )
    return canonical_json_bytes(parsed.model_dump(mode="json"), trailing_newline=True)


def prediction_model_artifact_sha256(
    artifact: PredictionModelArtifact | Mapping[str, Any],
) -> str:
    """Return the digest of the exact canonical model artifact bytes."""

    return sha256_bytes(prediction_model_artifact_bytes(artifact))


def prediction_model_fit_receipt_sha256(
    receipt: PredictionModelFitReceipt | Mapping[str, Any],
) -> str:
    """Return the receipt self-digest, excluding the digest field itself."""

    payload = (
        receipt.model_dump(mode="json")
        if isinstance(receipt, PredictionModelFitReceipt)
        else dict(receipt)
    )
    payload.pop("receipt_sha256", None)
    return canonical_sha256(payload)


__all__ = [
    "PredictionLogisticArtifact",
    "PredictionModelArtifact",
    "PredictionModelFitError",
    "PredictionModelFitReason",
    "PredictionModelFitReceipt",
    "PredictionModelFitSpec",
    "PredictionPackageVersion",
    "PredictionPreprocessingArtifact",
    "prediction_model_artifact_bytes",
    "prediction_model_artifact_sha256",
    "prediction_model_fit_receipt_sha256",
    "prediction_model_fit_spec_sha256",
]

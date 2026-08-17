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
    LINEAGE_SCHEMA_INVALID = "prediction_validation_lineage_schema_invalid"
    LINEAGE_EVIDENCE_MISSING = "prediction_validation_lineage_evidence_missing"
    LINEAGE_EVIDENCE_MISMATCH = "prediction_validation_lineage_evidence_mismatch"
    LINEAGE_EVIDENCE_STALE = "prediction_validation_lineage_evidence_stale"
    LINEAGE_COHORT_MISMATCH = "prediction_validation_lineage_cohort_mismatch"
    LINEAGE_SPLIT_MISMATCH = "prediction_validation_lineage_split_mismatch"
    LINEAGE_RUNTIME_MISMATCH = "prediction_validation_lineage_runtime_mismatch"
    VALIDATION_SEAL_INVALID = "prediction_validation_host_seal_invalid"
    AUTHORITY_CEILING_VIOLATION = "prediction_validation_authority_ceiling_violation"


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

    payload = (
        spec.model_dump(mode="python")
        if isinstance(spec, PredictionValidationSpec)
        else spec
    )
    parsed = PredictionValidationSpec.model_validate(payload)
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

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

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

    payload = (
        result.model_dump(mode="python")
        if isinstance(result, PredictionValidationResult)
        else result
    )
    parsed = PredictionValidationResult.model_validate(payload)
    return canonical_sha256(parsed.model_dump(mode="json"))


class PredictionValidationSourceBinding(BaseModel):
    """Portable identity of the exact CSV bytes consumed by the owner."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["easyicu.prediction_validation_source/1"] = (
        "easyicu.prediction_validation_source/1"
    )
    source_format: Literal["csv_utf8"] = "csv_utf8"
    parser: Literal["pandas.read_csv"] = "pandas.read_csv"
    parser_profile: Literal["easyicu.prediction_validation_csv_strict/1"] = (
        "easyicu.prediction_validation_csv_strict/1"
    )
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
        parsed = tuple(str(value or "") for value in values)
        if any(not value or value != value.strip() for value in parsed) or len(
            parsed
        ) != len(set(parsed)):
            raise ValueError(
                "source_columns must be non-empty, whitespace-canonical and unique"
            )
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

    payload = (
        receipt.model_dump(mode="python")
        if isinstance(receipt, PredictionValidationReceipt)
        else receipt
    )
    parsed = PredictionValidationReceipt.model_validate(payload)
    return canonical_sha256(parsed.model_dump(mode="json"))


class _StrictFrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class PredictionValidationRuntimeIdentity(_StrictFrozenModel):
    """Exact clean code and runtime coordinates used for host recomputation."""

    schema_version: Literal["easyicu.prediction_validation_runtime/1"] = (
        "easyicu.prediction_validation_runtime/1"
    )
    git_commit: str = Field(pattern=r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
    git_dirty: Literal[False] = False
    source_tree_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    environment_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    runtime_kind: Literal["container", "local_process"]
    container_image_digest: str | None = Field(
        default=None,
        pattern=r"^sha256:[0-9a-f]{64}$",
    )
    python_version: str = Field(min_length=1)
    package_version: str = Field(min_length=1)

    @field_validator("python_version", "package_version")
    @classmethod
    def _canonical_runtime_text(cls, value: str) -> str:
        if value != value.strip():
            raise ValueError("runtime version fields must be whitespace-canonical")
        return value

    @model_validator(mode="after")
    def _runtime_kind_matches_image(self) -> "PredictionValidationRuntimeIdentity":
        if self.runtime_kind == "container" and self.container_image_digest is None:
            raise ValueError("container runtime requires an image digest")
        if self.runtime_kind == "local_process" and self.container_image_digest:
            raise ValueError("local_process runtime cannot claim a container image")
        return self


def prediction_validation_runtime_identity_sha256(
    runtime: PredictionValidationRuntimeIdentity | Mapping[str, Any],
) -> str:
    """Return the canonical digest of one exact runtime identity."""

    payload = (
        runtime.model_dump(mode="python")
        if isinstance(runtime, PredictionValidationRuntimeIdentity)
        else runtime
    )
    parsed = PredictionValidationRuntimeIdentity.model_validate(payload)
    return canonical_sha256(parsed.model_dump(mode="json"))


PredictionValidationArtifactRole = Literal[
    "prediction_table",
    "cohort",
    "split_assignment",
    "model_artifact",
    "code_snapshot",
    "environment_lock",
    "runtime_receipt",
]

_PREDICTION_LINEAGE_ROLE_KINDS = {
    "prediction_table": "table",
    "cohort": "table",
    "split_assignment": "table",
    "model_artifact": "code",
    "code_snapshot": "code",
    "environment_lock": "code",
    "runtime_receipt": "log",
}


class PredictionValidationArtifactBinding(_StrictFrozenModel):
    """Exact EvidenceStore record expected for one upstream lineage role."""

    role: PredictionValidationArtifactRole
    evidence_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    kind: Literal["table", "code", "log"]
    produced_by_step: str = Field(min_length=1)

    @field_validator("produced_by_step")
    @classmethod
    def _canonical_step_id(cls, value: str) -> str:
        if value != value.strip():
            raise ValueError("produced_by_step must be whitespace-canonical")
        return value

    @model_validator(mode="after")
    def _role_uses_expected_kind(self) -> "PredictionValidationArtifactBinding":
        if self.kind != _PREDICTION_LINEAGE_ROLE_KINDS[self.role]:
            raise ValueError(
                f"{self.role} requires kind {_PREDICTION_LINEAGE_ROLE_KINDS[self.role]}"
            )
        return self


class PredictionValidationUpstreamLineage(_StrictFrozenModel):
    """Closed, byte-bound lineage for an already-produced probability table."""

    schema_version: Literal["easyicu.prediction_validation_lineage/1"] = (
        "easyicu.prediction_validation_lineage/1"
    )
    producer_run_id: str = Field(min_length=1)
    model_identifier: str = Field(min_length=1)
    evaluation_split: str = Field(min_length=1)
    split_policy: Literal["subject_disjoint"] = "subject_disjoint"
    runtime: PredictionValidationRuntimeIdentity
    artifacts: tuple[PredictionValidationArtifactBinding, ...] = Field(
        min_length=7,
        max_length=7,
    )

    @field_validator("producer_run_id", "model_identifier", "evaluation_split")
    @classmethod
    def _canonical_lineage_text(cls, value: str) -> str:
        if value != value.strip():
            raise ValueError("lineage text fields must be whitespace-canonical")
        return value

    @model_validator(mode="after")
    def _lineage_roles_and_runtime_reconcile(
        self,
    ) -> "PredictionValidationUpstreamLineage":
        by_role = {binding.role: binding for binding in self.artifacts}
        if set(by_role) != set(_PREDICTION_LINEAGE_ROLE_KINDS):
            raise ValueError("lineage requires each closed artifact role exactly once")
        if tuple(binding.role for binding in self.artifacts) != tuple(
            _PREDICTION_LINEAGE_ROLE_KINDS
        ):
            raise ValueError("lineage artifact roles must use canonical order")
        evidence_ids = [binding.evidence_id for binding in self.artifacts]
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("lineage evidence ids must be unique")
        if by_role["code_snapshot"].sha256 != self.runtime.source_tree_sha256:
            raise ValueError("code snapshot does not match runtime source tree")
        if by_role["environment_lock"].sha256 != self.runtime.environment_sha256:
            raise ValueError("environment lock does not match runtime identity")
        return self

    def binding_for(
        self,
        role: PredictionValidationArtifactRole,
    ) -> PredictionValidationArtifactBinding:
        return next(binding for binding in self.artifacts if binding.role == role)


def prediction_validation_upstream_lineage_sha256(
    lineage: PredictionValidationUpstreamLineage | Mapping[str, Any],
) -> str:
    """Return the canonical digest of one closed upstream lineage."""

    payload = (
        lineage.model_dump(mode="python")
        if isinstance(lineage, PredictionValidationUpstreamLineage)
        else lineage
    )
    parsed = PredictionValidationUpstreamLineage.model_validate(payload)
    return canonical_sha256(parsed.model_dump(mode="json"))


class PredictionValidationHostValidationSeal(_StrictFrozenModel):
    """Host receipt that a candidate was fully recomputed at one runtime."""

    schema_version: Literal["easyicu.prediction_validation_host_seal/1"] = (
        "easyicu.prediction_validation_host_seal/1"
    )
    validator: Literal["easyicu.prediction_validation_receipt_recompute/1"] = (
        "easyicu.prediction_validation_receipt_recompute/1"
    )
    receipt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    contract_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    result_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    runtime_identity_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    finding_count: Literal[0] = 0
    paper_authorization: Literal[False] = False


class PredictionValidationAnalysisPolicy(_StrictFrozenModel):
    """Closed authority ceiling for an incubator EvidenceStore artifact."""

    schema_version: Literal["easyicu.prediction_validation_analysis_policy/1"] = (
        "easyicu.prediction_validation_analysis_policy/1"
    )
    claim_ceiling: Literal["analysis_only"] = "analysis_only"
    paper_authorization: Literal[False] = False
    planner_selection_authorized: Literal[False] = False
    numeric_claim_registration_authorized: Literal[False] = False
    scientific_claim_registration_authorized: Literal[False] = False
    alias_publication_authorized: Literal[False] = False


class PredictionValidationAnalysisBundle(_StrictFrozenModel):
    """Self-contained result, lineage, host seal, and authority ceiling."""

    schema_version: Literal["easyicu.prediction_validation_analysis_bundle/1"] = (
        "easyicu.prediction_validation_analysis_bundle/1"
    )
    spec: PredictionValidationSpec
    receipt: PredictionValidationReceipt
    validation_seal: PredictionValidationHostValidationSeal
    lineage: PredictionValidationUpstreamLineage
    policy: PredictionValidationAnalysisPolicy = Field(
        default_factory=PredictionValidationAnalysisPolicy
    )

    @model_validator(mode="after")
    def _bundle_bindings_reconcile(self) -> "PredictionValidationAnalysisBundle":
        receipt_sha256 = prediction_validation_receipt_sha256(self.receipt)
        runtime_sha256 = prediction_validation_runtime_identity_sha256(
            self.lineage.runtime
        )
        if prediction_validation_spec_sha256(self.spec) != self.receipt.contract_sha256:
            raise ValueError("bundle spec does not match receipt contract")
        if self.validation_seal.receipt_sha256 != receipt_sha256:
            raise ValueError("host seal does not match bundle receipt")
        if (
            self.validation_seal.source_artifact_sha256
            != self.receipt.source.source_artifact_sha256
            or self.validation_seal.contract_sha256 != self.receipt.contract_sha256
            or self.validation_seal.result_sha256 != self.receipt.result_sha256
            or self.validation_seal.runtime_identity_sha256 != runtime_sha256
        ):
            raise ValueError("host seal coordinates do not match bundle authority")
        if (
            self.lineage.binding_for("prediction_table").sha256
            != self.receipt.source.source_artifact_sha256
        ):
            raise ValueError("prediction-table lineage does not match receipt source")
        if self.lineage.evaluation_split != self.spec.evaluation_split:
            raise ValueError("lineage evaluation split does not match the contract")
        return self


def prediction_validation_analysis_bundle_sha256(
    bundle: PredictionValidationAnalysisBundle | Mapping[str, Any],
) -> str:
    """Return the canonical digest of one analysis-only evidence bundle."""

    payload = (
        bundle.model_dump(mode="python")
        if isinstance(bundle, PredictionValidationAnalysisBundle)
        else bundle
    )
    parsed = PredictionValidationAnalysisBundle.model_validate(payload)
    return canonical_sha256(parsed.model_dump(mode="json"))


class PredictionValidationAnalysisRegistration(_StrictFrozenModel):
    """EvidenceStore registration receipt for one analysis-only bundle."""

    schema_version: Literal["easyicu.prediction_validation_analysis_registration/1"] = (
        "easyicu.prediction_validation_analysis_registration/1"
    )
    evidence_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
    evidence_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    bundle_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    lineage_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    producer_run_id: str = Field(min_length=1)
    validation_step_id: str = Field(min_length=1)
    upstream_evidence_ids: tuple[str, ...] = Field(min_length=7, max_length=7)
    claim_ceiling: Literal["analysis_only"] = "analysis_only"
    paper_authorization: Literal[False] = False
    planner_selection_authorized: Literal[False] = False
    aliases_published: Literal[False] = False
    numeric_claim_count_delta: Literal[0] = 0
    scientific_claim_count_delta: Literal[0] = 0

    @field_validator("producer_run_id", "validation_step_id")
    @classmethod
    def _canonical_registration_text(cls, value: str) -> str:
        if value != value.strip():
            raise ValueError("registration text fields must be whitespace-canonical")
        return value

    @model_validator(mode="after")
    def _registration_identity_reconciles(
        self,
    ) -> "PredictionValidationAnalysisRegistration":
        expected = f"prediction_validation_analysis_{self.bundle_sha256[:12]}"
        if self.evidence_id != expected:
            raise ValueError("registration evidence id does not match bundle digest")
        if len(self.upstream_evidence_ids) != len(set(self.upstream_evidence_ids)):
            raise ValueError("registration upstream evidence ids must be unique")
        return self


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
    "PredictionValidationAnalysisBundle",
    "PredictionValidationAnalysisPolicy",
    "PredictionValidationAnalysisRegistration",
    "PredictionValidationArtifactBinding",
    "PredictionValidationArtifactRole",
    "PredictionCalibrationBin",
    "PredictionValidationHostValidationSeal",
    "PredictionValidationRuntimeIdentity",
    "PredictionThresholdMetric",
    "PredictionValidationError",
    "PredictionValidationFinding",
    "PredictionValidationReason",
    "PredictionValidationReceipt",
    "PredictionValidationResult",
    "PredictionValidationSourceBinding",
    "PredictionValidationSpec",
    "PredictionValidationSummary",
    "PredictionValidationUpstreamLineage",
    "prediction_validation_analysis_bundle_sha256",
    "prediction_validation_receipt_sha256",
    "prediction_validation_result_sha256",
    "prediction_validation_runtime_identity_sha256",
    "prediction_validation_spec_sha256",
    "prediction_validation_upstream_lineage_sha256",
]

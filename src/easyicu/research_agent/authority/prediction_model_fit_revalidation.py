"""Re-fit a persisted V5 analysis from current EvidenceStore bytes."""

from __future__ import annotations

from collections.abc import Mapping
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ..canonical_json import canonical_json, canonical_sha256, sha256_bytes
from ..contracts.prediction_model_fit import (
    PredictionModelFitError,
    prediction_model_artifact_bytes,
)
from ..contracts.prediction_validation import (
    PredictionValidationError,
    prediction_validation_runtime_identity_sha256,
    prediction_validation_upstream_lineage_sha256,
)
from ..prediction_model_fit_owner import (
    revalidate_prediction_model_fit_persisted_artifacts,
)
from .evidence_store import EvidenceStore
from .prediction_model_fit_evidence import (
    PredictionModelFitEvidenceEnvelope,
    PredictionModelFitRuntimeAuthority,
    PredictionModelFitValidationRegistration,
)
from .prediction_model_fit_runtime import (
    revalidate_prediction_model_fit_runtime_authority,
)
from .prediction_validation_evidence import (
    prediction_validation_analysis_registration_findings,
    resolve_prediction_validation_artifact_bindings,
)


class PredictionModelFitPersistedValidationReason(str, Enum):
    """Stable failures owned by persisted fit revalidation."""

    INPUT_SCHEMA_INVALID = (
        "prediction_model_fit_persisted_validation_input_schema_invalid"
    )
    EVIDENCE_INVALID = "prediction_model_fit_persisted_validation_evidence_invalid"
    ENVELOPE_MISMATCH = "prediction_model_fit_persisted_validation_envelope_mismatch"
    REFIT_MISMATCH = "prediction_model_fit_persisted_validation_refit_mismatch"
    AUTHORITY_CEILING_VIOLATION = (
        "prediction_model_fit_persisted_validation_authority_ceiling_violation"
    )


class PredictionModelFitPersistedValidationError(RuntimeError):
    """Typed refusal raised while re-fitting persisted V5 evidence."""

    owner = "easyicu.prediction_model_fit_revalidation"
    phase = "prediction_model_fit_persisted_revalidation"

    def __init__(
        self,
        reason_code: PredictionModelFitPersistedValidationReason,
        message: str,
        **detail: Any,
    ) -> None:
        self.reason_code = reason_code
        self.detail = dict(detail)
        super().__init__(f"{reason_code.value}: {message}")


class PredictionModelFitPersistedValidationReceipt(BaseModel):
    """Analysis-only receipt for one current-store full model re-fit."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["easyicu.prediction_model_fit_persisted_validation/1"] = (
        "easyicu.prediction_model_fit_persisted_validation/1"
    )
    issuer: Literal["easyicu.prediction_model_fit_revalidation"] = (
        "easyicu.prediction_model_fit_revalidation"
    )
    registration_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    lineage_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    runtime_identity_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    fit_receipt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_projection_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    model_evidence_envelope_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    prediction_table_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    analysis_evidence_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    claim_ceiling: Literal["analysis_only"] = "analysis_only"
    paper_authorization: Literal[False] = False
    planner_selection_authorized: Literal[False] = False


def _raise(
    reason_code: PredictionModelFitPersistedValidationReason,
    message: str,
    **detail: Any,
) -> None:
    raise PredictionModelFitPersistedValidationError(reason_code, message, **detail)


def _parse_registration(
    value: PredictionModelFitValidationRegistration | Mapping[str, Any],
) -> PredictionModelFitValidationRegistration:
    payload = (
        value.model_dump(mode="python")
        if isinstance(value, PredictionModelFitValidationRegistration)
        else value
    )
    try:
        return PredictionModelFitValidationRegistration.model_validate(payload)
    except ValidationError as error:
        raise PredictionModelFitPersistedValidationError(
            PredictionModelFitPersistedValidationReason.INPUT_SCHEMA_INVALID,
            "persisted fit registration is not schema-valid",
        ) from error


def _read_current_artifacts(
    *,
    evidence_store: EvidenceStore,
    registration: PredictionModelFitValidationRegistration,
) -> dict[str, bytes]:
    try:
        resolved = resolve_prediction_validation_artifact_bindings(
            evidence_store=evidence_store,
            producer_run_id=registration.lineage.producer_run_id,
            artifacts=registration.lineage.artifacts,
        )
        return {role: path.read_bytes() for role, path in resolved.items()}
    except (OSError, PredictionValidationError) as error:
        detail: dict[str, Any] = {}
        if isinstance(error, PredictionValidationError):
            detail = {
                "cause_reason_code": error.reason_code.value,
                "cause_detail": error.detail,
            }
        raise PredictionModelFitPersistedValidationError(
            PredictionModelFitPersistedValidationReason.EVIDENCE_INVALID,
            "current persisted fit lineage cannot be resolved",
            **detail,
        ) from error


def _parse_envelope(
    payload: bytes,
) -> PredictionModelFitEvidenceEnvelope:
    try:
        envelope = PredictionModelFitEvidenceEnvelope.model_validate_json(payload)
    except ValidationError as error:
        raise PredictionModelFitPersistedValidationError(
            PredictionModelFitPersistedValidationReason.ENVELOPE_MISMATCH,
            "persisted model evidence envelope is invalid",
        ) from error
    canonical = canonical_json(
        envelope.model_dump(mode="json"),
        trailing_newline=True,
    ).encode("utf-8")
    if canonical != payload:
        _raise(
            PredictionModelFitPersistedValidationReason.ENVELOPE_MISMATCH,
            "persisted model evidence envelope is not canonical",
        )
    return envelope


def revalidate_persisted_prediction_model_fit_validation(
    *,
    evidence_store: EvidenceStore,
    registration: PredictionModelFitValidationRegistration | Mapping[str, Any],
) -> PredictionModelFitPersistedValidationReceipt:
    """Re-observe runtime, re-fit source bytes, and re-run V3.1 validation."""

    if not isinstance(evidence_store, EvidenceStore):
        _raise(
            PredictionModelFitPersistedValidationReason.INPUT_SCHEMA_INVALID,
            "persisted fit revalidation requires an EvidenceStore",
        )
    parsed = _parse_registration(registration)
    aliases_before = dict(evidence_store.aliases())
    numeric_before = len(evidence_store.numeric_claims())
    scientific_before = len(evidence_store.scientific_claims())
    current = _read_current_artifacts(
        evidence_store=evidence_store,
        registration=parsed,
    )
    envelope = _parse_envelope(current["model_artifact"])
    expected_envelope_sha256 = sha256_bytes(current["model_artifact"])
    if (
        expected_envelope_sha256 != parsed.model_evidence_envelope_sha256
        or envelope.source_projection_sha256 != sha256_bytes(current["cohort"])
        or envelope.source_projection_size_bytes != len(current["cohort"])
        or envelope.split_assignment_artifact_sha256
        != sha256_bytes(current["split_assignment"])
        or envelope.split_assignment_artifact_size_bytes
        != len(current["split_assignment"])
        or envelope.prediction_table_sha256 != sha256_bytes(current["prediction_table"])
        or envelope.prediction_table_size_bytes != len(current["prediction_table"])
    ):
        _raise(
            PredictionModelFitPersistedValidationReason.ENVELOPE_MISMATCH,
            "persisted fit roles do not match the model evidence envelope",
        )
    runtime_authority = PredictionModelFitRuntimeAuthority(
        producer_run_id=parsed.lineage.producer_run_id,
        runtime=parsed.lineage.runtime,
        artifacts=parsed.lineage.artifacts[4:],
    )
    revalidate_prediction_model_fit_runtime_authority(
        evidence_store=evidence_store,
        runtime_authority=runtime_authority,
        fit_receipt=envelope.fit_receipt,
    )
    try:
        recomputed_receipt = revalidate_prediction_model_fit_persisted_artifacts(
            source_projection_csv_bytes=current["cohort"],
            spec=envelope.fit_spec,
            source_receipt=envelope.source_input_receipt,
            expected_fit_receipt=envelope.fit_receipt,
            model_artifact_bytes=prediction_model_artifact_bytes(
                envelope.model_artifact
            ),
            prediction_csv_bytes=current["prediction_table"],
        )
    except PredictionModelFitError as error:
        raise PredictionModelFitPersistedValidationError(
            PredictionModelFitPersistedValidationReason.REFIT_MISMATCH,
            "persisted prediction model does not match a full current re-fit",
            cause_reason_code=error.reason_code.value,
            cause_detail=error.detail,
        ) from error
    findings = prediction_validation_analysis_registration_findings(
        evidence_store=evidence_store,
        registration=parsed.analysis_registration,
    )
    if findings:
        _raise(
            PredictionModelFitPersistedValidationReason.EVIDENCE_INVALID,
            "persisted prediction validation no longer passes current-store checks",
            findings=[finding.model_dump(mode="json") for finding in findings],
        )
    if (
        evidence_store.aliases() != aliases_before
        or len(evidence_store.numeric_claims()) != numeric_before
        or len(evidence_store.scientific_claims()) != scientific_before
    ):
        _raise(
            PredictionModelFitPersistedValidationReason.AUTHORITY_CEILING_VIOLATION,
            "persisted fit revalidation changed an alias or claim registry",
        )
    return PredictionModelFitPersistedValidationReceipt(
        registration_sha256=canonical_sha256(parsed.model_dump(mode="json")),
        lineage_sha256=prediction_validation_upstream_lineage_sha256(parsed.lineage),
        runtime_identity_sha256=prediction_validation_runtime_identity_sha256(
            parsed.lineage.runtime
        ),
        fit_receipt_sha256=recomputed_receipt.receipt_sha256,
        source_projection_sha256=sha256_bytes(current["cohort"]),
        model_evidence_envelope_sha256=expected_envelope_sha256,
        prediction_table_sha256=sha256_bytes(current["prediction_table"]),
        analysis_evidence_sha256=parsed.analysis_registration.evidence_sha256,
        claim_ceiling="analysis_only",
        paper_authorization=False,
        planner_selection_authorized=False,
    )


__all__ = [
    "PredictionModelFitPersistedValidationError",
    "PredictionModelFitPersistedValidationReason",
    "PredictionModelFitPersistedValidationReceipt",
    "revalidate_persisted_prediction_model_fit_validation",
]

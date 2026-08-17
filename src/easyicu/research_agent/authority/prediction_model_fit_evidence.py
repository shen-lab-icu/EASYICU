"""Host-owned bridge from a sealed V4 fit to V3.1 analysis evidence.

The public route accepts no loose prediction table, model artifact, fit
receipt, validation receipt, seal, or seven-role lineage.  It revalidates the
sealed fit against the immutable typed input, materializes the four fit-owned
roles, joins three already-registered runtime-authority roles, and delegates
metric recomputation plus analysis-only registration to the existing V3.1
bridge.
"""

from __future__ import annotations

import csv
import io
from collections.abc import Mapping
from enum import Enum
from typing import Any, Literal, cast

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from ..canonical_json import canonical_json, canonical_sha256, sha256_bytes
from ..contracts.prediction_model_fit import (
    PredictionModelArtifact,
    PredictionModelFitReceipt,
    PredictionModelFitSpec,
    prediction_model_artifact_bytes,
    prediction_model_fit_spec_sha256,
)
from ..contracts.prediction_validation import (
    PredictionValidationAnalysisRegistration,
    PredictionValidationArtifactBinding,
    PredictionValidationRuntimeIdentity,
    PredictionValidationSpec,
    PredictionValidationUpstreamLineage,
)
from ..prediction_model_fit_owner import (
    PredictionModelFitBundle,
    revalidate_prediction_model_fit_bundle,
)
from ..schema import EvidenceRecord
from .evidence_store import EvidenceStore
from .prediction_validation_evidence import (
    register_prediction_validation_analysis_artifact,
    resolve_prediction_validation_runtime_authority,
)
from .typed_input_receipt import (
    TypedInputConsumptionReceipt,
    TypedInputRowIdentity,
)
from .typed_input_sdk import LoadedTypedInput


class PredictionModelFitEvidenceReason(str, Enum):
    """Stable failures owned by the fit-to-evidence bridge."""

    INPUT_SCHEMA_INVALID = "prediction_model_fit_evidence_input_schema_invalid"
    VALIDATION_CONTRACT_MISMATCH = (
        "prediction_model_fit_evidence_validation_contract_mismatch"
    )
    MATERIALIZATION_INVALID = "prediction_model_fit_evidence_materialization_invalid"
    AUTHORITY_CEILING_VIOLATION = (
        "prediction_model_fit_evidence_authority_ceiling_violation"
    )


class PredictionModelFitEvidenceError(ValueError):
    """Typed refusal raised by the fit-to-evidence bridge."""

    owner = "easyicu.prediction_model_fit_evidence"
    phase = "prediction_model_fit_evidence_registration"

    def __init__(
        self,
        reason_code: PredictionModelFitEvidenceReason,
        message: str,
        **detail: Any,
    ) -> None:
        self.reason_code = reason_code
        self.detail = dict(detail)
        super().__init__(f"{reason_code.value}: {message}")


class PredictionModelFitRuntimeAuthority(BaseModel):
    """Three existing runtime records allowed into the composite lineage."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["easyicu.prediction_model_fit_runtime_authority/1"] = (
        "easyicu.prediction_model_fit_runtime_authority/1"
    )
    producer_run_id: str = Field(min_length=1)
    runtime: PredictionValidationRuntimeIdentity
    artifacts: tuple[PredictionValidationArtifactBinding, ...] = Field(
        min_length=3,
        max_length=3,
    )

    @model_validator(mode="after")
    def _closed_runtime_roles(self) -> "PredictionModelFitRuntimeAuthority":
        if self.producer_run_id != self.producer_run_id.strip():
            raise ValueError("producer_run_id must be whitespace-canonical")
        roles = tuple(binding.role for binding in self.artifacts)
        expected = ("code_snapshot", "environment_lock", "runtime_receipt")
        if roles != expected:
            raise ValueError("runtime artifacts must use the closed canonical order")
        if self.artifacts[0].sha256 != self.runtime.source_tree_sha256:
            raise ValueError("code snapshot does not match runtime source identity")
        if self.artifacts[1].sha256 != self.runtime.environment_sha256:
            raise ValueError("environment lock does not match runtime identity")
        return self


class PredictionModelFitEvidenceEnvelope(BaseModel):
    """Persisted V4 fit authority carried by the lineage model role."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["easyicu.prediction_model_fit_evidence/1"] = (
        "easyicu.prediction_model_fit_evidence/1"
    )
    issuer: Literal["easyicu.prediction_model_fit_evidence"]
    fit_spec: PredictionModelFitSpec
    fit_receipt: PredictionModelFitReceipt
    source_input_receipt: TypedInputConsumptionReceipt
    model_artifact: PredictionModelArtifact
    source_projection_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_projection_size_bytes: int = Field(ge=1)
    split_assignment_artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    split_assignment_artifact_size_bytes: int = Field(ge=1)
    prediction_table_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    prediction_table_size_bytes: int = Field(ge=1)
    authority_scope: Literal["analysis_only"]
    paper_authorization_allowed: Literal[False]
    planner_selection_authorized: Literal[False]

    @model_validator(mode="after")
    def _fit_authority_reconciles(self) -> "PredictionModelFitEvidenceEnvelope":
        receipt = self.fit_receipt
        source = self.source_input_receipt
        model = self.model_artifact
        if not isinstance(source.row_identity, TypedInputRowIdentity):
            raise ValueError("fit evidence requires a typed row identity")
        model_bytes = prediction_model_artifact_bytes(model)
        expected = {
            "contract_sha256": prediction_model_fit_spec_sha256(self.fit_spec),
            "source_input_receipt_sha256": source.receipt_sha256,
            "source_artifact_sha256": source.artifact_sha256,
            "source_loaded_frame_sha256": source.loaded_frame_sha256,
            "source_row_identity_sha256": source.row_identity.sha256,
            "model_artifact_sha256": sha256_bytes(model_bytes),
            "model_artifact_size_bytes": len(model_bytes),
            "prediction_table_sha256": self.prediction_table_sha256,
            "prediction_table_size_bytes": self.prediction_table_size_bytes,
        }
        observed = {field: getattr(receipt, field) for field in expected}
        if observed != expected:
            raise ValueError("fit receipt does not match its persisted evidence")
        model_bindings = {
            "contract_sha256": model.contract_sha256,
            "source_input_receipt_sha256": model.source_input_receipt_sha256,
            "source_artifact_sha256": model.source_artifact_sha256,
            "training_subjects_sha256": model.training_subjects_sha256,
            "evaluation_subjects_sha256": model.evaluation_subjects_sha256,
            "split_assignment_sha256": model.split_assignment_sha256,
        }
        receipt_bindings = {field: getattr(receipt, field) for field in model_bindings}
        if model_bindings != receipt_bindings:
            raise ValueError("model artifact does not match the fit receipt")
        if model.model_identifier != self.fit_spec.model_identifier:
            raise ValueError("model identifier does not match the fit declaration")
        return self


class PredictionModelFitValidationRegistration(BaseModel):
    """Composite receipt for the V4-to-V3.1 analysis-only route."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[
        "easyicu.prediction_model_fit_validation_registration/1"
    ] = "easyicu.prediction_model_fit_validation_registration/1"
    fit_receipt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    model_evidence_envelope_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    lineage: PredictionValidationUpstreamLineage
    analysis_registration: PredictionValidationAnalysisRegistration
    claim_ceiling: Literal["analysis_only"]
    paper_authorization: Literal[False]
    planner_selection_authorized: Literal[False]

    @model_validator(mode="after")
    def _analysis_uses_exact_lineage(
        self,
    ) -> "PredictionModelFitValidationRegistration":
        upstream_ids = tuple(binding.evidence_id for binding in self.lineage.artifacts)
        if self.analysis_registration.upstream_evidence_ids != upstream_ids:
            raise ValueError("analysis registration does not use the exact lineage")
        if self.analysis_registration.producer_run_id != self.lineage.producer_run_id:
            raise ValueError("analysis and lineage run identities differ")
        if (
            self.lineage.binding_for("model_artifact").sha256
            != self.model_evidence_envelope_sha256
        ):
            raise ValueError("model envelope digest does not match the lineage")
        return self


def _parse_model(
    value: Any,
    model_type: type[BaseModel],
    *,
    name: str,
) -> BaseModel:
    payload = (
        value.model_dump(mode="python") if isinstance(value, model_type) else value
    )
    try:
        return model_type.model_validate(payload)
    except ValidationError as error:
        raise PredictionModelFitEvidenceError(
            PredictionModelFitEvidenceReason.INPUT_SCHEMA_INVALID,
            f"{name} is not schema-valid",
            error_count=error.error_count(),
        ) from error


def _canonical_step_id(value: str, *, name: str) -> str:
    parsed = str(value or "")
    if not parsed or parsed != parsed.strip():
        raise PredictionModelFitEvidenceError(
            PredictionModelFitEvidenceReason.INPUT_SCHEMA_INVALID,
            f"{name} must be non-empty and whitespace-canonical",
        )
    return parsed


def _validate_contract_join(
    fit_spec: PredictionModelFitSpec,
    validation_spec: PredictionValidationSpec,
) -> None:
    fields = (
        "unit_id_column",
        "subject_id_column",
        "split_column",
        "outcome_column",
        "probability_column",
    )
    mismatched = [
        field
        for field in fields
        if getattr(fit_spec, field) != getattr(validation_spec, field)
    ]
    if validation_spec.evaluation_split != fit_spec.evaluation_split:
        mismatched.append("evaluation_split")
    if validation_spec.analysis_unit != fit_spec.analysis_unit:
        mismatched.append("analysis_unit")
    if mismatched:
        raise PredictionModelFitEvidenceError(
            PredictionModelFitEvidenceReason.VALIDATION_CONTRACT_MISMATCH,
            "prediction validation coordinates differ from the sealed fit",
            mismatched_fields=mismatched,
        )


def _source_projection_bytes(
    source_input: LoadedTypedInput,
    spec: PredictionModelFitSpec,
) -> bytes:
    frame = source_input.to_pandas()
    columns = (
        spec.unit_id_column,
        spec.subject_id_column,
        spec.split_column,
        spec.outcome_column,
        *spec.feature_columns,
    )
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(columns)
    try:
        for row in frame.loc[:, list(columns)].itertuples(index=False, name=None):
            coordinates = (
                str(row[0]),
                str(row[1]),
                str(row[2]),
                str(int(row[3])),
            )
            features = tuple(
                "" if pd.isna(value) else format(float(value), ".17g")
                for value in row[4:]
            )
            writer.writerow((*coordinates, *features))
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise PredictionModelFitEvidenceError(
            PredictionModelFitEvidenceReason.MATERIALIZATION_INVALID,
            "typed model source cannot be serialized canonically",
        ) from error
    return buffer.getvalue().encode("utf-8")


def _split_assignment_bytes(
    source_input: LoadedTypedInput,
    spec: PredictionModelFitSpec,
) -> bytes:
    frame = source_input.to_pandas()
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow((spec.subject_id_column, spec.split_column))
    try:
        for subject_id, split in frame.loc[
            :, [spec.subject_id_column, spec.split_column]
        ].itertuples(index=False, name=None):
            writer.writerow((str(subject_id), str(split)))
    except (KeyError, TypeError, ValueError) as error:
        raise PredictionModelFitEvidenceError(
            PredictionModelFitEvidenceReason.MATERIALIZATION_INVALID,
            "subject split assignment cannot be serialized canonically",
        ) from error
    return buffer.getvalue().encode("utf-8")


def _binding(
    role: str,
    record: EvidenceRecord,
) -> PredictionValidationArtifactBinding:
    return PredictionValidationArtifactBinding(
        role=role,
        evidence_id=record.evidence_id,
        sha256=record.sha256,
        kind=record.kind,
        produced_by_step=record.produced_by_step,
    )


def _register_text(
    *,
    evidence_store: EvidenceStore,
    kind: str,
    description: str,
    payload: bytes,
    filename: str,
    evidence_id: str,
    produced_by_step: str,
    inputs: tuple[str, ...],
    metadata: dict[str, Any],
) -> EvidenceRecord:
    try:
        text = payload.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:  # pragma: no cover - owner emits UTF-8
        raise PredictionModelFitEvidenceError(
            PredictionModelFitEvidenceReason.MATERIALIZATION_INVALID,
            "fit evidence artifact is not UTF-8",
            filename=filename,
        ) from error
    return evidence_store.register_text(
        kind=kind,
        description=description,
        text=text,
        filename=filename,
        evidence_id=evidence_id,
        produced_by_step=produced_by_step,
        inputs=inputs,
        producer="prediction_model_fit",
        generation_mode="deterministic_skill",
        metadata=metadata,
        publish_aliases=False,
    )


def register_prediction_model_fit_validation_artifact(
    *,
    evidence_store: EvidenceStore,
    source_input: LoadedTypedInput,
    fit_spec: PredictionModelFitSpec | Mapping[str, Any],
    fit_bundle: PredictionModelFitBundle,
    validation_spec: PredictionValidationSpec | Mapping[str, Any],
    runtime_authority: PredictionModelFitRuntimeAuthority | Mapping[str, Any],
    fit_step_id: str,
    validation_step_id: str,
) -> PredictionModelFitValidationRegistration:
    """Materialize a sealed fit and delegate metric authority to V3.1."""

    parsed_fit_spec = cast(
        PredictionModelFitSpec,
        _parse_model(fit_spec, PredictionModelFitSpec, name="fit_spec"),
    )
    parsed_validation_spec = cast(
        PredictionValidationSpec,
        _parse_model(
            validation_spec,
            PredictionValidationSpec,
            name="validation_spec",
        ),
    )
    parsed_runtime = cast(
        PredictionModelFitRuntimeAuthority,
        _parse_model(
            runtime_authority,
            PredictionModelFitRuntimeAuthority,
            name="runtime_authority",
        ),
    )
    fit_step = _canonical_step_id(fit_step_id, name="fit_step_id")
    validation_step = _canonical_step_id(
        validation_step_id,
        name="validation_step_id",
    )
    _validate_contract_join(parsed_fit_spec, parsed_validation_spec)
    revalidate_prediction_model_fit_bundle(
        bundle=fit_bundle,
        source_input=source_input,
        spec=parsed_fit_spec,
    )
    resolve_prediction_validation_runtime_authority(
        evidence_store=evidence_store,
        producer_run_id=parsed_runtime.producer_run_id,
        runtime=parsed_runtime.runtime,
        artifacts=parsed_runtime.artifacts,
    )

    source_bytes = _source_projection_bytes(source_input, parsed_fit_spec)
    split_bytes = _split_assignment_bytes(source_input, parsed_fit_spec)
    prediction_bytes = fit_bundle.prediction_csv_bytes
    fit_sha256 = fit_bundle.receipt.receipt_sha256
    materialization_key = canonical_sha256(
        {
            "producer_run_id": parsed_runtime.producer_run_id,
            "fit_receipt_sha256": fit_sha256,
            "fit_step_id": fit_step,
        }
    )[:12]
    evidence_ids = {
        "cohort": f"prediction_fit_source_{materialization_key}",
        "split_assignment": f"prediction_fit_split_{materialization_key}",
        "model_artifact": f"prediction_fit_model_{materialization_key}",
        "prediction_table": f"prediction_fit_predictions_{materialization_key}",
    }
    envelope = PredictionModelFitEvidenceEnvelope(
        issuer="easyicu.prediction_model_fit_evidence",
        fit_spec=parsed_fit_spec,
        fit_receipt=fit_bundle.receipt,
        source_input_receipt=source_input.receipt,
        model_artifact=fit_bundle.model_artifact,
        source_projection_sha256=sha256_bytes(source_bytes),
        source_projection_size_bytes=len(source_bytes),
        split_assignment_artifact_sha256=sha256_bytes(split_bytes),
        split_assignment_artifact_size_bytes=len(split_bytes),
        prediction_table_sha256=sha256_bytes(prediction_bytes),
        prediction_table_size_bytes=len(prediction_bytes),
        authority_scope="analysis_only",
        paper_authorization_allowed=False,
        planner_selection_authorized=False,
    )
    envelope_bytes = canonical_json(
        envelope.model_dump(mode="json"),
        trailing_newline=True,
    ).encode("utf-8")
    runtime_ids = tuple(binding.evidence_id for binding in parsed_runtime.artifacts)
    common_metadata = {
        "schema_version": "easyicu.prediction_model_fit_evidence/1",
        "capability_id": "prediction_model_fit",
        "maturity": "experimental",
        "claim_ceiling": "analysis_only",
        "paper_authorization": False,
        "planner_selection_authorized": False,
        "fit_receipt_sha256": fit_sha256,
        "run_id": parsed_runtime.producer_run_id,
    }
    aliases_before = dict(evidence_store.aliases())
    numeric_before = len(evidence_store.numeric_claims())
    scientific_before = len(evidence_store.scientific_claims())

    source_record = _register_text(
        evidence_store=evidence_store,
        kind="table",
        description=(
            "Experimental typed model-source projection; analysis-only and "
            "not manuscript-authoritative."
        ),
        payload=source_bytes,
        filename=f"{evidence_ids['cohort']}.csv",
        evidence_id=evidence_ids["cohort"],
        produced_by_step=fit_step,
        inputs=runtime_ids,
        metadata={**common_metadata, "artifact_role": "cohort"},
    )
    split_record = _register_text(
        evidence_store=evidence_store,
        kind="table",
        description=(
            "Experimental subject-disjoint split projection; analysis-only and "
            "not manuscript-authoritative."
        ),
        payload=split_bytes,
        filename=f"{evidence_ids['split_assignment']}.csv",
        evidence_id=evidence_ids["split_assignment"],
        produced_by_step=fit_step,
        inputs=(source_record.evidence_id,),
        metadata={**common_metadata, "artifact_role": "split_assignment"},
    )
    model_record = _register_text(
        evidence_store=evidence_store,
        kind="code",
        description=(
            "Experimental sealed prediction-model fit evidence; analysis-only "
            "and not manuscript-authoritative."
        ),
        payload=envelope_bytes,
        filename=f"{evidence_ids['model_artifact']}.json",
        evidence_id=evidence_ids["model_artifact"],
        produced_by_step=fit_step,
        inputs=(source_record.evidence_id, split_record.evidence_id, *runtime_ids),
        metadata={**common_metadata, "artifact_role": "model_artifact"},
    )
    prediction_record = _register_text(
        evidence_store=evidence_store,
        kind="table",
        description=(
            "Experimental sealed binary prediction table; analysis-only and "
            "not manuscript-authoritative."
        ),
        payload=prediction_bytes,
        filename=f"{evidence_ids['prediction_table']}.csv",
        evidence_id=evidence_ids["prediction_table"],
        produced_by_step=fit_step,
        inputs=(
            source_record.evidence_id,
            split_record.evidence_id,
            model_record.evidence_id,
        ),
        metadata={**common_metadata, "artifact_role": "prediction_table"},
    )
    fit_bindings = (
        _binding("prediction_table", prediction_record),
        _binding("cohort", source_record),
        _binding("split_assignment", split_record),
        _binding("model_artifact", model_record),
    )
    lineage = PredictionValidationUpstreamLineage(
        producer_run_id=parsed_runtime.producer_run_id,
        model_identifier=parsed_fit_spec.model_identifier,
        evaluation_split=parsed_fit_spec.evaluation_split,
        split_policy="subject_disjoint",
        runtime=parsed_runtime.runtime,
        artifacts=(*fit_bindings, *parsed_runtime.artifacts),
    )
    analysis_registration = register_prediction_validation_analysis_artifact(
        evidence_store=evidence_store,
        spec=parsed_validation_spec,
        lineage=lineage,
        validation_step_id=validation_step,
    )
    if (
        evidence_store.aliases() != aliases_before
        or len(evidence_store.numeric_claims()) != numeric_before
        or len(evidence_store.scientific_claims()) != scientific_before
    ):
        raise PredictionModelFitEvidenceError(
            PredictionModelFitEvidenceReason.AUTHORITY_CEILING_VIOLATION,
            "fit evidence registration changed an alias or claim registry",
        )
    return PredictionModelFitValidationRegistration(
        fit_receipt_sha256=fit_sha256,
        model_evidence_envelope_sha256=model_record.sha256,
        lineage=lineage,
        analysis_registration=analysis_registration,
        claim_ceiling="analysis_only",
        paper_authorization=False,
        planner_selection_authorized=False,
    )


__all__ = [
    "PredictionModelFitEvidenceEnvelope",
    "PredictionModelFitEvidenceError",
    "PredictionModelFitEvidenceReason",
    "PredictionModelFitRuntimeAuthority",
    "PredictionModelFitValidationRegistration",
    "register_prediction_model_fit_validation_artifact",
]

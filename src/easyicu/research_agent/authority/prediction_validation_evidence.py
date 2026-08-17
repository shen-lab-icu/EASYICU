"""Analysis-only EvidenceStore bridge for prediction validation.

This authority adapter recomputes from current upstream EvidenceStore bytes;
callers cannot submit a receipt or validation seal.  It deliberately does not
publish aliases or manuscript-bindable claims and is not imported by the
Planner or execution selection path.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from ..contracts.prediction_validation import (
    PredictionValidationAnalysisBundle,
    PredictionValidationAnalysisRegistration,
    PredictionValidationArtifactBinding,
    PredictionValidationArtifactRole,
    PredictionValidationError,
    PredictionValidationFinding,
    PredictionValidationReason,
    PredictionValidationRuntimeIdentity,
    PredictionValidationSpec,
    PredictionValidationUpstreamLineage,
    prediction_validation_analysis_bundle_sha256,
    prediction_validation_upstream_lineage_sha256,
)
from ..prediction_validation_owner import (
    recompute_prediction_validation_analysis,
)
from .evidence_store import EvidenceStore
from .runtime_artifacts import verified_run_evidence_path


def _raise(
    reason_code: PredictionValidationReason,
    message: str,
    **detail: Any,
) -> None:
    raise PredictionValidationError(reason_code, message, **detail)


def _finding(
    reason_code: PredictionValidationReason,
    message: str,
    **detail: Any,
) -> PredictionValidationFinding:
    return PredictionValidationFinding(
        reason_code=reason_code,
        message=message,
        detail=detail,
    )


def _model_input(value: Any, model_type: type[Any]) -> Any:
    return value.model_dump(mode="python") if isinstance(value, model_type) else value


def _parse_registration_inputs(
    *,
    spec: PredictionValidationSpec | Mapping[str, Any],
    lineage: PredictionValidationUpstreamLineage | Mapping[str, Any],
) -> tuple[PredictionValidationSpec, PredictionValidationUpstreamLineage]:
    try:
        parsed_spec = PredictionValidationSpec.model_validate(
            _model_input(spec, PredictionValidationSpec)
        )
    except ValidationError as error:
        raise PredictionValidationError(
            PredictionValidationReason.LINEAGE_SCHEMA_INVALID,
            "prediction-validation specification is not schema-valid",
            error_count=error.error_count(),
        ) from error
    try:
        parsed_lineage = PredictionValidationUpstreamLineage.model_validate(
            _model_input(lineage, PredictionValidationUpstreamLineage)
        )
    except ValidationError as error:
        raise PredictionValidationError(
            PredictionValidationReason.LINEAGE_SCHEMA_INVALID,
            "prediction-validation upstream lineage is not schema-valid",
            error_count=error.error_count(),
        ) from error
    return parsed_spec, parsed_lineage


def resolve_prediction_validation_artifact_bindings(
    *,
    evidence_store: EvidenceStore,
    producer_run_id: str,
    artifacts: tuple[PredictionValidationArtifactBinding, ...],
) -> dict[PredictionValidationArtifactRole, Path]:
    resolved: dict[PredictionValidationArtifactRole, Path] = {}
    for binding in artifacts:
        record = evidence_store.get(binding.evidence_id)
        if record is None or record.evidence_id != binding.evidence_id:
            _raise(
                PredictionValidationReason.LINEAGE_EVIDENCE_MISSING,
                "one required upstream evidence record is missing",
                role=binding.role,
                evidence_id=binding.evidence_id,
            )
        observed = {
            "sha256": record.sha256,
            "kind": record.kind,
            "produced_by_step": record.produced_by_step,
            "run_id": str((record.metadata or {}).get("run_id") or ""),
        }
        expected = {
            "sha256": binding.sha256,
            "kind": binding.kind,
            "produced_by_step": binding.produced_by_step,
            "run_id": producer_run_id,
        }
        if observed != expected:
            _raise(
                PredictionValidationReason.LINEAGE_EVIDENCE_MISMATCH,
                "one upstream evidence record does not match its lineage binding",
                role=binding.role,
                evidence_id=binding.evidence_id,
                expected=expected,
                observed=observed,
            )
        path = verified_run_evidence_path(evidence_store.root, record)
        if path is None:
            _raise(
                PredictionValidationReason.LINEAGE_EVIDENCE_STALE,
                "one upstream evidence artifact is missing or digest-stale",
                role=binding.role,
                evidence_id=binding.evidence_id,
            )
        resolved[binding.role] = path
    return resolved


def _verify_runtime_receipt(
    *,
    resolved: Mapping[PredictionValidationArtifactRole, Path],
    runtime: PredictionValidationRuntimeIdentity,
    runtime_binding: PredictionValidationArtifactBinding,
) -> None:
    runtime_path = resolved["runtime_receipt"]
    try:
        runtime_payload = runtime_path.read_text(encoding="utf-8")
        observed_runtime = PredictionValidationRuntimeIdentity.model_validate_json(
            runtime_payload
        )
    except (OSError, UnicodeDecodeError, ValidationError) as error:
        raise PredictionValidationError(
            PredictionValidationReason.LINEAGE_RUNTIME_MISMATCH,
            "runtime receipt is not a valid runtime identity",
            evidence_id=runtime_binding.evidence_id,
        ) from error
    if observed_runtime != runtime:
        _raise(
            PredictionValidationReason.LINEAGE_RUNTIME_MISMATCH,
            "runtime receipt does not match the declared runtime identity",
            expected=runtime.model_dump(mode="json"),
            observed=observed_runtime.model_dump(mode="json"),
        )


def resolve_prediction_validation_runtime_authority(
    *,
    evidence_store: EvidenceStore,
    producer_run_id: str,
    runtime: PredictionValidationRuntimeIdentity,
    artifacts: tuple[PredictionValidationArtifactBinding, ...],
) -> dict[PredictionValidationArtifactRole, Path]:
    """Resolve the exact code/environment/runtime subset of one lineage."""

    expected_roles = ("code_snapshot", "environment_lock", "runtime_receipt")
    if tuple(binding.role for binding in artifacts) != expected_roles:
        _raise(
            PredictionValidationReason.LINEAGE_SCHEMA_INVALID,
            "runtime authority requires code, environment and receipt in order",
            expected_roles=list(expected_roles),
            observed_roles=[binding.role for binding in artifacts],
        )
    if artifacts[0].sha256 != runtime.source_tree_sha256:
        _raise(
            PredictionValidationReason.LINEAGE_RUNTIME_MISMATCH,
            "runtime source tree does not match the code snapshot",
        )
    if artifacts[1].sha256 != runtime.environment_sha256:
        _raise(
            PredictionValidationReason.LINEAGE_RUNTIME_MISMATCH,
            "runtime environment does not match the environment lock",
        )
    resolved = resolve_prediction_validation_artifact_bindings(
        evidence_store=evidence_store,
        producer_run_id=producer_run_id,
        artifacts=artifacts,
    )
    _verify_runtime_receipt(
        resolved=resolved,
        runtime=runtime,
        runtime_binding=artifacts[2],
    )
    return resolved


def _resolve_lineage_artifacts(
    *,
    evidence_store: EvidenceStore,
    lineage: PredictionValidationUpstreamLineage,
) -> dict[PredictionValidationArtifactRole, Path]:
    resolved = resolve_prediction_validation_artifact_bindings(
        evidence_store=evidence_store,
        producer_run_id=lineage.producer_run_id,
        artifacts=lineage.artifacts,
    )

    _verify_runtime_receipt(
        resolved=resolved,
        runtime=lineage.runtime,
        runtime_binding=lineage.binding_for("runtime_receipt"),
    )
    return resolved


def _authority_ceiling_finding(
    *,
    evidence_store: EvidenceStore,
    evidence_id: str,
) -> PredictionValidationFinding | None:
    published_aliases = sorted(
        alias
        for alias, target in evidence_store.aliases().items()
        if target == evidence_id
    )
    numeric_claim_count = sum(
        claim.evidence_id == evidence_id for claim in evidence_store.numeric_claims()
    )
    scientific_claim_count = sum(
        claim.evidence_id == evidence_id for claim in evidence_store.scientific_claims()
    )
    if not published_aliases and not numeric_claim_count and not scientific_claim_count:
        return None
    return _finding(
        PredictionValidationReason.AUTHORITY_CEILING_VIOLATION,
        "analysis-only prediction validation acquired forbidden authority",
        evidence_id=evidence_id,
        published_aliases=published_aliases,
        numeric_claim_count=numeric_claim_count,
        scientific_claim_count=scientific_claim_count,
    )


def register_prediction_validation_analysis_artifact(
    *,
    evidence_store: EvidenceStore,
    spec: PredictionValidationSpec | Mapping[str, Any],
    lineage: PredictionValidationUpstreamLineage | Mapping[str, Any],
    validation_step_id: str,
) -> PredictionValidationAnalysisRegistration:
    """Recompute and register one prediction result without claim authority."""

    step_id = str(validation_step_id or "").strip()
    if not step_id or step_id != validation_step_id:
        _raise(
            PredictionValidationReason.LINEAGE_SCHEMA_INVALID,
            "validation_step_id must be non-empty and whitespace-canonical",
        )
    parsed_spec, parsed_lineage = _parse_registration_inputs(
        spec=spec,
        lineage=lineage,
    )
    resolved = _resolve_lineage_artifacts(
        evidence_store=evidence_store,
        lineage=parsed_lineage,
    )
    parsed_receipt, parsed_seal = recompute_prediction_validation_analysis(
        prediction_path=resolved["prediction_table"],
        prediction_sha256=parsed_lineage.binding_for("prediction_table").sha256,
        cohort_path=resolved["cohort"],
        cohort_sha256=parsed_lineage.binding_for("cohort").sha256,
        split_path=resolved["split_assignment"],
        split_sha256=parsed_lineage.binding_for("split_assignment").sha256,
        spec=parsed_spec,
        runtime_identity=parsed_lineage.runtime,
    )
    try:
        bundle = PredictionValidationAnalysisBundle(
            spec=parsed_spec,
            receipt=parsed_receipt,
            validation_seal=parsed_seal,
            lineage=parsed_lineage,
        )
    except ValidationError as error:
        raise PredictionValidationError(
            PredictionValidationReason.LINEAGE_EVIDENCE_MISMATCH,
            "prediction-validation bundle bindings do not reconcile",
            error_count=error.error_count(),
        ) from error

    bundle_sha256 = prediction_validation_analysis_bundle_sha256(bundle)
    lineage_sha256 = prediction_validation_upstream_lineage_sha256(parsed_lineage)
    evidence_id = f"prediction_validation_analysis_{bundle_sha256[:12]}"
    upstream_ids = tuple(binding.evidence_id for binding in parsed_lineage.artifacts)
    numeric_before = len(evidence_store.numeric_claims())
    scientific_before = len(evidence_store.scientific_claims())
    record = evidence_store.register_json(
        kind="statistic",
        description=(
            "Experimental host-recomputed prediction validation bundle; "
            "analysis-only and not manuscript-authoritative."
        ),
        payload=bundle.model_dump(mode="json"),
        filename=f"{evidence_id}.json",
        produced_by_step=step_id,
        inputs=upstream_ids,
        evidence_id=evidence_id,
        producer="prediction_validation",
        generation_mode="deterministic_skill",
        metadata={
            "schema_version": bundle.schema_version,
            "capability_id": "prediction_validation",
            "maturity": "experimental",
            "claim_ceiling": bundle.policy.claim_ceiling,
            "paper_authorization": bundle.policy.paper_authorization,
            "planner_selection_authorized": (
                bundle.policy.planner_selection_authorized
            ),
            "numeric_claim_registration_authorized": (
                bundle.policy.numeric_claim_registration_authorized
            ),
            "scientific_claim_registration_authorized": (
                bundle.policy.scientific_claim_registration_authorized
            ),
            "alias_publication_authorized": (
                bundle.policy.alias_publication_authorized
            ),
            "bundle_sha256": bundle_sha256,
            "lineage_sha256": lineage_sha256,
            "receipt_sha256": parsed_seal.receipt_sha256,
            "runtime_identity_sha256": parsed_seal.runtime_identity_sha256,
            "contract_sha256": parsed_receipt.contract_sha256,
            "result_sha256": parsed_receipt.result_sha256,
            "run_id": parsed_lineage.producer_run_id,
        },
        publish_aliases=False,
    )
    numeric_delta = len(evidence_store.numeric_claims()) - numeric_before
    scientific_delta = len(evidence_store.scientific_claims()) - scientific_before
    if numeric_delta or scientific_delta:
        _raise(
            PredictionValidationReason.AUTHORITY_CEILING_VIOLATION,
            "analysis-only registration changed a claim registry",
            numeric_claim_count_delta=numeric_delta,
            scientific_claim_count_delta=scientific_delta,
        )
    registration = PredictionValidationAnalysisRegistration(
        evidence_id=record.evidence_id,
        evidence_sha256=record.sha256,
        bundle_sha256=bundle_sha256,
        lineage_sha256=lineage_sha256,
        producer_run_id=parsed_lineage.producer_run_id,
        validation_step_id=step_id,
        upstream_evidence_ids=upstream_ids,
    )
    findings = prediction_validation_analysis_registration_findings(
        evidence_store=evidence_store,
        registration=registration,
    )
    if findings:
        finding = findings[0]
        _raise(
            finding.reason_code,
            "registered analysis artifact failed authority validation",
            finding=finding.model_dump(mode="json"),
        )
    return registration


def prediction_validation_analysis_registration_findings(
    *,
    evidence_store: EvidenceStore,
    registration: PredictionValidationAnalysisRegistration | Mapping[str, Any],
) -> tuple[PredictionValidationFinding, ...]:
    """Revalidate one registered bundle and its analysis-only authority ceiling."""

    try:
        parsed = PredictionValidationAnalysisRegistration.model_validate(
            _model_input(registration, PredictionValidationAnalysisRegistration)
        )
    except ValidationError as error:
        return (
            _finding(
                PredictionValidationReason.LINEAGE_SCHEMA_INVALID,
                "analysis registration receipt is not schema-valid",
                error_count=error.error_count(),
            ),
        )

    ceiling_finding = _authority_ceiling_finding(
        evidence_store=evidence_store,
        evidence_id=parsed.evidence_id,
    )
    if ceiling_finding is not None:
        return (ceiling_finding,)

    record = evidence_store.get(parsed.evidence_id)
    if record is None or record.evidence_id != parsed.evidence_id:
        return (
            _finding(
                PredictionValidationReason.LINEAGE_EVIDENCE_MISSING,
                "registered analysis evidence record is missing",
                evidence_id=parsed.evidence_id,
            ),
        )
    expected_metadata = {
        "schema_version": "easyicu.prediction_validation_analysis_bundle/1",
        "capability_id": "prediction_validation",
        "maturity": "experimental",
        "claim_ceiling": "analysis_only",
        "paper_authorization": False,
        "planner_selection_authorized": False,
        "numeric_claim_registration_authorized": False,
        "scientific_claim_registration_authorized": False,
        "alias_publication_authorized": False,
        "bundle_sha256": parsed.bundle_sha256,
        "lineage_sha256": parsed.lineage_sha256,
        "run_id": parsed.producer_run_id,
        "aliases_published": False,
    }
    observed_metadata = {
        key: (record.metadata or {}).get(key) for key in expected_metadata
    }
    if (
        record.sha256 != parsed.evidence_sha256
        or record.kind != "statistic"
        or record.produced_by_step != parsed.validation_step_id
        or record.producer != "prediction_validation"
        or record.generation_mode != "deterministic_skill"
        or tuple(record.inputs) != parsed.upstream_evidence_ids
        or observed_metadata != expected_metadata
    ):
        return (
            _finding(
                PredictionValidationReason.LINEAGE_EVIDENCE_MISMATCH,
                "registered analysis evidence metadata does not match its receipt",
                evidence_id=parsed.evidence_id,
            ),
        )
    bundle_path = verified_run_evidence_path(evidence_store.root, record)
    if bundle_path is None:
        return (
            _finding(
                PredictionValidationReason.LINEAGE_EVIDENCE_STALE,
                "registered analysis bundle is missing or digest-stale",
                evidence_id=parsed.evidence_id,
            ),
        )
    try:
        bundle = PredictionValidationAnalysisBundle.model_validate_json(
            bundle_path.read_text(encoding="utf-8")
        )
    except (OSError, UnicodeDecodeError, ValidationError) as error:
        return (
            _finding(
                PredictionValidationReason.LINEAGE_EVIDENCE_MISMATCH,
                "registered analysis bundle is not schema-valid",
                evidence_id=parsed.evidence_id,
                error_type=type(error).__name__,
            ),
        )
    if (
        prediction_validation_analysis_bundle_sha256(bundle) != parsed.bundle_sha256
        or prediction_validation_upstream_lineage_sha256(bundle.lineage)
        != parsed.lineage_sha256
        or bundle.lineage.producer_run_id != parsed.producer_run_id
    ):
        return (
            _finding(
                PredictionValidationReason.LINEAGE_EVIDENCE_MISMATCH,
                "registered analysis bundle digest coordinates do not match",
                evidence_id=parsed.evidence_id,
            ),
        )
    expected_bundle_metadata = {
        "receipt_sha256": bundle.validation_seal.receipt_sha256,
        "runtime_identity_sha256": (bundle.validation_seal.runtime_identity_sha256),
        "contract_sha256": bundle.receipt.contract_sha256,
        "result_sha256": bundle.receipt.result_sha256,
    }
    observed_bundle_metadata = {
        key: (record.metadata or {}).get(key) for key in expected_bundle_metadata
    }
    if observed_bundle_metadata != expected_bundle_metadata:
        return (
            _finding(
                PredictionValidationReason.LINEAGE_EVIDENCE_MISMATCH,
                "registered analysis metadata drifted from its sealed bundle",
                evidence_id=parsed.evidence_id,
                expected=expected_bundle_metadata,
                observed=observed_bundle_metadata,
            ),
        )
    try:
        resolved = _resolve_lineage_artifacts(
            evidence_store=evidence_store,
            lineage=bundle.lineage,
        )
        expected_receipt, expected_seal = recompute_prediction_validation_analysis(
            prediction_path=resolved["prediction_table"],
            prediction_sha256=bundle.lineage.binding_for("prediction_table").sha256,
            cohort_path=resolved["cohort"],
            cohort_sha256=bundle.lineage.binding_for("cohort").sha256,
            split_path=resolved["split_assignment"],
            split_sha256=bundle.lineage.binding_for("split_assignment").sha256,
            spec=bundle.spec,
            runtime_identity=bundle.lineage.runtime,
        )
    except PredictionValidationError as error:
        return (
            _finding(
                error.reason_code,
                "registered analysis bundle has invalid upstream authority",
                **error.detail,
            ),
        )
    if bundle.receipt != expected_receipt:
        return (
            _finding(
                PredictionValidationReason.RECEIPT_MISMATCH,
                "registered analysis receipt differs from host recomputation",
                evidence_id=parsed.evidence_id,
            ),
        )
    if bundle.validation_seal != expected_seal:
        return (
            _finding(
                PredictionValidationReason.VALIDATION_SEAL_INVALID,
                "registered analysis seal differs from host recomputation",
                evidence_id=parsed.evidence_id,
            ),
        )
    return ()


__all__ = [
    "prediction_validation_analysis_registration_findings",
    "register_prediction_validation_analysis_artifact",
    "resolve_prediction_validation_artifact_bindings",
    "resolve_prediction_validation_runtime_authority",
]

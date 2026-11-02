"""Host-reviewed, source-bound preparation of the conservative eICU subset.

This adapter is not registered as a user entry point. It grants preparation
only; research execution still requires the complete plan's existing grant.
The attestation protects against untrusted request data, not same-UID code.
"""
from __future__ import annotations

import hashlib
import hmac
import io
import json
import os
import stat
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd

from easyicu.databases.research_plan_requirements import _constraints
from easyicu.research_agent.acquisition.first_icu_stay import FirstObservedIcuCoordinate
from easyicu.research_agent.acquisition.patient_grouping import PatientGroupingError, exact_integer_series
from easyicu.research_agent.acquisition.first_icu_stay import (
    FLAG_COLUMN, REASON_COLUMN, REQUIRED_COLUMNS, EICU_FIRST_OBSERVED_CANDIDATE_SCHEMA as SCHEMA,
)
from easyicu.research_agent.authority.filesystem import AnchoredDirectory
from easyicu.research_agent.authority.evidence_store import NumericClaim, _lookup_evidence_record
from easyicu.research_agent.authority.evidence_snapshot import load_current_evidence_snapshot
from easyicu.research_agent.schema import EvidenceRecord
from easyicu.research_agent.authority.plan_review import PlanReviewAuthority
from easyicu.research_agent.authority.raw_preparation import RawPreparationCheckpoint, verify_raw_preparation_review
from easyicu.research_agent.authority.runtime_artifacts import verified_run_evidence_path
from easyicu.research_agent.canonical_json import canonical_json, canonical_sha256
from easyicu.research_agent.planning.scientific_review import PlanScientificReview
from easyicu.research_agent.research_context.export_selection import SourceIdentity
from easyicu.research_agent.research_context.typed import parse_research_context_json
from easyicu.webserver import sources, state_paths
from easyicu.webserver.demo_source_storage import verified_public_demo_raw
from easyicu.webserver.durable_research_store import runtime_digest
from easyicu.webserver.eicu_first_observed_source import derive_from_verified_eicu_demo
from easyicu.webserver.pi_copilot.population_spec_review import reviewed_population_spec
from easyicu.webserver.raw_preparation_decisions import (
    RawPreparationHostBinding, _publish_once, _signature, consume_host_review, verify_consumed_host_review,
)
from easyicu.webserver.raw_source_authority import resolve_raw_source_identity_binding

_SCOPE_KEY = "first_observed_icu_preparation_scope"
_SUMMARY = (
    "Prepare the source-visible adult first ICU subset from the complete eICU Demo patient table. "
    "Read patient, hospitalization, unit order, unit type and age fields; exclude people with "
    "multiple visible hospitalizations or unverified identity, order or age. Completeness verification "
    "may repeatedly read and hash the complete official archive and patient Parquet bytes, and streams the "
    "decompressed patient CSV; these files may contain other information including outcomes. "
    "Only the six named fields are selected for comparison and cohort calculation. This is not a "
    "lifetime first ICU guarantee and does not authorize outcome analysis or research execution."
)
_ADAPTER = "eicu-demo-first-observed-single-hospital-adult/1"
_PATIENT_EXCLUSION_REASONS = frozenset({
    "hospitalization_identifier_missing_or_invalid",
    "hospitalization_identifier_shared_across_patients",
    "multiple_observed_hospitalizations",
    "first_icu_order_or_type_unverifiable",
    "age_unknown_or_invalid",
    "under_18",
})


def _private_source_flow(candidate: Any) -> dict[str, Any]:
    """Keep only checked source-global counts, never a study-flow claim.

    This private record is not a disclosure grant.  In particular, it does not
    identify the eventual analysis population or permit unsuppressed counts to
    be sent to a model, shown in the UI, or included in a publication.
    """
    receipt, frame = candidate.receipt, candidate.frame
    if (not isinstance(receipt, Mapping) or not isinstance(frame, pd.DataFrame)
            or receipt.get("schema_version") != SCHEMA
            or receipt.get("source_global_table_required") is not True
            or receipt.get("not_lifetime_first_icu") is not True
            or receipt.get("outcome_columns_read") is not False
            or receipt.get("receipt_contains_identifier_values") is not False
            or receipt.get("research_execution_authorized") is not False
            or set(frame.columns) != {"stay_id", FLAG_COLUMN, REASON_COLUMN}):
        raise ValueError("first_observed_source_flow_invalid")
    source_stays, source_patients, eligible = (
        receipt.get(name) for name in ("source_stays", "source_patients", "eligible_patients")
    )
    if any(type(value) is not int or value < 0 for value in (source_stays, source_patients, eligible)):
        raise ValueError("first_observed_source_flow_invalid")
    patients, stays = (receipt.get(name) for name in
                       ("excluded_patients_by_reason", "excluded_stays_by_reason"))
    for counts, allowed in ((patients, _PATIENT_EXCLUSION_REASONS),
                            (stays, _PATIENT_EXCLUSION_REASONS | {"non_selected_unit"})):
        if (not isinstance(counts, Mapping) or not set(counts).issubset(allowed)
                or any(type(value) is not int or value <= 0 for value in counts.values())):
            raise ValueError("first_observed_source_flow_invalid")
    if (not pd.api.types.is_bool_dtype(frame[FLAG_COLUMN]) or frame[FLAG_COLUMN].isna().any()
            or not frame[REASON_COLUMN].isin(_PATIENT_EXCLUSION_REASONS | {"kept", "non_selected_unit"}).all()
            or not frame[FLAG_COLUMN].eq(frame[REASON_COLUMN].eq("kept")).all()):
        raise ValueError("first_observed_source_flow_invalid")
    actual_stays = {str(key): int(value) for key, value in
                    frame.loc[~frame[FLAG_COLUMN], REASON_COLUMN].value_counts().items()}
    try:
        identities = exact_integer_series(frame["stay_id"], label="coordinate stay")
    except PatientGroupingError as exc:
        raise ValueError("first_observed_source_flow_invalid") from exc
    if (source_stays != len(frame) or eligible != int(frame[FLAG_COLUMN].sum())
            or source_patients != eligible + sum(patients.values())
            or identities.isna().any() or identities.le(0).any() or identities.duplicated().any()
            or any(count > actual_stays.get(reason, 0) for reason, count in patients.items())
            or actual_stays != dict(stays)):
        raise ValueError("first_observed_source_flow_invalid")
    return {
        "schema_version": "easyicu.first-observed-private-source-flow/1",
        "scope": "complete_source_before_other_study_filters",
        "not_lifetime_first_icu": True,
        "source_stays": source_stays,
        "source_patients": source_patients,
        "eligible_patients": eligible,
        "excluded_patients_by_reason": dict(sorted(patients.items())),
        "excluded_stays_by_reason": dict(sorted(stays.items())),
        "public_projection_authorized": False,
    }


def _readonly_evidence_record(run_dir: Path, evidence_id: str) -> EvidenceRecord | None:
    """Resolve the selected evidence generation without repairing the run."""
    try:
        snapshot = load_current_evidence_snapshot(run_dir)
        records = [EvidenceRecord.model_validate(record) for record in snapshot.records]
        # The mutable store validates every numeric claim when loading a
        # generation. Preserve that fail-closed contract for read-only views.
        for claim in snapshot.numeric_claims:
            NumericClaim.from_dict(claim)
        return _lookup_evidence_record(records, snapshot.aliases, evidence_id)
    except Exception:
        # EvidenceStore._load_snapshot likewise rejects any schema error.
        return None


def _verified_scientific_review(*, run_dir: Path, authority: PlanReviewAuthority, context: Any) -> str:
    """Revalidate the persisted review, not just the digest supplied in a request."""
    expected = authority.evidence_sha256.get("scientific_plan_review")
    if not expected or not (run_dir / "evidence").is_dir():
        raise ValueError("first_observed_preparation_scientific_review_missing")
    record = _readonly_evidence_record(run_dir, "scientific_plan_review")
    path = verified_run_evidence_path(run_dir, record) if record is not None else None
    if path is None or record.sha256 != expected:
        raise ValueError("first_observed_preparation_scientific_review_missing")
    with AnchoredDirectory.open(path.parent) as directory:
        raw = directory.read_bytes(path.name, max_bytes=5 * 1024 * 1024)
        directory.assert_still_selected()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("first_observed_preparation_scientific_review_mismatch")
    scientific_review = PlanScientificReview.model_validate_json(raw)
    if (scientific_review.plan_sha256 != authority.plan_sha256
            or scientific_review.context_sha256 != canonical_sha256(context.model_dump(mode="json"))):
        raise ValueError("first_observed_preparation_scientific_review_mismatch")
    return expected


def _pending_execution_proposal(run_dir: Path, review: PlanReviewAuthority) -> str:
    """Bind a prepare-only proposal to its still-unapproved execution checkpoint."""
    from easyicu.research_agent.orchestration.human_review_checkpoint import (
        HumanReviewCheckpointError, load_checkpoint,
    )

    try:
        checkpoint = load_checkpoint(run_dir / "human_review_checkpoint.json")
    except HumanReviewCheckpointError as exc:
        raise ValueError("first_observed_preparation_review_not_pending") from exc
    execution = review.execution
    if (execution is None or checkpoint.run_id != run_dir.name
            or checkpoint.consumed_decision_sha256 is not None
            or checkpoint.approved_decisions_sha256 is not None or checkpoint.approved_decisions
            or checkpoint.approved_decision_records or checkpoint.execution_start_receipt is not None
            or execution.pipeline_config_sha256 != checkpoint.pipeline_config_sha256
            or execution.capability_activation_sha256 != checkpoint.capability_activation_sha256
            or execution.run_input_capsule_sha256 != checkpoint.run_input_capsule_sha256):
        raise ValueError("first_observed_preparation_review_not_pending")
    matching = [item for item in checkpoint.requests
                if isinstance(item.payload.get("plan_review_authority"), Mapping)]
    if (not matching or len(matching) != len(checkpoint.requests)
            or any(item.authority_sha256 != canonical_sha256(item.payload)
            or item.payload.get("plan_review_authority") != review.model_dump(mode="json") for item in matching)
            or not any(item.kind == "scientific_stop" and item.payload.get("approval_allowed") is False for item in matching)):
        raise ValueError("first_observed_preparation_review_authority_mismatch")
    with AnchoredDirectory.open(run_dir) as directory:
        raw = directory.read_bytes("run_input_capsule.json", max_bytes=5 * 1024 * 1024)
        directory.assert_still_selected()
    if hashlib.sha256(raw).hexdigest() != execution.run_input_capsule_sha256:
        raise ValueError("first_observed_preparation_review_capsule_mismatch")
    record = _readonly_evidence_record(run_dir, "run_input_capsule")
    path = verified_run_evidence_path(run_dir, record) if record is not None else None
    if path is None or record.sha256 != execution.run_input_capsule_sha256:
        raise ValueError("first_observed_preparation_review_capsule_mismatch")
    with AnchoredDirectory.open(path.parent) as directory:
        evidence = directory.read_bytes(path.name, max_bytes=5 * 1024 * 1024)
        directory.assert_still_selected()
    from easyicu.research_agent.authority.run_input import RunInputCapsule, RunInputCapsuleV2, RunInputCapsuleV3

    payload = json.loads(raw)
    schema = payload.get("schema_version") if isinstance(payload, dict) else None
    models = {"easyicu.run_input_capsule/1": RunInputCapsule, "easyicu.run_input_capsule/2": RunInputCapsuleV2,
              "easyicu.run_input_capsule/3": RunInputCapsuleV3}
    if schema not in models or evidence != raw:
        raise ValueError("first_observed_preparation_review_capsule_mismatch")
    capsule = models[schema].model_validate(payload)
    with AnchoredDirectory.open(run_dir) as directory:
        context_raw = directory.read_bytes("research_context.json", max_bytes=5 * 1024 * 1024)
        directory.assert_still_selected()
    context = json.loads(context_raw)
    identity = capsule.scientific_identity
    context_cohort = context.get("cohort") if isinstance(context, dict) else None
    context_preferences = context.get("user_preferences") if isinstance(context, dict) else None
    identity_preferences = identity.get("user_preferences")
    if (canonical_sha256(identity) != capsule.scientific_identity_sha256
            or capsule.context_evidence_id != "research_context" or capsule.context_relative_path != "research_context.json"
            or capsule.context_sha256 != review.evidence_sha256.get("research_context")
            or hashlib.sha256(context_raw).hexdigest() != capsule.context_sha256
            or not isinstance(context_cohort, dict) or not isinstance(context_preferences, dict)
            or not isinstance(identity_preferences, dict)
            or identity.get("question") != context.get("research_question")
            or review.plan_payload.get("research_question") != context.get("research_question")
            or identity.get("database") != context_cohort.get("database")
            or identity_preferences.get("data_constraints") != context_preferences.get("data_constraints")):
        raise ValueError("first_observed_preparation_review_capsule_mismatch")
    try:
        constraints = json.loads(context_preferences.get("data_constraints") or "{}")
    except (ValueError, TypeError) as exc:
        raise ValueError("first_observed_preparation_review_capsule_mismatch") from exc
    if (not isinstance(constraints, dict) or not isinstance(context_cohort.get("provenance"), dict)
            or not isinstance(constraints.get("source_identity"), dict)
            or constraints["source_identity"].get("database") != context_cohort.get("database")
            or context_cohort["provenance"].get("source_identity") != constraints["source_identity"]):
        raise ValueError("first_observed_preparation_review_capsule_mismatch")
    return checkpoint.checkpoint_sha256


def _assert_pending_request_set(run_dir: Path, pending: Any) -> None:
    from easyicu.research_agent.orchestration.human_review_checkpoint import HumanReviewCheckpointError, load_checkpoint

    try:
        checkpoint = load_checkpoint(run_dir / "human_review_checkpoint.json")
    except HumanReviewCheckpointError as exc:
        raise ValueError("first_observed_preparation_review_not_pending") from exc
    if tuple(checkpoint.requests) != tuple(pending.requests):
        raise ValueError("first_observed_preparation_review_request_set_mismatch")


def _metadata_source(identity: SourceIdentity) -> tuple[Path, dict[str, str]]:
    """Bind the named Demo and metadata only; no patient or archive byte reads."""
    from easyicu.webserver.demo_source_storage import read_marker, verified_public_demo_replanning_export

    registered_path = sources.stored_source_path_for_identity(identity)
    if registered_path is None:
        raise ValueError("first_observed_preparation_source_unavailable")
    export = Path(registered_path)
    historical = verified_public_demo_replanning_export(export)
    if historical is None:
        raise ValueError("first_observed_preparation_source_unavailable")
    source, paths = historical
    if (source.database != "eicu_demo" or identity.source_id != source.id
            or identity.version != source.version or identity.archive_sha256 != source.archive_sha256):
        raise ValueError("first_observed_preparation_source_mismatch")
    with AnchoredDirectory.open(export) as directory:
        manifest_bytes = directory.read_bytes("_manifest.json", max_bytes=16 * 1024 * 1024)
        directory.assert_still_selected()
    manifest = json.loads(manifest_bytes)
    if not isinstance(manifest, dict) or manifest.get("database") != "eicu_demo" or manifest.get("data_path") != str(paths.raw):
        raise ValueError("first_observed_preparation_source_mismatch")
    metadata = {"source_id": source.id, "source_version": source.version,
                "export_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
                "archive_sha256": source.archive_sha256,
                "raw_path_sha256": hashlib.sha256(str(paths.raw).encode()).hexdigest()}
    for name, path in (("conversion", paths.raw / "conversion_manifest.json"),
                       ("prepared", paths.prepared_marker), ("extracted", paths.extracted_marker),
                       ("converted", paths.converted_marker)):
        if name != "conversion" and (read_marker(path, source) or {}).get("archive_sha256") != source.archive_sha256:
            raise ValueError("first_observed_preparation_source_mismatch")
        with AnchoredDirectory.open(path.parent) as directory:
            content = directory.read_bytes(path.name, max_bytes=4 * 1024 * 1024)
            directory.assert_still_selected()
        metadata[name + "_metadata_sha256"] = hashlib.sha256(content).hexdigest()
    return export, metadata


def _host_inputs(*, run_dir: Path, request: Any, verify_source_bytes: bool = True) -> tuple[dict[str, Any], Path]:
    """Construct a plan-bound scope; byte verification requires an existing grant."""
    raw = request.payload.get("plan_review_authority")
    review = PlanReviewAuthority.model_validate(raw)
    checkpoint_sha256 = _pending_execution_proposal(run_dir, review) if review.execution is not None else None
    projection = reviewed_population_spec(run_dir=run_dir, pending=SimpleNamespace(requests=[request]))
    if projection is None or projection["plan_sha256"] != review.plan_sha256:
        raise ValueError("first_observed_preparation_population_unverified")
    matches = [item["criterion"] for item in projection["criteria"]
               if item["criterion"].get("kind") == "first_observed_icu_single_hospital_adult"
               and item["criterion"].get("role") == "include"]
    if len(matches) != 1:
        raise ValueError("first_observed_preparation_scope_ambiguous")
    with AnchoredDirectory.open(run_dir) as directory:
        raw_context = directory.read_bytes("research_context.json", max_bytes=5 * 1024 * 1024)
        directory.assert_still_selected()
    context_sha = hashlib.sha256(raw_context).hexdigest()
    if review.evidence_sha256.get("research_context") != context_sha:
        raise ValueError("first_observed_preparation_context_mismatch")
    context = parse_research_context_json(raw_context.decode("utf-8"))
    scientific_sha = _verified_scientific_review(run_dir=run_dir, authority=review, context=context)
    identity = SourceIdentity.from_record(_constraints(context).get("source_identity"))
    if identity is None or identity.database != "eicu_demo":
        raise ValueError("first_observed_preparation_source_unavailable")
    export, metadata = _metadata_source(identity)
    if verify_source_bytes:
        binding = resolve_raw_source_identity_binding(export_path=export, database="eicu_demo")
        if binding is None or binding.family != "eicu" or "patient" not in binding.table_paths:
            raise ValueError("first_observed_preparation_source_unavailable")
        source = verified_public_demo_raw(Path(binding.table_paths["patient"]).parent)
        if (source is None or source.database != "eicu_demo" or source.archive_sha256 != identity.archive_sha256
                or binding.export_manifest_sha256 != metadata["export_manifest_sha256"]):
            raise ValueError("first_observed_preparation_source_mismatch")
    return ({"adapter_id": _ADAPTER, "read_scope_version": 2, "purpose": "prepare_only", "analysis_authorized": False,
             **({"human_review_checkpoint_sha256": checkpoint_sha256} if checkpoint_sha256 else {}),
             "producer_runtime_sha256": runtime_digest(),
             "database": "eicu_demo", "source_identity_sha256": canonical_sha256(identity.to_dict()),
             "plan_sha256": review.plan_sha256, "context_sha256": context_sha,
             "scientific_review_sha256": scientific_sha, "criterion": matches[0],
             "source_table": "patient", "source_columns": list(REQUIRED_COLUMNS),
             "population_definition": "adult_first_observed_icu_in_single_visible_hospitalization",
             "unknown_identity_or_order_policy": "exclude_or_refuse_source",
             "multiple_hospitalization_policy": "exclude_patient",
             "missing_age_policy": "exclude_patient", "not_lifetime_first_icu": True,
             "read_outcome_columns": False, "public_coordinate_authorized": False,
             "full_archive_and_patient_file_bytes_read": True,
             "official_patient_csv_decompression_streamed": True,
             **metadata}, export)


def proposed_first_observed_scope(*, run_dir: Path, request: Any) -> dict[str, Any]:
    """Review the pinned source and metadata before any patient byte reads."""
    scope, _ = _host_inputs(run_dir=run_dir, request=request, verify_source_bytes=False)
    return scope


def _private_root(*, create: bool) -> Path:
    root = state_paths.state_root() / "private-first-observed"
    if create:
        root.mkdir(mode=0o700, parents=True, exist_ok=True)
    info = root.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_mode & 0o077 or info.st_uid != os.getuid():
        raise ValueError("first_observed_private_directory_invalid")
    return root


def _verify_consumed_owner(*, run_dir: Path, checkpoint: RawPreparationCheckpoint,
                           binding: RawPreparationHostBinding, scope: Mapping[str, Any]) -> None:
    """Keep a consumed preparation decision attached to its live study owner."""
    receipt = verify_consumed_host_review(checkpoint=checkpoint, binding=binding, scope=scope)
    # These are host-layer imports, deliberately kept out of the research
    # materializer: a constructible coordinate is never a preparation grant.
    from easyicu.webserver import agent_pipeline_runs
    from easyicu.webserver.raw_preparation_durable import _owner

    root = _owner(binding, receipt, str(receipt.get("job_id") or ""))
    if Path(root).absolute() != Path(run_dir).absolute():
        raise ValueError("first_observed_preparation_run_owner_changed")
    pending, _ = agent_pipeline_runs.raw_preparation_review_owner(binding.run_id, binding.study_context_id)
    expected = PlanReviewAuthority.model_validate(checkpoint.request.payload.get("plan_review_authority"))
    if expected.execution is not None:
        _assert_pending_request_set(run_dir, pending)
    authorities = [PlanReviewAuthority.model_validate(request.payload["plan_review_authority"])
                   for request in pending.requests
                   if isinstance(request.payload.get("plan_review_authority"), Mapping)]
    if not authorities or any(authority != expected for authority in authorities):
        raise ValueError("first_observed_preparation_plan_owner_changed")


def prepare_first_observed_coordinate(*, run_dir: Path, checkpoint: RawPreparationCheckpoint,
                                      binding: RawPreparationHostBinding,
                                      scope: Mapping[str, Any]) -> FirstObservedIcuCoordinate:
    """Consume host approval before reading source rows and attest exact bytes."""
    expected, export = _host_inputs(run_dir=run_dir, request=checkpoint.request, verify_source_bytes=False)
    scope = dict(scope)
    if scope != expected:
        raise ValueError("first_observed_preparation_scope_mismatch")
    verify_raw_preparation_review(checkpoint=checkpoint, requested_scope=scope, scope_key=_SCOPE_KEY,
        expected_summary=_SUMMARY, registered_database="eicu_demo",
        registered_identity_sha256=scope["source_identity_sha256"], plan_sha256=scope["plan_sha256"],
        context_sha256=scope["context_sha256"], scientific_review_sha256=scope["scientific_review_sha256"],
        allow_execution_bound=bool(scope.get("human_review_checkpoint_sha256")))
    consume_host_review(checkpoint=checkpoint, scope=scope, **binding.model_dump(mode="python"))
    _verify_consumed_owner(run_dir=run_dir, checkpoint=checkpoint, binding=binding, scope=scope)
    # Only the consumed, exact host decision permits the first source-byte read.
    current_scope, current_export = _host_inputs(run_dir=run_dir, request=checkpoint.request)
    if current_scope != scope or current_export != export:
        raise ValueError("first_observed_preparation_scope_mismatch")
    source_binding = resolve_raw_source_identity_binding(export_path=export, database="eicu_demo")
    if source_binding is None or "patient" not in source_binding.table_sha256:
        raise ValueError("first_observed_preparation_source_unavailable")
    patient_sha256 = source_binding.table_sha256["patient"]
    candidate = derive_from_verified_eicu_demo(export)
    proof = candidate.receipt.get("source_binding", {})
    completeness = proof.get("complete_patient_source", {})
    if (proof.get("official_release_independently_verified") is not True
        or completeness.get("schema_version") != "easyicu.eicu_demo_complete_patient_source/1"
        or completeness.get("all_source_rows_compared") is not True
        or completeness.get("archive_sha256") != scope["archive_sha256"]
        or proof.get("patient_table_sha256") != patient_sha256
        or proof.get("export_manifest_sha256") != scope["export_manifest_sha256"]):
        raise ValueError("first_observed_preparation_source_proof_mismatch")
    source_flow = _private_source_flow(candidate)
    buffer = io.BytesIO()
    pq.write_table(pa.Table.from_pandas(candidate.frame, preserve_index=False), buffer)
    data = buffer.getvalue()
    body = {"schema_version": "easyicu.first-observed-coordinate-attestation/3",
            "scope_sha256": canonical_sha256(scope), "checkpoint_sha256": checkpoint.checkpoint_sha256,
            "binding_sha256": canonical_sha256(binding.model_dump(mode="json")),
            "coordinate_sha256": hashlib.sha256(data).hexdigest(), "patient_table_sha256": patient_sha256,
            "source_complete_six_columns": True,
            "analysis_authorized": False, "private_source_flow": source_flow}
    # The source and reviewed plan can change while the candidate is being
    # computed. Do not publish an attestation for the earlier snapshot after
    # its current host bindings have stopped matching.
    current_scope, current_export = _host_inputs(run_dir=run_dir, request=checkpoint.request)
    if current_scope != scope or current_export != export:
        raise ValueError("first_observed_preparation_scope_changed_before_publication")
    current_binding = resolve_raw_source_identity_binding(export_path=export, database="eicu_demo")
    if current_binding is None or current_binding.table_sha256.get("patient") != patient_sha256:
        raise ValueError("first_observed_preparation_source_changed_before_publication")
    _verify_consumed_owner(run_dir=run_dir, checkpoint=checkpoint, binding=binding, scope=scope)
    document = canonical_json({"body": body, "signature": _signature(body)}).encode()
    root = _private_root(create=True)
    with AnchoredDirectory.open(root) as directory:
        _publish_once(directory, binding.nonce + ".parquet", data)
        _publish_once(directory, binding.nonce + ".json", document)
        directory.assert_still_selected()
    return _coordinate(root, binding, scope, checkpoint, body)


def _coordinate(root: Path, binding: RawPreparationHostBinding, scope: Mapping[str, Any], checkpoint: RawPreparationCheckpoint,
                body: Mapping[str, Any]) -> FirstObservedIcuCoordinate:
    return FirstObservedIcuCoordinate(root / (binding.nonce + ".parquet"), str(body["coordinate_sha256"]),
        scope["archive_sha256"], body["patient_table_sha256"], scope["export_manifest_sha256"],
        scope["plan_sha256"], checkpoint.checkpoint_sha256, preparation_run_id=binding.run_id,
        preparation_binding_sha256=canonical_sha256(binding.model_dump(mode="json")))


def _resolve_consumed_coordinate_evidence(*, run_dir: Path, checkpoint: RawPreparationCheckpoint,
                                         binding: RawPreparationHostBinding,
                                         scope: Mapping[str, Any]) -> tuple[FirstObservedIcuCoordinate, dict[str, Any] | None]:
    """Verify completed private evidence without rereading the source tables.

    This is for reporting an already consumed decision, never a new read grant.
    """
    scope = dict(scope)
    if checkpoint.request.payload.get(_SCOPE_KEY) != scope:
        raise ValueError("first_observed_preparation_scope_mismatch")
    _verify_consumed_owner(run_dir=run_dir, checkpoint=checkpoint, binding=binding, scope=scope)
    root = _private_root(create=False)
    with AnchoredDirectory.open(root) as directory:
        document = json.loads(directory.read_bytes(binding.nonce + ".json", max_bytes=64 * 1024))
        with directory.open_regular(binding.nonce + ".parquet") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        directory.assert_still_selected()
    body, signature = document["body"], document["signature"]
    if not isinstance(signature, str) or not hmac.compare_digest(signature, _signature(body)):
        raise ValueError("first_observed_attestation_invalid")
    version = body.get("schema_version")
    if version != "easyicu.first-observed-coordinate-attestation/3" or scope.get("read_scope_version") != 2:
        raise ValueError("first_observed_attestation_binding_mismatch")
    expected_body = {"schema_version": version,
                     "scope_sha256": canonical_sha256(scope), "checkpoint_sha256": checkpoint.checkpoint_sha256,
                     "binding_sha256": canonical_sha256(binding.model_dump(mode="json")), "coordinate_sha256": digest,
                     "source_complete_six_columns": True, "analysis_authorized": False}
    flow = body.get("private_source_flow")
    patient_sha256 = body.get("patient_table_sha256")
    if (not isinstance(flow, dict) or not isinstance(patient_sha256, str)
            or len(patient_sha256) != 64 or any(c not in "0123456789abcdef" for c in patient_sha256)):
        raise ValueError("first_observed_attestation_binding_mismatch")
    expected_body["private_source_flow"] = flow
    expected_body["patient_table_sha256"] = patient_sha256
    if body != expected_body:
        raise ValueError("first_observed_attestation_binding_mismatch")
    return _coordinate(root, binding, scope, checkpoint, body), flow


def _resolve_prepared_first_observed(*, run_dir: Path, checkpoint: RawPreparationCheckpoint,
                                     binding: RawPreparationHostBinding,
                                     scope: Mapping[str, Any]) -> tuple[FirstObservedIcuCoordinate, dict[str, Any] | None]:
    """Reverify current source as well as the already consumed coordinate."""
    coordinate, flow = _resolve_consumed_coordinate_evidence(
        run_dir=run_dir, checkpoint=checkpoint, binding=binding, scope=scope)
    expected, export = _host_inputs(run_dir=run_dir, request=checkpoint.request)
    if dict(scope) != expected:
        raise ValueError("first_observed_preparation_scope_mismatch")
    current_binding = resolve_raw_source_identity_binding(export_path=export, database="eicu_demo")
    if current_binding is None or current_binding.table_sha256.get("patient") != coordinate.patient_table_sha256:
        raise ValueError("first_observed_preparation_source_changed")
    return coordinate, flow


def resolve_prepared_first_observed_coordinate(*, run_dir: Path, checkpoint: RawPreparationCheckpoint,
                                               binding: RawPreparationHostBinding,
                                               scope: Mapping[str, Any]) -> FirstObservedIcuCoordinate:
    """Reverify the host decision, reviewed scope, signed result and exact bytes."""
    coordinate, _ = _resolve_prepared_first_observed(
        run_dir=run_dir, checkpoint=checkpoint, binding=binding, scope=scope)
    return coordinate


def resolve_private_first_observed_source_flow(*, run_dir: Path, checkpoint: RawPreparationCheckpoint,
                                               binding: RawPreparationHostBinding,
                                               scope: Mapping[str, Any]) -> dict[str, Any]:
    """Read the signed source-global counts privately, never a public flow table.

    Older signed coordinates remain usable, but lack this record; they cannot
    supply a source-flow report without repeating preparation under a new grant.
    """
    _, flow = _resolve_prepared_first_observed(
        run_dir=run_dir, checkpoint=checkpoint, binding=binding, scope=scope)
    if flow is None:
        raise ValueError("first_observed_source_flow_unavailable")
    return flow

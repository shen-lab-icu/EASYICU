"""Host-owned adapter from a paused scientific review to a proposal.

This does not approve a plan, read patient rows or launch a producer.
"""
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping
from datetime import datetime, timezone
import hashlib
import json
import re

from easyicu.concept.loader import load_dictionary
from easyicu.concept.provenance import capture_data_provenance, metadata_digest
from easyicu.config import load_src_cfg
from easyicu.databases.research_plan_requirements import _constraints
from easyicu.research_agent.authority.evidence_store import EvidenceStore
from easyicu.research_agent.authority.filesystem import AnchoredDirectory
from easyicu.research_agent.authority.plan_review import PlanReviewAuthority
from easyicu.research_agent.authority.raw_preparation import RawPreparationCheckpoint
from easyicu.research_agent.orchestration.workflow import HumanReviewRequest
from easyicu.research_agent.canonical_json import canonical_json, canonical_sha256
from easyicu.research_agent.research_context.export_selection import SourceIdentity
from easyicu.research_agent.research_context.typed import parse_research_context_json
from easyicu.research_agent.schema import AnalysisPlan
from easyicu.webserver import agent_pipeline_runs, sources
from easyicu.webserver.raw_preparation_decisions import _publish_once
from easyicu.webserver.raw_preparation_dependencies import (
    candidate_from_scientific_review, reviewed_plan_provenance_concepts,
    reviewed_population_provenance_concepts,
)


def _plan_authority(pending: Any) -> PlanReviewAuthority:
    candidates = []
    for request in pending.requests:
        payload = request.payload if isinstance(request.payload, Mapping) else {}
        if isinstance(payload.get("plan_review_authority"), Mapping):
            candidates.append(PlanReviewAuthority.model_validate(payload["plan_review_authority"]))
    if not candidates or any(value != candidates[0] for value in candidates[1:]):
        raise ValueError("raw_preparation_plan_authority_unavailable")
    return candidates[0]


def _first_observed_service_available() -> bool:
    """A review offer requires the durable producer and deployment lease."""
    from easyicu.webserver.durable_research import current_service

    service = current_service()
    if service is None or service.manager._journal is None or not service.dispatcher._launched:
        return False
    try:
        service._require_handoff_deployment_lease()
    except (OSError, ValueError, RuntimeError):
        return False
    return True


def _single_table_pending(plan: PlanReviewAuthority, dependencies: dict, identity: SourceIdentity):
    """Offer a partial step only when the sole blocker is the unknown cohort."""
    digest = dependencies.get("declaration_sha256")
    body = {key: value for key, value in dependencies.items() if key != "declaration_sha256"}
    declaration = body.get("declaration", {})
    files = body.get("files", [])
    if (digest != canonical_sha256(body) or body.get("blocker_codes") != ["dependency_cohort_inputs_unknown"]
            or declaration.get("plan_sha256") != plan.plan_sha256
            or declaration.get("source_identity_sha256") != canonical_sha256(identity.to_dict())
            or len(files) != 1 or files[0].get("format") != "csv"):
        return None
    scope = {"adapter_id": "registered-csv-projection/1", "purpose": "prepare_single_table_only",
             "source_identity": identity.to_dict(), "relative_file": files[0]["path"],
             "columns": files[0]["columns"], "plan_sha256": plan.plan_sha256,
             "dependencies_sha256": digest}
    from easyicu.webserver.reviewed_csv_projection import require_reviewable_csv_scope
    require_reviewable_csv_scope(scope)
    payload = {"plan_review_authority": plan.model_dump(mode="json"), "raw_preparation_scope": scope}
    summary = (f"读取登记来源 {identity.database}/{identity.registration_id} 中的 "
               f"{scope['relative_file']} 全部行和全部字段；仅输出字段：{', '.join(scope['columns'])}。"
               "队列依赖尚未解决，此步骤不构成完整队列或研究分析。 "
               "This reads the entire file and outputs only the listed columns; cohort and analysis remain incomplete.")
    if len(summary) > 1000:
        return None
    request = HumanReviewRequest.create(kind="scientific_stop", summary=summary, payload=payload,
                                        authority_sha256=canonical_sha256(payload))
    return RawPreparationCheckpoint.pending(plan_sha256=plan.plan_sha256, request=request)


def _partial_multitable_preview(*, identity: SourceIdentity, plan: PlanReviewAuthority,
                                context: Any, run_dir: Path, candidate: dict) -> dict:
    """Derive a review-only projection, never a dependency closure or grant."""
    from easyicu.research_agent.authority.runtime_artifacts import verified_run_evidence_path
    from easyicu.webserver.reviewed_multitable_source import build_source_bound_multitable_scope

    evidence = EvidenceStore(run_dir).get("research_context")
    path = verified_run_evidence_path(run_dir, evidence) if evidence is not None else None
    if path is None or evidence.sha256 != candidate.get("research_context_evidence_sha256"):
        raise ValueError("raw_preparation_context_evidence_missing")
    with AnchoredDirectory.open(path.parent) as directory:
        raw = directory.read_bytes(path.name, max_bytes=5 * 1024 * 1024)
        directory.assert_still_selected()
    if hashlib.sha256(raw).hexdigest() != evidence.sha256:
        raise ValueError("raw_preparation_context_evidence_mismatch")
    evidence_context = parse_research_context_json(raw.decode("utf-8"))
    if canonical_sha256(evidence_context.model_dump(mode="json")) != canonical_sha256(context.model_dump(mode="json")):
        raise ValueError("raw_preparation_context_evidence_mismatch")
    if candidate.get("source_identity_sha256") != canonical_sha256(identity.to_dict()):
        raise ValueError("raw_preparation_source_mismatch")
    files = candidate.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("raw_preparation_multitable_roster_missing")
    requested: dict[str, list[str]] = {}
    expected = []
    for item in files:
        if not isinstance(item, dict) or not isinstance(item.get("table"), str) or not isinstance(item.get("columns"), list):
            raise ValueError("raw_preparation_multitable_roster_invalid")
        table, columns = item["table"], item["columns"]
        if table in requested and requested[table] != columns:
            raise ValueError("raw_preparation_multitable_columns_inconsistent")
        requested[table] = columns
        expected.append({"table": table, "path": item.get("path"), "columns": columns})
    if len(requested) < 2:
        raise ValueError("raw_preparation_multitable_roster_missing")
    scope, roster = build_source_bound_multitable_scope(identity=identity, requested_columns=requested,
                                                        plan_sha256=plan.plan_sha256)
    expected.sort(key=lambda row: (row["table"], row["path"]))
    if scope["inputs"] != expected or roster.get("source_identity_sha256") != candidate["source_identity_sha256"]:
        raise ValueError("raw_preparation_multitable_candidate_changed")
    return {"adapter_id": scope["adapter_id"], "plan_sha256": plan.plan_sha256,
            "scope_sha256": canonical_sha256(scope), "roster_candidate_sha256": roster["candidate_sha256"],
            "inputs": scope["inputs"], "partial_projection_only": True,
            "would_stage_full_input_files": True, "would_project_all_rows": True,
            "cohort_complete": False, "joins_performed": False, "derived_features_computed": False,
            "dependency_closure_verified": False, "approvable": False, "analysis_authorized": False}


def create_candidate_for_review(*, run_id: str, study_context_id: str) -> dict:
    pending, study = agent_pipeline_runs.raw_preparation_review_owner(run_id, study_context_id)
    run_dir = Path(pending.run_dir)
    if not run_dir.is_absolute() or ".." in run_dir.parts:
        raise ValueError("raw_preparation_run_unavailable")
    with AnchoredDirectory.open(run_dir) as directory:
        context_bytes = directory.read_bytes("research_context.json", max_bytes=5 * 1024 * 1024)
        directory.assert_still_selected()
    context = parse_research_context_json(context_bytes.decode("utf-8"))
    authority = _plan_authority(pending)
    plan = AnalysisPlan.model_validate(authority.model_dump(mode="json")["plan_payload"])
    identity = SourceIdentity.from_record(_constraints(context).get("source_identity"))
    if identity is None:
        raise ValueError("raw_preparation_source_unavailable")
    if identity.database == "eicu_demo":
        from easyicu.webserver.pi_copilot.population_spec_review import reviewed_population_spec
        population = reviewed_population_spec(run_dir=run_dir, pending=pending)
        criteria = population.get("criteria", ()) if isinstance(population, Mapping) else ()
        matches = [item for item in criteria if isinstance(item, Mapping)
                   and isinstance(item.get("criterion"), Mapping)
                   and item["criterion"].get("kind") == "first_observed_icu_single_hospital_adult"
                   and item["criterion"].get("role") == "include"]
        if matches:
            if len(matches) != 1 or population.get("plan_sha256") != authority.plan_sha256:
                raise ValueError("raw_preparation_first_observed_population_ambiguous")
            requests = [item for item in pending.requests
                        if isinstance(item.payload.get("plan_review_authority"), Mapping)
                        and PlanReviewAuthority.model_validate(item.payload["plan_review_authority"]) == authority]
            if len(requests) != 1:
                raise ValueError("raw_preparation_first_observed_review_ambiguous")
            from easyicu.webserver.first_observed_preparation import (
                _SCOPE_KEY, _SUMMARY, proposed_first_observed_scope,
            )
            scope = proposed_first_observed_scope(run_dir=run_dir, request=requests[0])
            if scope.get("criterion") != matches[0]["criterion"]:
                raise ValueError("raw_preparation_first_observed_population_changed")
            payload = {"plan_review_authority": authority.model_dump(mode="json"), _SCOPE_KEY: scope}
            request = HumanReviewRequest.create(kind="scientific_stop", summary=_SUMMARY, payload=payload,
                                                authority_sha256=canonical_sha256(payload))
            checkpoint = RawPreparationCheckpoint.pending(plan_sha256=authority.plan_sha256, request=request)
            with AnchoredDirectory.open(run_dir) as directory:
                if not directory.is_absent("sic_raw_preparation_checkpoint.json"):
                    raise ValueError("raw_preparation_pending_conflict")
                try:
                    _publish_once(directory, "raw_preparation_checkpoint.json", checkpoint.model_dump_json().encode())
                except FileExistsError:
                    existing = RawPreparationCheckpoint.model_validate_json(
                        directory.read_bytes("raw_preparation_checkpoint.json", max_bytes=1024 * 1024))
                    if (existing.state != "pending" or existing.request.payload != checkpoint.request.payload
                            or existing.request.summary != checkpoint.request.summary):
                        raise ValueError("raw_preparation_pending_conflict")
                    checkpoint = existing
                directory.assert_still_selected()
            can_review = _first_observed_service_available()
            return {"status": "pending", "checkpoint_sha256": checkpoint.checkpoint_sha256,
                    "approvable": can_review, "analysis_authorized": False,
                    **({} if can_review else {"reason_code": "raw_preparation_durable_service_unavailable"})}
    matches = [item for item in sources.load_registry().get("sources", [])
               if isinstance(item, Mapping) and item.get("path")
               and sources.registered_source_identity(str(item["path"])) == identity]
    if len(matches) != 1:
        raise ValueError("raw_preparation_source_unavailable")
    source_root = Path(str(matches[0]["path"]))
    config = load_src_cfg(identity.database)
    # A criterion whose input is unavailable may not compile to a plan
    # predicate yet. The host-verified source spec still names the concept
    # needed for a preparation proposal; this grants no rows or execution.
    from easyicu.webserver.pi_copilot.population_spec_review import reviewed_population_spec

    population = reviewed_population_spec(run_dir=run_dir, pending=pending)
    concepts = tuple(dict.fromkeys((
        *reviewed_plan_provenance_concepts(context=context, plan=plan),
        *reviewed_population_provenance_concepts(
            projection=population, plan_sha256=authority.plan_sha256),
    )))
    if not concepts:
        raise ValueError("raw_preparation_concepts_unavailable")
    snapshot = capture_data_provenance(load_dictionary(include_sofa2=True), concepts=concepts,
        source_database=identity.database, class_prefixes=tuple(config.class_prefix), source_config=config,
        source_config_sha256=metadata_digest(config.model_dump(mode="json")))
    result = candidate_from_scientific_review(plan_result=SimpleNamespace(
        context=context, plan=plan, evidence=EvidenceStore(run_dir)), plan_authority=authority,
        run_dir=run_dir, source_root=source_root, snapshot=snapshot, source_config=config)
    if result["pending"] is None:
        result["pending"] = _single_table_pending(authority, result["dependencies"], identity)
    if result["pending"] is None:
        blocked = {"status": "blocked", "blocker_codes": result["dependencies"]["blocker_codes"]}
        # This is only a metadata preview, not the blocked dependency roster
        # and not a grant. Source/context and the current configuration must
        # still match after the read, or the preview is omitted.
        from easyicu.webserver import run_artifact_disclosure, study_contexts
        from easyicu.webserver.source_partition_candidates import candidate_from_bound_research_context
        from easyicu.research_agent.authority.filesystem import AuthorityFilesystemError
        try:
            candidate = candidate_from_bound_research_context(study=study, run_dir=run_dir)
            current = study_contexts.get_context(study_context_id)
            if (current and study_contexts.scientific_configuration_sha256(current)
                    == study_contexts.scientific_configuration_sha256(study)):
                partial = None
                try:
                    partial = _partial_multitable_preview(
                        identity=identity, plan=authority, context=context, run_dir=run_dir, candidate=candidate)
                except (ValueError, OSError, TypeError, KeyError, AuthorityFilesystemError):
                    pass
                current = study_contexts.get_context(study_context_id)
                if (current and study_contexts.scientific_configuration_sha256(current)
                        == study_contexts.scientific_configuration_sha256(study)
                        and run_artifact_disclosure.scan_browser_projection(candidate).get("passed")):
                    blocked["source_partition_candidate"] = candidate
                    if partial and run_artifact_disclosure.scan_browser_projection(partial).get("passed"):
                        blocked["partial_multitable_projection"] = partial
        except (ValueError, OSError, TypeError, KeyError, AuthorityFilesystemError):
            pass
        return blocked
    checkpoint = result["pending"]
    with AnchoredDirectory.open(run_dir) as directory:
        try:
            _publish_once(directory, "raw_preparation_checkpoint.json", checkpoint.model_dump_json().encode())
        except FileExistsError:
            existing = RawPreparationCheckpoint.model_validate_json(
                directory.read_bytes("raw_preparation_checkpoint.json", max_bytes=1024 * 1024))
            if (existing.state != "pending" or existing.request.payload != checkpoint.request.payload
                    or existing.request.summary != checkpoint.request.summary):
                raise ValueError("raw_preparation_pending_conflict")
            checkpoint = existing
        directory.assert_still_selected()
    partial = checkpoint.request.payload.get("raw_preparation_scope", {}).get("adapter_id") == "registered-csv-projection/1"
    # The single-table executor is not yet restartable after consuming its
    # one-time grant, and its delivery is not integrated into the workspace.
    # Keep the review offer closed until both are available.
    return {"status": "pending", "checkpoint_sha256": checkpoint.checkpoint_sha256,
            "approvable": False, "reason_code": "single_table_recovery_delivery_unavailable" if partial else "producer_consumption_closure_unverified"}


def available_host_review(*, run_id: str, study_context_id: str):
    """Offer only current scopes accepted by an internal closed adapter."""
    from easyicu.webserver import reviewed_snapshot_producer, study_contexts
    from easyicu.webserver.host_action_contracts import RawPreparationReviewDecision

    pending, study = agent_pipeline_runs.raw_preparation_review_owner(run_id, study_context_id)
    with AnchoredDirectory.open(Path(pending.run_dir)) as directory:
        checkpoint = RawPreparationCheckpoint.model_validate_json(
            directory.read_bytes("raw_preparation_checkpoint.json", max_bytes=1024 * 1024))
        directory.assert_still_selected()
    if checkpoint.state != "pending":
        raise ValueError("raw_preparation_pending_unavailable")
    expiry = datetime.fromisoformat(checkpoint.expires_at.replace("Z", "+00:00"))
    if expiry.tzinfo is None or datetime.now(timezone.utc) >= expiry:
        raise ValueError("raw_preparation_pending_expired")
    raw_scope = checkpoint.request.payload.get("raw_preparation_scope")
    first_scope = checkpoint.request.payload.get("first_observed_icu_preparation_scope")
    if raw_scope is not None and first_scope is not None:
        raise ValueError("raw_preparation_pending_conflict")
    scope = json.loads(canonical_json(first_scope if first_scope is not None else raw_scope or {}))
    # The adapter is in-tree host code. Unknown or inconsistent scope refuses.
    if scope.get("adapter_id") == "registered-csv-projection/1":
        raise ValueError("raw_preparation_partial_review_surface_unavailable")
    elif scope.get("adapter_id") == "eicu-demo-first-observed-single-hospital-adult/1":
        from easyicu.webserver.first_observed_preparation import proposed_first_observed_scope
        if not _first_observed_service_available():
            raise ValueError("raw_preparation_durable_service_unavailable")
        if proposed_first_observed_scope(run_dir=Path(pending.run_dir), request=checkpoint.request) != scope:
            raise ValueError("raw_preparation_pending_stale")
    else:
        reviewed_snapshot_producer.require_reviewable_scope(scope)
    current_study = study_contexts.get_context(study_context_id)
    old_digest = study_contexts.scientific_configuration_sha256(study)
    if not current_study or study_contexts.scientific_configuration_sha256(current_study) != old_digest:
        raise ValueError("raw_preparation_scientific_context_changed")
    # Reconstruct from the current review, registration and exact metadata.
    checked = create_candidate_for_review(run_id=run_id, study_context_id=study_context_id)
    if checked.get("status") != "pending" or checked.get("checkpoint_sha256") != checkpoint.checkpoint_sha256:
        raise ValueError("raw_preparation_pending_stale")
    choice = RawPreparationReviewDecision(run_id=run_id, scientific_configuration_sha256=old_digest,
        checkpoint_sha256=checkpoint.checkpoint_sha256, scope_sha256=canonical_sha256(scope),
        review_authority_sha256=checkpoint.request.authority_sha256,
        nonce=canonical_sha256({"run_id": run_id, "study_context_id": study_context_id,
                                "checkpoint_sha256": checkpoint.checkpoint_sha256}))
    return choice, checkpoint, scope, Path(pending.run_dir)


def recoverable_first_observed_review(*, run_id: str, study_context_id: str):
    """Offer the original decision only for a signed, unstarted preparation.

    This does not resume a job or grant new authority.  The receiving route
    independently verifies the original host job, source and lifecycle.
    """
    from easyicu import state_paths
    from easyicu.webserver.durable_research import current_service
    from easyicu.webserver.first_observed_preparation import _SCOPE_KEY, proposed_first_observed_scope
    from easyicu.webserver.host_action_contracts import RawPreparationReviewDecision
    from easyicu.webserver.raw_preparation_decisions import RawPreparationHostBinding, recover_host_review

    pending_review, _ = agent_pipeline_runs.raw_preparation_review_owner(run_id, study_context_id)
    run_dir = Path(pending_review.run_dir)
    with AnchoredDirectory.open(run_dir) as directory:
        checkpoint = RawPreparationCheckpoint.model_validate_json(
            directory.read_bytes("raw_preparation_checkpoint.json", max_bytes=1024 * 1024))
        directory.assert_still_selected()
    scope = checkpoint.request.payload.get(_SCOPE_KEY)
    if (checkpoint.state != "approved" or not isinstance(scope, Mapping)
            or scope.get("adapter_id") != "eicu-demo-first-observed-single-hospital-adult/1"):
        raise ValueError("raw_preparation_recovery_unavailable")
    body = checkpoint.model_dump(mode="json", exclude={"checkpoint_sha256"})
    body.update(state="pending", decision=None)
    pending_sha = canonical_sha256(body)
    original = RawPreparationCheckpoint.model_validate({**body, "checkpoint_sha256": pending_sha})
    nonce = canonical_sha256({"run_id": run_id, "study_context_id": study_context_id,
                              "checkpoint_sha256": pending_sha})
    root = state_paths.state_root() / "private-preparation-decisions"
    with AnchoredDirectory.open(root) as directory:
        receipt = json.loads(directory.read_bytes(nonce + ".json", max_bytes=64 * 1024))
        directory.assert_still_selected()
    binding = RawPreparationHostBinding(project_id=receipt["project_id"], session_id=receipt["session_id"],
        study_context_id=study_context_id, run_id=run_id, nonce=nonce)
    job_id = receipt["job_id"]
    if not isinstance(job_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", job_id):
        raise ValueError("raw_preparation_recovery_unavailable")
    service = current_service()
    if (service is None or service.manager._journal is None or not service.dispatcher._launched
            or service.store.get(job_id) is not None or service.manager.get(job_id) is None):
        raise ValueError("raw_preparation_recovery_unavailable")
    job = service.manager.get(job_id)
    state = job.snapshot()
    if (job.kind != "raw-preparation-review" or job.is_cancel_requested()
            or not (state.get("status") == "running" or
                    (state.get("status") == "failed" and state.get("recovered") and state.get("interrupted")))):
        raise ValueError("raw_preparation_recovery_unavailable")
    lease = run_dir / (".preparation-execution-" + job_id)
    if lease.exists() or lease.is_symlink():
        raise ValueError("raw_preparation_recovery_unavailable")
    with AnchoredDirectory.open(root) as directory:
        if any(not directory.is_absent(name) for name in (
            nonce + ".used", job_id + ".preparation-started",
            job_id + ".preparation-snapshot.json", job_id + ".preparation-completed.json")):
            raise ValueError("raw_preparation_recovery_unavailable")
        directory.assert_still_selected()
    service._require_handoff_deployment_lease()
    if recover_host_review(pending=original, binding=binding, scope=scope, job_id=job_id, run_dir=run_dir) != checkpoint:
        raise ValueError("raw_preparation_recovery_unavailable")
    if proposed_first_observed_scope(run_dir=run_dir, request=checkpoint.request) != dict(scope):
        raise ValueError("raw_preparation_recovery_unavailable")
    choice = RawPreparationReviewDecision(run_id=run_id,
        scientific_configuration_sha256=receipt["scientific_configuration_sha256"],
        checkpoint_sha256=pending_sha, scope_sha256=canonical_sha256(dict(scope)),
        review_authority_sha256=checkpoint.request.authority_sha256, nonce=nonce)
    return choice, binding.session_id

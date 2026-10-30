"""Host-owned adapter from a paused scientific review to a proposal.

This does not approve a plan, read patient rows or launch a producer.
"""
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping
from datetime import datetime, timezone
import json

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
from easyicu.webserver.raw_preparation_dependencies import candidate_from_scientific_review


def _plan_authority(pending: Any) -> PlanReviewAuthority:
    candidates = []
    for request in pending.requests:
        payload = request.payload if isinstance(request.payload, Mapping) else {}
        if isinstance(payload.get("plan_review_authority"), Mapping):
            candidates.append(PlanReviewAuthority.model_validate(payload["plan_review_authority"]))
    if not candidates or any(value != candidates[0] for value in candidates[1:]):
        raise ValueError("raw_preparation_plan_authority_unavailable")
    return candidates[0]


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


def create_candidate_for_review(*, run_id: str, study_context_id: str) -> dict:
    pending, _ = agent_pipeline_runs.raw_preparation_review_owner(run_id, study_context_id)
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
    matches = [item for item in sources.load_registry().get("sources", [])
               if isinstance(item, Mapping) and item.get("path")
               and sources.registered_source_identity(str(item["path"])) == identity]
    if len(matches) != 1:
        raise ValueError("raw_preparation_source_unavailable")
    source_root = Path(str(matches[0]["path"]))
    config = load_src_cfg(identity.database)
    concepts = list(dict.fromkeys(str(item.source_concept or item.name) for item in context.variables))
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
        return {"status": "blocked", "blocker_codes": result["dependencies"]["blocker_codes"]}
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
    # The current shared UI does not yet render this scope's complete
    # authorization details. A candidate can be previewed, but cannot be
    # approved until the review surface has been updated and verified.
    return {"status": "pending", "checkpoint_sha256": checkpoint.checkpoint_sha256,
            "approvable": False, "reason_code": "single_table_review_surface_unavailable" if partial else "producer_consumption_closure_unverified"}


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
    scope = json.loads(canonical_json(checkpoint.request.payload.get("raw_preparation_scope") or {}))
    # The adapter is in-tree host code. Unknown or inconsistent scope refuses.
    if scope.get("adapter_id") == "registered-csv-projection/1":
        raise ValueError("raw_preparation_partial_review_surface_unavailable")
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

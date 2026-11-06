"""Shared synthetic host-authorized raw preparation setup."""
from types import SimpleNamespace

import pandas as pd

from easyicu.databases.research_plan_requirements import _constraints
from easyicu.databases.research_substitutions import case_duration_selection
from easyicu.research_agent.authority.evidence_store import EvidenceStore
from easyicu.research_agent.authority.plan_review import PlanReviewAuthority
from easyicu.research_agent.canonical_json import canonical_sha256
from easyicu.research_agent.orchestration.scientific_plan_review_gate import persist_or_validate_scientific_plan_review
from easyicu.research_agent.orchestration.workflow import HumanReviewRequest
from easyicu.research_agent.research_context.export_selection import SourceIdentity
from easyicu.webserver import sources
from easyicu.webserver.sic_raw_preparation_grant import SICRawPreparationCheckpoint, SICRawPreparationScope, render_sic_raw_preparation_scope, sic_preparation_schema_sha256
from easyicu.webserver.raw_preparation_decisions import RawPreparationHostBinding, record_host_review
from easyicu.webserver.time_origin_source_authority import prepare_case_duration_metadata_plan
from tests.support.sic_references import write_sic_references
from tests.support.source_substitutions import _review
from tests.support.signed_case_duration import signed_case_duration_case


def _case(tmp_path, monkeypatch, *, approved=True, summary_override=None):
    context, plan, *_ = signed_case_duration_case()
    context = context.model_copy(update={"variables": [
        v.model_copy(update={"source_concept": "lact"}) if v.name == "lact" else v
        for v in context.variables
    ]})
    identity = SourceIdentity.from_record(_constraints(context)["source_identity"])
    assert identity is not None
    source_path = tmp_path / "source"
    source_path.mkdir()
    pd.DataFrame({name: pd.Series(dtype="int64") for name in (
        "CaseID", "ICUOffset", "TimeOfStay", "OffsetOfDeath", "HospitalDischargeType",
    )}).to_parquet(source_path / "cases.parquet", index=False)
    write_sic_references(source_path)
    (source_path / "laboratory").mkdir()
    pd.DataFrame({name: pd.Series(dtype="int64") for name in (
        "CaseID", "Offset", "LaboratoryID", "LaboratoryValue",
    )}).to_parquet(source_path / "laboratory" / "1.parquet", index=False)
    monkeypatch.setattr(sources, "registered_source_identity", lambda _path: identity)
    receipt = prepare_case_duration_metadata_plan(source_path, expected_identity=identity)
    base = _review(context, plan)
    facts = dict(base.facts)
    facts["database_case_duration_review"] = {
        **facts["database_case_duration_review"],
        "selection": case_duration_selection(context).model_dump(mode="json"),
        "metadata_plan_receipt": receipt.to_dict(),
        "metadata_status": "source_metadata_verified_not_executable",
    }
    review = base.model_copy(update={"facts": facts})
    evidence = EvidenceStore(tmp_path)
    persist_or_validate_scientific_plan_review(run_dir=tmp_path, evidence=evidence, current_review=review)
    record = evidence.get("scientific_plan_review")
    scope = SICRawPreparationScope(
        source_identity_sha256=canonical_sha256(identity.to_dict()),
        plan_sha256=canonical_sha256(plan.model_dump(mode="json")),
        context_sha256=canonical_sha256(context.model_dump(mode="json")),
        metadata_receipt_sha256=canonical_sha256(receipt.to_dict()),
        source_schema_sha256=sic_preparation_schema_sha256(
            source_path, ("cases.parquet", "d_references.parquet", "laboratory/1.parquet"),
        ),
        output_concepts=("care_unit_to_case_closure_days", "death_during_case", "death_time_during_case_hours", "lact"),
        files=("cases.parquet", "d_references.parquet", "laboratory/1.parquet"),
        case_columns=("CaseID", "HospitalDischargeType", "ICUOffset", "OffsetOfDeath", "TimeOfStay"),
        laboratory_columns=("CaseID", "LaboratoryID", "LaboratoryValue", "Offset"),
    )
    plan_review = PlanReviewAuthority.create(plan=plan, evidence_sha256={"scientific_plan_review": record.sha256})
    payload = {"plan_review_authority": plan_review.model_dump(mode="json"),
               "sic_raw_preparation_scope": scope.model_dump(mode="json")}
    request = HumanReviewRequest.create(kind="scientific_stop", summary=summary_override or render_sic_raw_preparation_scope(scope),
                                        payload=payload, authority_sha256=canonical_sha256(payload))
    checkpoint = SICRawPreparationCheckpoint.pending(plan_sha256=scope.plan_sha256, request=request)
    host_binding = None
    if approved:
        from easyicu.webserver import jobs
        from easyicu.webserver import agent_pipeline_runs
        from easyicu.webserver.host_action_contracts import (
            HostActionRequest, HostActionTag, RawPreparationReviewDecision, decision_key, host_action_id,
        )
        from easyicu.webserver.pi_copilot import service

        monkeypatch.setenv("EASYICU_HOME", str(tmp_path / "hosthome"))
        choice = RawPreparationReviewDecision(
            run_id="synthetic-run", scientific_configuration_sha256="a" * 64,
            checkpoint_sha256=checkpoint.checkpoint_sha256, scope_sha256=canonical_sha256(scope.model_dump(mode="json")),
            review_authority_sha256=request.authority_sha256, nonce="b" * 64,
        )
        action = HostActionRequest(session_id="synthetic-session", project_id="synthetic-project",
                                   action_code="review_raw_preparation", decision=choice)
        key = decision_key(choice, review_decision="approved")
        tag = HostActionTag(session_id=action.session_id, project_id=action.project_id,
            study_context_id="synthetic-study", action_code=action.action_code, decision_key=key,
            action_id=host_action_id(action.session_id, action.action_code, key))
        monkeypatch.setattr(jobs.MANAGER, "get", lambda job_id: SimpleNamespace(host_action=tag) if job_id == "synthetic-job" else None)
        monkeypatch.setattr(agent_pipeline_runs, "server_reviewer_identity", lambda: "synthetic-host-operator")
        owner = SimpleNamespace(host_action_session=lambda *args, **kwargs: None,
            host_action_child_job=lambda *args, **kwargs: {"child_job_id": "synthetic-job"})
        monkeypatch.setattr(service, "get_pi_copilot_service", lambda: owner)
        checkpoint = record_host_review(host_action=action, pending=checkpoint, decision="approved",
                                        study_context_id="synthetic-study", job_id="synthetic-job",
                                        run_id="synthetic-run", scientific_configuration_sha256="a" * 64)
        host_binding = RawPreparationHostBinding(project_id=action.project_id, session_id=action.session_id,
            study_context_id="synthetic-study", run_id=choice.run_id, nonce=choice.nonce)
    (tmp_path / "sic_raw_preparation_checkpoint.json").write_text(checkpoint.model_dump_json())
    result = SimpleNamespace(context=context, plan=plan, evidence=evidence, host_binding=host_binding)
    return result, source_path, scope, checkpoint

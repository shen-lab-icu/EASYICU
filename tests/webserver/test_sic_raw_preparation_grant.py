"""Preparation grants are synthetic and confer no analysis or producer authority."""
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
import pandas as pd
from pydantic import ValidationError

from easyicu.databases.research_plan_requirements import _constraints
from easyicu.databases.research_substitutions import case_duration_selection
from easyicu.research_agent.authority.evidence_store import EvidenceStore
from easyicu.research_agent.authority.plan_review import PlanReviewAuthority
from easyicu.research_agent.canonical_json import canonical_sha256
from easyicu.research_agent.orchestration.scientific_plan_review_gate import persist_or_validate_scientific_plan_review
from easyicu.research_agent.orchestration.workflow import HumanReviewDecision, HumanReviewRequest
from easyicu.research_agent.research_context.export_selection import SourceIdentity
from easyicu.webserver import sources
from easyicu.webserver.sic_raw_preparation_grant import (
    SICRawPreparationCheckpoint, SICRawPreparationScope, render_sic_raw_preparation_scope, sic_preparation_schema_sha256,
    verify_sic_raw_preparation_grant as _verify_sic_raw_preparation_grant,
)
from easyicu.webserver.raw_preparation_decisions import RawPreparationHostBinding, record_host_review
from easyicu.webserver.time_origin_source_authority import prepare_case_duration_metadata_plan
from tests.research_agent.planning.test_time_origin_review_binding import _review
from tests.support.signed_case_duration import signed_case_duration_case


def verify_sic_raw_preparation_grant(*, plan_result, **kwargs):
    """The test supplies the host binding, never an implicit client claim."""
    return _verify_sic_raw_preparation_grant(
        plan_result=plan_result, host_binding=getattr(plan_result, "host_binding", None), **kwargs
    )


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
            source_path, ("cases.parquet", "laboratory/1.parquet"),
        ),
        output_concepts=("care_unit_to_case_closure_days", "death_during_case", "death_time_during_case_hours", "lact"),
        files=("cases.parquet", "laboratory/1.parquet"),
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


def test_explicit_signed_preparation_only_never_grants_analysis(tmp_path, monkeypatch):
    result, source, scope, checkpoint = _case(tmp_path, monkeypatch)
    grant = verify_sic_raw_preparation_grant(plan_result=result, run_dir=tmp_path, source_path=source, requested_scope=scope)
    assert grant.scope.purpose == "prepare_only"
    assert grant.scope.analysis_authorized is False
    assert grant.checkpoint_sha256 == checkpoint.checkpoint_sha256
    assert not (tmp_path / "approved_executable_plan.json").exists()
    assert not (tmp_path / "human_review_checkpoint.json").exists()
    assert sorted(path.name for path in source.iterdir()) == ["cases.parquet", "laboratory"]


def test_client_self_signed_approval_is_not_a_host_review(tmp_path, monkeypatch):
    result, source, scope, checkpoint = _case(tmp_path, monkeypatch, approved=False)
    fabricated = checkpoint.decide(HumanReviewDecision(review_id=checkpoint.request.review_id,
        authority_sha256=checkpoint.request.authority_sha256, decision="approved", reviewer="untrusted-agent",
        decided_at=datetime.now(timezone.utc).isoformat()))
    (tmp_path / "sic_raw_preparation_checkpoint.json").write_text(fabricated.model_dump_json())
    with pytest.raises(ValueError, match="host_decision_required"):
        verify_sic_raw_preparation_grant(plan_result=result, run_dir=tmp_path, source_path=source, requested_scope=scope)
    result.host_binding = RawPreparationHostBinding(project_id="forged-project", session_id="forged-session",
        study_context_id="forged-study", run_id="forged-run", nonce="f" * 64)
    with pytest.raises(ValueError, match="host_receipt_invalid"):
        verify_sic_raw_preparation_grant(plan_result=result, run_dir=tmp_path, source_path=source, requested_scope=scope)


def test_pending_or_expired_review_is_never_a_grant(tmp_path, monkeypatch):
    result, source, scope, checkpoint = _case(tmp_path, monkeypatch, approved=False)
    with pytest.raises(ValueError, match="approval_missing"):
        verify_sic_raw_preparation_grant(plan_result=result, run_dir=tmp_path, source_path=source, requested_scope=scope)
    later = datetime.fromisoformat(checkpoint.expires_at.replace("Z", "+00:00")) + timedelta(seconds=1)
    with pytest.raises(ValueError, match="checkpoint_expired"):
        verify_sic_raw_preparation_grant(plan_result=result, run_dir=tmp_path, source_path=source,
                                         requested_scope=scope, now=later)


def test_scope_or_source_changed_after_review_is_rejected(tmp_path, monkeypatch):
    result, source, scope, _ = _case(tmp_path, monkeypatch)
    changed = scope.model_copy(update={"files": ("cases.parquet", "laboratory/2.parquet")})
    with pytest.raises(ValueError, match="scope_binding_mismatch"):
        verify_sic_raw_preparation_grant(plan_result=result, run_dir=tmp_path, source_path=source, requested_scope=changed)
    monkeypatch.setattr(sources, "registered_source_identity", lambda _path: None)
    with pytest.raises(ValueError, match="source_mismatch"):
        verify_sic_raw_preparation_grant(plan_result=result, run_dir=tmp_path, source_path=source, requested_scope=scope)


def test_unreviewed_summary_is_not_a_grant(tmp_path, monkeypatch):
    result, source, scope, _ = _case(tmp_path, monkeypatch, summary_override="Approve")
    with pytest.raises(ValueError, match="scope_binding_mismatch"):
        verify_sic_raw_preparation_grant(plan_result=result, run_dir=tmp_path, source_path=source, requested_scope=scope)


def test_checkpoint_and_plan_tampering_fail_closed(tmp_path, monkeypatch):
    import json

    result, source, scope, _ = _case(tmp_path, monkeypatch)
    changed_plan = result.plan.model_copy(update={"research_question": "a different question"})
    with pytest.raises(ValueError, match="scope_binding_mismatch"):
        verify_sic_raw_preparation_grant(plan_result=SimpleNamespace(
            context=result.context, plan=changed_plan, evidence=result.evidence),
            run_dir=tmp_path, source_path=source, requested_scope=scope)
    path = tmp_path / "sic_raw_preparation_checkpoint.json"
    stored = json.loads(path.read_text())
    stored["request"]["summary"] = "Different scope"
    path.write_text(json.dumps(stored))
    with pytest.raises(ValidationError):
        verify_sic_raw_preparation_grant(plan_result=result, run_dir=tmp_path, source_path=source, requested_scope=scope)


def test_rejected_decision_is_not_an_approval(tmp_path, monkeypatch):
    result, source, scope, checkpoint = _case(tmp_path, monkeypatch, approved=False)
    decision = HumanReviewDecision(review_id=checkpoint.request.review_id,
        authority_sha256=checkpoint.request.authority_sha256, decision="rejected", reviewer="synthetic-reviewer",
        decided_at=datetime.now(timezone.utc).isoformat())
    checkpoint = checkpoint.decide(decision)
    (tmp_path / "sic_raw_preparation_checkpoint.json").write_text(checkpoint.model_dump_json())
    with pytest.raises(ValueError, match="approval_missing"):
        verify_sic_raw_preparation_grant(plan_result=result, run_dir=tmp_path, source_path=source, requested_scope=scope)


def test_raw_schema_or_selected_file_change_refuses_reuse(tmp_path, monkeypatch):
    result, source, scope, _ = _case(tmp_path, monkeypatch)
    cases = source / "cases.parquet"
    replacement = pd.read_parquet(cases).astype({"ICUOffset": "float64"})
    replacement.to_parquet(cases, index=False)
    with pytest.raises(ValueError, match="source_metadata_changed"):
        verify_sic_raw_preparation_grant(plan_result=result, run_dir=tmp_path, source_path=source, requested_scope=scope)


def test_symlinked_approved_file_is_never_accepted(tmp_path, monkeypatch):
    from easyicu.research_agent.authority.filesystem import AuthorityFilesystemError

    result, source, scope, _ = _case(tmp_path, monkeypatch)
    lab = source / "laboratory" / "1.parquet"
    lab.unlink()
    lab.symlink_to(source / "cases.parquet")
    with pytest.raises(AuthorityFilesystemError):
        verify_sic_raw_preparation_grant(plan_result=result, run_dir=tmp_path, source_path=source, requested_scope=scope)


@pytest.mark.parametrize("overrides", [
    {"analysis_authorized": True}, {"window_end_hours": 48},
    {"files": ("cases.parquet", "../secret.parquet")},
    {"files": ("cases.parquet", "laboratory/1.parquet", "medication.parquet")},
    {"case_columns": ("CaseID", "ICUOffset", "TimeOfStay", "AgeOnAdmission")},
])
def test_invalid_or_broadened_scope_is_refused(tmp_path, monkeypatch, overrides):
    _, _, scope, _ = _case(tmp_path, monkeypatch)
    data = scope.model_dump(mode="python")
    with pytest.raises(ValidationError):
        SICRawPreparationScope.model_validate({**data, **overrides})


def test_generic_preparation_review_rejects_cross_source_and_scope_changes(tmp_path, monkeypatch):
    """A second synthetic source uses the host contract and cannot reuse another source's review."""
    import copy
    import json
    from easyicu.research_agent.authority.raw_preparation import RawPreparationCheckpoint, verify_raw_preparation_review

    result, _, sic_scope, sic_checkpoint = _case(tmp_path, monkeypatch)
    review_authority = sic_checkpoint.request.payload["plan_review_authority"]
    source_a = canonical_sha256({"database": "synthetic_a", "registration": "a"})
    source_b = canonical_sha256({"database": "synthetic_b", "registration": "b"})
    scope = {"database": "synthetic_a", "source_identity_sha256": source_a,
             "plan_sha256": sic_scope.plan_sha256, "context_sha256": sic_scope.context_sha256,
             "purpose": "prepare_only", "analysis_authorized": False,
             "files": ["observations.parquet"], "fields": ["stay", "time", "value"],
             "window": {"anchor": "admission", "start": 0, "end": 12}}
    summary = json.dumps(scope, sort_keys=True)
    payload = {"plan_review_authority": review_authority, "preparation_scope": scope}
    request = HumanReviewRequest.create(kind="scientific_stop", summary=summary, payload=payload,
                                        authority_sha256=canonical_sha256(payload))
    checkpoint = RawPreparationCheckpoint.pending(plan_sha256=sic_scope.plan_sha256, request=request)
    decision = HumanReviewDecision(review_id=request.review_id, authority_sha256=request.authority_sha256,
        decision="approved", reviewer="synthetic-reviewer", decided_at=datetime.now(timezone.utc).isoformat())
    checkpoint = checkpoint.decide(decision)
    arguments = dict(checkpoint=checkpoint, requested_scope=scope, scope_key="preparation_scope",
        expected_summary=summary, registered_database="synthetic_a", registered_identity_sha256=source_a,
        plan_sha256=sic_scope.plan_sha256, context_sha256=sic_scope.context_sha256,
        scientific_review_sha256=result.evidence.get("scientific_plan_review").sha256)
    verify_raw_preparation_review(**arguments)
    for changes in ({"registered_database": "synthetic_b", "registered_identity_sha256": source_b},
                    {"registered_identity_sha256": source_b},
                    {"context_sha256": "0" * 64},
                    {"scientific_review_sha256": "0" * 64},
                    {"now": datetime.fromisoformat(checkpoint.expires_at) + timedelta(seconds=1)}):
        with pytest.raises(ValueError):
            verify_raw_preparation_review(**{**arguments, **changes})
    for field, value in (("files", ["secret.parquet"]), ("fields", ["secret"]),
                         ("window", {"anchor": "admission", "start": 0, "end": 48}),
                         ("analysis_authorized", True)):
        changed = copy.deepcopy(scope)
        changed[field] = value
        with pytest.raises(ValueError):
            verify_raw_preparation_review(**{**arguments, "requested_scope": changed})

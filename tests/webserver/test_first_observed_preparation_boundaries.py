"""The unregistered first-observed producer must not read rows before review."""
from pathlib import Path
from types import SimpleNamespace
import json

import pandas as pd
import pytest

from easyicu.research_agent.authority.plan_review import PlanReviewAuthority
from easyicu.research_agent.authority.raw_preparation import RawPreparationCheckpoint
from easyicu.research_agent.canonical_json import canonical_sha256
from easyicu.research_agent.orchestration.workflow import HumanReviewRequest
from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep
from easyicu.webserver import first_observed_preparation as owner
from easyicu.webserver.raw_preparation_decisions import RawPreparationHostBinding
from easyicu.webserver.host_action_contracts import (
    HostActionRequest, HostActionTag, RawPreparationReviewDecision, decision_key, host_action_id,
)
from easyicu.webserver.raw_preparation_decisions import record_host_review
from easyicu.research_agent.acquisition.eicu_first_observed_candidate import FLAG_COLUMN, REASON_COLUMN
from easyicu.research_agent.acquisition.first_observed_icu_coordinate import restrict_to_first_observed_icu


def _pending():
    plan = AnalysisPlan(research_question="Synthetic source only", steps=[AnalysisStep(step_id="one", intent="Describe data")])
    authority = PlanReviewAuthority.create(plan=plan, evidence_sha256={"scientific_plan_review": "c" * 64})
    scope = {"purpose": "prepare_only", "analysis_authorized": False, "database": "eicu_demo",
             "source_identity_sha256": "a" * 64, "plan_sha256": authority.plan_sha256,
             "context_sha256": "b" * 64, "scientific_review_sha256": "c" * 64,
             "archive_sha256": "d" * 64, "patient_table_sha256": "f" * 64,
             "export_manifest_sha256": "1" * 64}
    payload = {"plan_review_authority": authority.model_dump(mode="json"), owner._SCOPE_KEY: scope}
    request = HumanReviewRequest.create(kind="scientific_stop", summary=owner._SUMMARY,
        payload=payload, authority_sha256=canonical_sha256(payload))
    checkpoint = RawPreparationCheckpoint.pending(plan_sha256=authority.plan_sha256, request=request)
    binding = RawPreparationHostBinding(project_id="project", session_id="session", study_context_id="study",
                                        run_id="run", nonce="e" * 64)
    return scope, checkpoint, binding


def test_pending_review_never_reads_source_rows(tmp_path, monkeypatch):
    scope, checkpoint, binding = _pending()
    monkeypatch.setattr(owner, "_host_inputs", lambda **_: (scope, tmp_path))
    def forbidden(*args, **kwargs):
        raise AssertionError("patient data read before host approval")
    monkeypatch.setattr(owner, "derive_from_verified_eicu_demo", forbidden)
    with pytest.raises(ValueError, match="approval_missing"):
        owner.prepare_first_observed_coordinate(run_dir=tmp_path, checkpoint=checkpoint, binding=binding, scope=scope)
    assert list(tmp_path.iterdir()) == []


def test_changed_scope_refuses_before_review_or_data(tmp_path, monkeypatch):
    scope, checkpoint, binding = _pending()
    monkeypatch.setattr(owner, "_host_inputs", lambda **_: (scope, Path(tmp_path)))
    def forbidden(*args, **kwargs):
        raise AssertionError("a mismatched scope must not be reviewed or read")
    monkeypatch.setattr(owner, "verify_raw_preparation_review", forbidden)
    monkeypatch.setattr(owner, "derive_from_verified_eicu_demo", forbidden)
    with pytest.raises(ValueError, match="scope_mismatch"):
        owner.prepare_first_observed_coordinate(run_dir=tmp_path, checkpoint=checkpoint, binding=binding,
                                                scope={**scope, "analysis_authorized": True})


def _approved(tmp_path, monkeypatch):
    from easyicu.webserver import jobs, agent_pipeline_runs
    from easyicu.webserver.pi_copilot import service

    monkeypatch.setenv("EASYICU_HOME", str(tmp_path / "host"))
    scope, pending, binding = _pending()
    choice = RawPreparationReviewDecision(
        run_id=binding.run_id, scientific_configuration_sha256="2" * 64,
        checkpoint_sha256=pending.checkpoint_sha256, scope_sha256=canonical_sha256(scope),
        review_authority_sha256=pending.request.authority_sha256, nonce=binding.nonce,
    )
    action = HostActionRequest(session_id=binding.session_id, project_id=binding.project_id,
                               action_code="review_raw_preparation", decision=choice)
    key = decision_key(choice, review_decision="approved")
    tag = HostActionTag(session_id=binding.session_id, project_id=binding.project_id,
        study_context_id=binding.study_context_id, action_code=action.action_code,
        decision_key=key, action_id=host_action_id(binding.session_id, action.action_code, key))
    monkeypatch.setattr(jobs.MANAGER, "get", lambda job_id: SimpleNamespace(host_action=tag) if job_id == "job" else None)
    monkeypatch.setattr(agent_pipeline_runs, "server_reviewer_identity", lambda: "synthetic-host")
    monkeypatch.setattr(service, "get_pi_copilot_service", lambda: SimpleNamespace(
        host_action_session=lambda *args, **kwargs: None,
        host_action_child_job=lambda *args, **kwargs: {"child_job_id": "job"}))
    checkpoint = record_host_review(host_action=action, pending=pending, decision="approved",
        study_context_id=binding.study_context_id, job_id="job", run_id=binding.run_id,
        scientific_configuration_sha256="2" * 64)
    monkeypatch.setattr(owner, "_host_inputs", lambda **_: (scope, tmp_path))
    return scope, checkpoint, binding


def test_synthetic_host_signed_grant_binds_coordinate_and_rejects_tampering(tmp_path, monkeypatch):
    scope, checkpoint, binding = _approved(tmp_path, monkeypatch)
    proof = {"official_release_independently_verified": True,
             "patient_table_sha256": scope["patient_table_sha256"],
             "export_manifest_sha256": scope["export_manifest_sha256"],
             "complete_patient_source": {"schema_version": "easyicu.eicu_demo_complete_patient_source/1",
                                         "all_source_rows_compared": True,
                                         "archive_sha256": scope["archive_sha256"]}}
    # Source proof is deliberately mocked: these rows are entirely synthetic.
    candidate = SimpleNamespace(frame=pd.DataFrame({"stay_id": [1, 2], FLAG_COLUMN: [True, False],
        REASON_COLUMN: ["kept", "multiple_observed_hospitalizations"]}), receipt={
            "source_binding": proof,
            "schema_version": owner.SCHEMA, "source_global_table_required": True,
            "not_lifetime_first_icu": True, "outcome_columns_read": False,
            "receipt_contains_identifier_values": False, "research_execution_authorized": False,
            "source_stays": 2, "source_patients": 2, "eligible_patients": 1,
            "excluded_patients_by_reason": {"multiple_observed_hospitalizations": 1},
            "excluded_stays_by_reason": {"multiple_observed_hospitalizations": 1},
        })
    calls = []
    monkeypatch.setattr(owner, "derive_from_verified_eicu_demo", lambda _: calls.append(1) or candidate)
    coordinate = owner.prepare_first_observed_coordinate(run_dir=tmp_path, checkpoint=checkpoint,
                                                        binding=binding, scope=scope)
    assert calls == [1]
    resolved = owner.resolve_prepared_first_observed_coordinate(run_dir=tmp_path, checkpoint=checkpoint,
                                                               binding=binding, scope=scope)
    assert resolved == coordinate
    flow = owner.resolve_private_first_observed_source_flow(run_dir=tmp_path, checkpoint=checkpoint,
                                                            binding=binding, scope=scope)
    assert flow == {"schema_version": "easyicu.first-observed-private-source-flow/1",
                    "scope": "complete_source_before_other_study_filters", "not_lifetime_first_icu": True,
                    "source_stays": 2, "source_patients": 2, "eligible_patients": 1,
                    "excluded_patients_by_reason": {"multiple_observed_hospitalizations": 1},
                    "excluded_stays_by_reason": {"multiple_observed_hospitalizations": 1},
                    "public_projection_authorized": False}
    with pytest.raises(ValueError, match="binding_mismatch"):
        owner.resolve_private_first_observed_source_flow(
            run_dir=tmp_path, checkpoint=checkpoint,
            binding=binding.model_copy(update={"session_id": "other-session"}), scope=scope)
    with pytest.raises(ValueError, match="scope_mismatch"):
        owner.resolve_private_first_observed_source_flow(
            run_dir=tmp_path, checkpoint=checkpoint, binding=binding,
            scope={**scope, "archive_sha256": "0" * 64})
    result = restrict_to_first_observed_icu(pd.DataFrame({"stay_id": [1, 2]}), coordinate=resolved)
    assert result["stay_id"].tolist() == [1]
    with pytest.raises(ValueError, match="nonce_replayed"):
        owner.prepare_first_observed_coordinate(run_dir=tmp_path, checkpoint=checkpoint, binding=binding, scope=scope)
    assert calls == [1]
    attestation = coordinate.path.with_suffix(".json")
    original = attestation.read_text()
    document = json.loads(original)
    legacy_body = {**document["body"], "schema_version": "easyicu.first-observed-coordinate-attestation/1"}
    legacy_body.pop("private_source_flow")
    attestation.write_text(json.dumps({"body": legacy_body, "signature": owner._signature(legacy_body)}))
    assert owner.resolve_prepared_first_observed_coordinate(run_dir=tmp_path, checkpoint=checkpoint,
                                                            binding=binding, scope=scope) == coordinate
    with pytest.raises(ValueError, match="source_flow_unavailable"):
        owner.resolve_private_first_observed_source_flow(run_dir=tmp_path, checkpoint=checkpoint,
                                                         binding=binding, scope=scope)
    attestation.write_text(original)
    document["body"]["private_source_flow"]["eligible_patients"] = 2
    attestation.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="attestation_invalid"):
        owner.resolve_private_first_observed_source_flow(run_dir=tmp_path, checkpoint=checkpoint,
                                                         binding=binding, scope=scope)
    attestation.write_text(original)
    coordinate.path.write_bytes(coordinate.path.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="attestation_binding_mismatch"):
        owner.resolve_prepared_first_observed_coordinate(run_dir=tmp_path, checkpoint=checkpoint,
                                                         binding=binding, scope=scope)


@pytest.mark.parametrize("changed", ["source", "plan", "export_path", "grant"])
def test_changed_host_binding_after_read_never_publishes_coordinate(tmp_path, monkeypatch, changed):
    scope, checkpoint, binding = _approved(tmp_path, monkeypatch)
    proof = {"official_release_independently_verified": True,
             "patient_table_sha256": scope["patient_table_sha256"],
             "export_manifest_sha256": scope["export_manifest_sha256"],
             "complete_patient_source": {"schema_version": "easyicu.eicu_demo_complete_patient_source/1",
                                         "all_source_rows_compared": True,
                                         "archive_sha256": scope["archive_sha256"]}}
    candidate = SimpleNamespace(frame=pd.DataFrame({"stay_id": [1], FLAG_COLUMN: [True],
        REASON_COLUMN: ["kept"]}), receipt={"source_binding": proof,
        "schema_version": owner.SCHEMA, "source_global_table_required": True,
        "not_lifetime_first_icu": True, "outcome_columns_read": False,
        "receipt_contains_identifier_values": False, "research_execution_authorized": False,
        "source_stays": 1, "source_patients": 1, "eligible_patients": 1,
        "excluded_patients_by_reason": {}, "excluded_stays_by_reason": {}})
    second_scope = dict(scope)
    if changed == "source":
        second_scope["patient_table_sha256"] = "0" * 64
    if changed == "plan":
        second_scope["plan_sha256"] = "0" * 64
    values = iter([(scope, tmp_path), (second_scope, tmp_path / "other" if changed == "export_path" else tmp_path)])
    monkeypatch.setattr(owner, "_host_inputs", lambda **_: next(values))
    monkeypatch.setattr(owner, "derive_from_verified_eicu_demo", lambda _: candidate)
    if changed == "grant":
        def invalid_grant(**_):
            raise ValueError("raw_preparation_host_receipt_invalid")
        monkeypatch.setattr(owner, "verify_consumed_host_review", invalid_grant)
    def forbidden_publication(**_):
        raise AssertionError("changed host binding must not publish")
    monkeypatch.setattr(owner, "_private_root", forbidden_publication)
    expected = "host_receipt_invalid" if changed == "grant" else "scope_changed_before_publication"
    with pytest.raises(ValueError, match=expected):
        owner.prepare_first_observed_coordinate(run_dir=tmp_path, checkpoint=checkpoint, binding=binding, scope=scope)


def test_source_flow_rejects_invalid_or_sensitive_content():
    valid = {"schema_version": owner.SCHEMA, "source_global_table_required": True,
             "not_lifetime_first_icu": True, "outcome_columns_read": False,
             "receipt_contains_identifier_values": False, "research_execution_authorized": False,
             "source_stays": 2, "source_patients": 2, "eligible_patients": 1,
             "excluded_patients_by_reason": {"multiple_observed_hospitalizations": 1},
             "excluded_stays_by_reason": {"multiple_observed_hospitalizations": 1}}
    frame = pd.DataFrame({"stay_id": [1, 2], FLAG_COLUMN: [True, False],
                          REASON_COLUMN: ["kept", "multiple_observed_hospitalizations"]})
    for changes in ({"eligible_patients": 2}, {"source_patients": True},
                    {"excluded_patients_by_reason": {"patient-123": 1}},
                    {"excluded_stays_by_reason": {"under_18": 1}},
                    {"outcome_columns_read": True}):
        with pytest.raises(ValueError, match="source_flow_invalid"):
            owner._private_source_flow(SimpleNamespace(frame=frame, receipt={**valid, **changes}))
    repeated = frame.copy()
    repeated["stay_id"] = [1, 1]
    with pytest.raises(ValueError, match="source_flow_invalid"):
        owner._private_source_flow(SimpleNamespace(frame=repeated, receipt=valid))


def test_crashed_source_read_cannot_replay_the_consumed_grant(tmp_path, monkeypatch):
    scope, checkpoint, binding = _approved(tmp_path, monkeypatch)
    calls = []
    def crash(_):
        calls.append(1)
        raise RuntimeError("synthetic producer failure")
    monkeypatch.setattr(owner, "derive_from_verified_eicu_demo", crash)
    with pytest.raises(RuntimeError, match="synthetic producer failure"):
        owner.prepare_first_observed_coordinate(run_dir=tmp_path, checkpoint=checkpoint, binding=binding, scope=scope)
    with pytest.raises(ValueError, match="nonce_replayed"):
        owner.prepare_first_observed_coordinate(run_dir=tmp_path, checkpoint=checkpoint, binding=binding, scope=scope)
    assert calls == [1]


def test_source_completeness_mismatch_never_produces_an_attestation(tmp_path, monkeypatch):
    scope, checkpoint, binding = _approved(tmp_path, monkeypatch)
    candidate = SimpleNamespace(frame=pd.DataFrame({"stay_id": [1], FLAG_COLUMN: [True], REASON_COLUMN: ["kept"]}),
        receipt={"source_binding": {"official_release_independently_verified": True,
            "patient_table_sha256": scope["patient_table_sha256"],
            "export_manifest_sha256": scope["export_manifest_sha256"],
            "complete_patient_source": {"schema_version": "easyicu.eicu_demo_complete_patient_source/1",
                "all_source_rows_compared": False, "archive_sha256": scope["archive_sha256"]}}})
    monkeypatch.setattr(owner, "derive_from_verified_eicu_demo", lambda _: candidate)
    with pytest.raises(ValueError, match="source_proof_mismatch"):
        owner.prepare_first_observed_coordinate(run_dir=tmp_path, checkpoint=checkpoint, binding=binding, scope=scope)
    assert not (tmp_path / "host" / ".easyicu" / "private-first-observed").exists()

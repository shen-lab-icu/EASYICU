"""A consumed first-observed grant is never replayed against raw data."""
from types import SimpleNamespace

import pandas as pd
import pytest

from easyicu.research_agent.authority.raw_preparation import RawPreparationCheckpoint
from easyicu.research_agent.canonical_json import canonical_sha256
from easyicu.research_agent.orchestration.workflow import HumanReviewRequest
from easyicu.research_agent.acquisition.eicu_first_observed_candidate import FLAG_COLUMN, REASON_COLUMN
from easyicu.webserver import agent_pipeline_runs, first_observed_preparation as producer, jobs, raw_preparation_durable as durable, study_contexts
from easyicu.webserver.host_action_contracts import HostActionRequest, HostActionTag, RawPreparationReviewDecision, decision_key, host_action_id
from easyicu.webserver.pi_copilot import service
from easyicu.webserver.raw_preparation_decisions import record_host_review
from tests.webserver.test_first_observed_preparation_boundaries import _pending


def _case(tmp_path, monkeypatch):
    monkeypatch.setenv("EASYICU_HOME", str(tmp_path / "host"))
    scope, old, binding = _pending()
    scope["adapter_id"] = producer._ADAPTER
    payload = {"plan_review_authority": old.request.payload["plan_review_authority"], producer._SCOPE_KEY: scope}
    request = HumanReviewRequest.create(kind="scientific_stop", summary=producer._SUMMARY,
        payload=payload, authority_sha256=canonical_sha256(payload))
    pending = RawPreparationCheckpoint.pending(plan_sha256=scope["plan_sha256"], request=request)
    choice = RawPreparationReviewDecision(run_id=binding.run_id, scientific_configuration_sha256="2" * 64,
        checkpoint_sha256=pending.checkpoint_sha256, scope_sha256=canonical_sha256(scope),
        review_authority_sha256=request.authority_sha256, nonce=binding.nonce)
    action = HostActionRequest(session_id=binding.session_id, project_id=binding.project_id,
                               action_code="review_raw_preparation", decision=choice)
    key = decision_key(choice, review_decision="approved")
    tag = HostActionTag(session_id=binding.session_id, project_id=binding.project_id,
        study_context_id=binding.study_context_id, action_code=action.action_code, decision_key=key,
        action_id=host_action_id(binding.session_id, action.action_code, key))
    monkeypatch.setattr(jobs.MANAGER, "get", lambda job_id: SimpleNamespace(host_action=tag) if job_id == "job" else None)
    monkeypatch.setattr(agent_pipeline_runs, "server_reviewer_identity", lambda: "synthetic-host")
    monkeypatch.setattr(service, "get_pi_copilot_service", lambda: SimpleNamespace(
        host_action_session=lambda *a, **k: None, host_action_child_job=lambda *a, **k: {"child_job_id": "job"}))
    monkeypatch.setattr(agent_pipeline_runs, "raw_preparation_review_owner", lambda *_: (
        SimpleNamespace(run_dir=str(tmp_path), requests=[request]), {"id": "study"}))
    monkeypatch.setattr(study_contexts, "get_context", lambda _: {"id": "study"})
    monkeypatch.setattr(study_contexts, "scientific_configuration_sha256", lambda _: "2" * 64)
    checkpoint = record_host_review(host_action=action, pending=pending, decision="approved", study_context_id="study",
                                    job_id="job", run_id="run", scientific_configuration_sha256="2" * 64)
    monkeypatch.setattr(producer, "_host_inputs", lambda **_: (scope, tmp_path))
    job = SimpleNamespace(id="job", is_cancel_requested=lambda: False, tag_host_action=lambda _: True,
                          register_cancel_callback=lambda _: lambda: None)
    return scope, checkpoint, binding, job


def test_consumed_completed_coordinate_recovers_without_a_second_source_read(tmp_path, monkeypatch):
    scope, checkpoint, binding, job = _case(tmp_path, monkeypatch)
    proof = {"official_release_independently_verified": True, "patient_table_sha256": scope["patient_table_sha256"],
             "export_manifest_sha256": scope["export_manifest_sha256"], "complete_patient_source": {
                 "schema_version": "easyicu.eicu_demo_complete_patient_source/1", "all_source_rows_compared": True,
                 "archive_sha256": scope["archive_sha256"]}}
    candidate = SimpleNamespace(frame=pd.DataFrame({"stay_id": [1], FLAG_COLUMN: [True], REASON_COLUMN: ["kept"]}),
        receipt={"source_binding": proof, "schema_version": producer.SCHEMA, "source_global_table_required": True,
        "not_lifetime_first_icu": True, "outcome_columns_read": False, "receipt_contains_identifier_values": False,
        "research_execution_authorized": False, "source_stays": 1, "source_patients": 1, "eligible_patients": 1,
        "excluded_patients_by_reason": {}, "excluded_stays_by_reason": {}})
    reads = []
    monkeypatch.setattr(producer, "derive_from_verified_eicu_demo", lambda _: reads.append(1) or candidate)
    monkeypatch.setattr(durable, "assert_host_execution_quiescent", lambda *a, **k: None)
    def worker(**kwargs):
        return producer.prepare_first_observed_coordinate(run_dir=kwargs["host_work_root"], checkpoint=checkpoint,
                                                         binding=binding, scope=scope).path
    monkeypatch.setattr(durable, "guarded_multitable_projection", worker)
    durable.admit_preparation_intent(job_id="job", checkpoint=checkpoint, binding=binding, scope=scope)
    output = durable.recover_preparation_with_sealed_snapshot(job_id="job", job=job, timeout=5)
    assert reads == [1] and output.is_file()
    assert durable.recover_preparation_with_sealed_snapshot(job_id="job", job=job, timeout=5) == output
    assert reads == [1]
    with monkeypatch.context() as changed:
        changed.setattr(study_contexts, "scientific_configuration_sha256", lambda _: "0" * 64)
        with pytest.raises(ValueError, match="configuration_changed"):
            durable.recover_preparation_with_sealed_snapshot(job_id="job", job=job, timeout=5)
    with monkeypatch.context() as changed:
        def unsafe_cleanup(*args, **kwargs):
            raise ValueError("cleanup_unproven")
        changed.setattr(durable, "assert_host_execution_quiescent", unsafe_cleanup)
        with pytest.raises(ValueError, match="cleanup_unproven"):
            durable.recover_preparation_with_sealed_snapshot(job_id="job", job=job, timeout=5)
    assert reads == [1]
    output.with_suffix(".json").unlink()
    with pytest.raises((OSError, ValueError, RuntimeError)):
        durable.recover_preparation_with_sealed_snapshot(job_id="job", job=job, timeout=5)
    assert reads == [1]


def test_changed_scope_does_not_create_intent(tmp_path, monkeypatch):
    scope, checkpoint, binding, _ = _case(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="scope_mismatch"):
        durable.admit_preparation_intent(job_id="job", checkpoint=checkpoint, binding=binding,
                                        scope={**scope, "analysis_authorized": True})


def test_real_guarded_worker_refuses_unapproved_first_observed_request(tmp_path, monkeypatch):
    from easyicu.runtime.host_execution_lease import assert_host_execution_quiescent
    from easyicu.webserver.multitable_guarded_worker import guarded_multitable_projection

    monkeypatch.setenv("EASYICU_HOME", str(tmp_path / "host"))
    scope, pending, binding = _pending()
    scope["adapter_id"] = producer._ADAPTER
    with pytest.raises(ValueError, match="guarded_preparation_failed"):
        guarded_multitable_projection(lease_dir=tmp_path / "lease", checkpoint=pending, binding=binding,
            scope=scope, host_work_root=tmp_path, timeout=15)
    assert_host_execution_quiescent(tmp_path / "lease", allow_missing=False, wait_timeout=5)
    assert not (tmp_path / "host" / "private-first-observed").exists()

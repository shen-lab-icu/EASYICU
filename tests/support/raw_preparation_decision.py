"""Synthetic host-bound raw-preparation decision fixture."""
from types import SimpleNamespace

from easyicu.research_agent.authority.raw_preparation import RawPreparationCheckpoint
from easyicu.research_agent.canonical_json import canonical_sha256
from easyicu.research_agent.orchestration.workflow import HumanReviewRequest
from easyicu.webserver.host_action_contracts import (
    HostActionRequest, HostActionTag, RawPreparationReviewDecision, decision_key, host_action_id,
)


def _host_decision(tmp_path, monkeypatch):
    from easyicu.webserver import jobs, agent_pipeline_runs
    from easyicu.webserver.pi_copilot import service

    monkeypatch.setenv("EASYICU_HOME", str(tmp_path / "host"))
    # Deliberately non-SIC: the authority envelope has no database-specific logic.
    scope = {"source": "synthetic-other-database", "files": ["observations.csv"], "purpose": "prepare_only"}
    payload = {"synthetic_preparation_scope": scope}
    request = HumanReviewRequest.create(kind="scientific_stop", summary="Review source and files",
                                        payload=payload, authority_sha256=canonical_sha256(payload))
    pending = RawPreparationCheckpoint.pending(plan_sha256="c" * 64, request=request)
    choice = RawPreparationReviewDecision(run_id="run", scientific_configuration_sha256="a" * 64,
        checkpoint_sha256=pending.checkpoint_sha256, scope_sha256=canonical_sha256(scope),
        review_authority_sha256=request.authority_sha256, nonce="b" * 64)
    action = HostActionRequest(session_id="session", project_id="project",
                               action_code="review_raw_preparation", decision=choice)
    key = decision_key(choice, review_decision="approved")
    tag = HostActionTag(session_id="session", project_id="project", study_context_id="study",
        action_code=action.action_code, decision_key=key, action_id=host_action_id("session", action.action_code, key))
    monkeypatch.setattr(jobs.MANAGER, "get", lambda job_id: SimpleNamespace(host_action=tag) if job_id == "job" else None)
    monkeypatch.setattr(agent_pipeline_runs, "server_reviewer_identity", lambda: "actual-host-operator")
    owner = SimpleNamespace(host_action_session=lambda *args, **kwargs: None,
                            host_action_child_job=lambda *args, **kwargs: {"child_job_id": "job"})
    monkeypatch.setattr(service, "get_pi_copilot_service", lambda: owner)
    record_args = dict(host_action=action, pending=pending, decision="approved", study_context_id="study",
                       job_id="job", run_id="run", scientific_configuration_sha256="a" * 64)
    binding = dict(project_id="project", session_id="session", study_context_id="study",
                   run_id="run", nonce=choice.nonce, scope=scope)
    return pending, record_args, binding

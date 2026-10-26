"""Only the typed not-started runner fact crosses into provider stop copy."""

import json
from types import SimpleNamespace

import pytest

from easyicu.webserver import agent_pipeline_runs, stop_next_actions
from easyicu.webserver.execution_failure import FailureLocation
from easyicu.webserver.pi_copilot.projections import gate_detail_projection, project_run_outcome
from easyicu.webserver.pi_copilot.workflow import project_run_history


@pytest.mark.parametrize("kind", ["http", "timeout"])
def test_provider_stop_preserves_not_started_from_receipt_to_conversation(tmp_path, kind):
    error = RuntimeError("private synthetic provider body") if kind == "http" else TimeoutError("private")
    if kind == "http":
        error.response = SimpleNamespace(status_code=503)
    error.easyicu_failure_location = FailureLocation(phase="plan", runner_kind="host")
    agent_pipeline_runs._record_pipeline_failure(
        wrapper_dir=tmp_path, study={"id": "synthetic"}, provider={}, exc=error,
        code=agent_pipeline_runs._pipeline_failure_code(error), execution_retry_id=None,
    )
    gate = json.loads((tmp_path / "quality_gate.json").read_text())["gate"]
    assert gate["detail"]["runner_kind"] == "not_started"
    assert gate_detail_projection(gate["detail"])["gate_detail_runner_kind"] == "not_started"
    assert project_run_outcome({"ok": True, "run_id": "synthetic", "gate": gate})["gate_detail_runner_kind"] == "not_started"
    [history] = project_run_history([{"run_id": "synthetic", "run_status": "failed",
                                     "gate_reason": gate["reason"], "gate_detail": gate["detail"]}])
    assert history["gate_detail_runner_kind"] == "not_started"
    assert "private" not in json.dumps(gate)


@pytest.mark.parametrize("runner", [None, {}, True, "host", "unknown", "private-path", "not_started "])
def test_other_runner_values_do_not_claim_analysis_has_not_started(runner):
    detail = {"reason_code": "provider_http_error", "runner_kind": runner}
    assert "runner_kind" not in stop_next_actions.safe_stop_coordinates(detail)
    assert "gate_detail_runner_kind" not in gate_detail_projection(detail)


@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("location,coordinates", [
    (None, {}),
    (FailureLocation(), {}),
    (FailureLocation(phase="planning", step_id="plan", runner_kind="host"),
     {"phase": "plan", "step_id": "plan", "runner_kind": "not_started"}),
    (FailureLocation(phase="execute", step_id="synthetic_fit", runner_kind="docker"),
     {"phase": "execute", "step_id": "synthetic_fit", "runner_kind": "docker"}),
    (FailureLocation(phase="execute", step_id="private/step", runner_kind="private-runner"),
     {"phase": "execute"}),
    (FailureLocation(phase="private-phase", step_id="private/step", runner_kind="private-runner"), {}),
])
def test_typed_timeout_keeps_only_observed_bounded_coordinates(tmp_path, location, coordinates, wrapped):
    from easyicu.research_agent.providers.codex_app_server import CodexAppServerError

    timeout = CodexAppServerError("codex_auth_notification_hard_timeout", "private synthetic detail")
    if location is not None:
        timeout.easyicu_failure_location = location
    error = RuntimeError("private wrapper") if wrapped else timeout
    if wrapped:
        error.__cause__ = timeout
    relative = agent_pipeline_runs._write_pipeline_failure_diagnostic(
        wrapper_dir=tmp_path, exc=error, code=agent_pipeline_runs._pipeline_failure_code(error),
    )
    receipt = json.loads((tmp_path / relative).read_text())
    assert receipt["typed_failure"] == {
        "owner": "easyicu.providers.codex_app_server_v1",
        "reason_code": "codex_auth_notification_hard_timeout",
        **coordinates,
    }
    assert "private" not in json.dumps(receipt)

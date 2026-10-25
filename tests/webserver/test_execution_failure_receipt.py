"""Terminal failures keep bounded location and cause without exception text."""

import json
from pathlib import Path
import shutil

import pytest

from easyicu.webserver import agent_pipeline_runs as web
from tests.support.node import run_node


def _record(tmp_path, exc):
    path = web._record_pipeline_failure(wrapper_dir=tmp_path, study={"id": "synthetic"}, provider={},
                                       exc=exc, code="research_pipeline_execution_failed", execution_retry_id=None)
    return json.loads((tmp_path / path).read_text())


@pytest.mark.parametrize("exception", [ValueError, KeyError, RuntimeError, type("PrivatePatientValue", (Exception,), {})])
def test_even_an_unlocated_execution_failure_has_a_closed_receipt(tmp_path, exception):
    error = exception("synthetic private message")
    error.easyicu_safe_diagnostic = {"owner": "private_owner", "step_id": "private_step", "reason_code": "private_reason"}
    record = _record(tmp_path, error)
    typed = record["typed_failure"]
    assert typed["owner"] == "easyicu.execution.pipeline_v1"
    assert typed["reason_code"] == "pipeline_operation_raised"
    assert typed["step_id"] == "unknown"
    assert typed["phase"] == "unknown"
    assert typed["runner_kind"] == "unknown"
    assert typed["exception_type"] == (exception.__name__ if exception.__module__ == "builtins" else "UnclassifiedError")
    assert typed["exception_type"] in record["exception_types"]
    assert "private" not in json.dumps(record).lower()
    assert "PrivatePatientValue" not in json.dumps(record)


@pytest.mark.parametrize("chain", ["direct", "outer_builtin", "inner_builtin"])
def test_a_recorded_owner_exception_never_acquires_a_fallback_type(tmp_path, chain):
    from easyicu.research_agent.authority.run_input import RunInputIdentityError

    error = RunInputIdentityError("synthetic private identity")
    if chain == "outer_builtin":
        outer = RuntimeError("synthetic private wrapper")
        outer.__cause__ = error
        error = outer
    elif chain == "inner_builtin":
        error.__cause__ = ValueError("synthetic private value")
    record = _record(tmp_path, error)
    assert record["exception_types"] == ["RunInputIdentityError"]
    assert "private" not in json.dumps(record).lower()


@pytest.mark.parametrize("phase,step,runner", [("execute", "adjusted_association", "host"), ("execute", "absolute_risk_context", "docker"), ("planning", None, "host")])
def test_the_pipeline_boundary_preserves_the_active_operation(tmp_path, phase, step, runner):
    from easyicu.webserver.execution_failure import call_with_failure_location

    events = []
    def operation(*, progress_callback):
        progress_callback({"phase": phase, "step_id": step, "message": "synthetic private progress", "status": "running"})
        raise KeyError("synthetic private column")
    with pytest.raises(KeyError) as caught:
        call_with_failure_location(operation, runner_kind=runner, progress_callback=events.append)
    assert len(events) == 1
    outer = RuntimeError("synthetic private wrapper")
    outer.__cause__ = caught.value
    record = _record(tmp_path, outer)
    diagnostic = record["typed_failure"]
    assert diagnostic["phase"] == ("plan" if phase == "planning" else phase)
    assert diagnostic["step_id"] == (step or "plan")
    assert diagnostic["runner_kind"] == (runner if phase == "execute" else "not_started")
    assert diagnostic["exception_type"] == "KeyError"
    assert diagnostic["reason_code"] == ("execution_step_raised" if step else "pipeline_operation_raised")
    assert "private" not in json.dumps(record).lower()
    gate = json.loads((tmp_path / "quality_gate.json").read_text())["gate"]
    assert gate["detail"]["step_id"] == (step or "plan")
    assert json.loads((tmp_path / "source_run_manifest.json").read_text())["analysis_started"] is (phase == "execute")


def test_a_specific_owner_reason_wins_over_the_boundary_fallback(tmp_path):
    from easyicu.webserver.execution_failure import call_with_failure_location
    from types import SimpleNamespace

    def operation(*, progress_callback):
        progress_callback({"phase": "execute", "step_id": "adjusted_association"})
        error = RuntimeError("synthetic private HTTP body")
        error.response = SimpleNamespace(status_code=503)
        raise error
    with pytest.raises(RuntimeError) as caught:
        call_with_failure_location(operation, runner_kind="host")
    diagnostic = _record(tmp_path, caught.value)["typed_failure"]
    assert diagnostic == {
        "owner": "easyicu.providers.http_transport_v1", "reason_code": "provider_http_error", "status_code": 503,
        "phase": "execute", "step_id": "adjusted_association", "runner_kind": "host", "exception_type": "RuntimeError",
    }


@pytest.mark.parametrize("location", [None, {"phase": "execute", "step_id": "private_value"}, "malformed"])
def test_arbitrary_exception_attributes_cannot_supply_the_location(tmp_path, location):
    error = KeyError("private value")
    error.easyicu_failure_location = location
    error.__cause__ = error  # The existing bounded chain must also terminate.
    diagnostic = _record(tmp_path, error)["typed_failure"]
    assert diagnostic["step_id"] == diagnostic["phase"] == diagnostic["runner_kind"] == "unknown"


def test_bad_location_codes_and_completed_steps_are_not_projected(tmp_path):
    from easyicu.webserver.execution_failure import FailureLocation, call_with_failure_location

    error = ValueError("private value")
    error.easyicu_failure_location = FailureLocation(phase=[], runner_kind={}, step_id="private/" + "x" * 100)
    diagnostic = _record(tmp_path, error)["typed_failure"]
    assert diagnostic["step_id"] == diagnostic["phase"] == diagnostic["runner_kind"] == "unknown"

    def operation(*, progress_callback):
        progress_callback({"phase": "execute", "step_id": "adjusted_association", "status": "running"})
        progress_callback({"phase": "execute", "step_id": "adjusted_association", "status": "complete"})
        raise ValueError("private value")
    with pytest.raises(ValueError) as caught:
        call_with_failure_location(operation, runner_kind="host")
    assert _record(tmp_path, caught.value)["typed_failure"]["step_id"] == "unknown"


def test_the_real_web_launch_records_the_active_step(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from easyicu import research_agent
    from easyicu.research_agent.acquisition import foundation
    from easyicu.research_agent.execution import runner as runners
    from easyicu.webserver import provider_adapter, research_pipeline_run_preparation
    from easyicu.webserver.jobs import Job
    from easyicu.webserver.research_pipeline_run_errors import ResearchPipelineRunError
    from tests.support.typed_export import typed_export
    from tests.webserver.copilot.research_workflow_fixtures import _acquisition_receipt, _foundation_profile, complete_study

    monkeypatch.setattr(runners, "probe_runner_availability", lambda kind, **kw: runners.RunnerAvailability(kind=kind, available=True, image="synthetic"))
    acquisition = _acquisition_receipt()
    acquisition.blocked = False
    acquisition.universe_path = tmp_path / "universe.parquet"
    # This test stops in a scripted pipeline; no analysis reads the placeholder.
    acquisition.universe_path.write_bytes(b"synthetic metadata placeholder")
    for key in ("cohort_authority_path", "cohort_authority_ref", "trajectory_path", "trajectory_authority_path", "trajectory_authority_ref"):
        setattr(acquisition, key, None)
    monkeypatch.setattr(foundation, "acquire_universe_for_question", lambda **kw: acquisition)
    monkeypatch.setattr(research_pipeline_run_preparation, "_data_foundation_profile", lambda **kw: _foundation_profile())
    client = SimpleNamespace(complete_with_usage=lambda *a, **kw: ("OK", {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}))
    monkeypatch.setattr(provider_adapter, "build_research_agent_provider_client", lambda *a, **kw: (client, {"provider": "mock", "model": "synthetic"}))
    class Pipeline:
        def run(self, **kwargs):
            kwargs["progress_callback"]({"phase": "execute", "step_id": "adjusted_association", "status": "running"})
            raise KeyError("synthetic private column")
    monkeypatch.setattr(research_agent.ResearchAgentPipeline, "from_config", lambda config, **kw: Pipeline())
    launch = web.make_research_pipeline_run_runner(
        export_path=str(typed_export(tmp_path / "export")), study_context=complete_study(),
        project_root=str(tmp_path / "projects"), provider={"provider": "openai", "external": True},
        provider_environment={"EASYICU_DISABLE_PROVIDER_ENV_FILE": "1"}, budget_mode="full_reviewed",
    )
    with pytest.raises(ResearchPipelineRunError) as caught:
        launch(Job("synthetic-step-error", "agent-run"))
    assert caught.value.code == "research_pipeline_execution_failed"
    diagnostic, = (tmp_path / "projects").rglob("research_pipeline_failure.json")
    typed = json.loads(diagnostic.read_text())["typed_failure"]
    assert typed["step_id"] == "adjusted_association"
    assert typed["phase"] == "execute"
    assert typed["runner_kind"] == "docker"
    assert typed["exception_type"] == "KeyError"


@pytest.mark.parametrize("code,detail,words", [
    ("research_pipeline_scientific_authority_refused", {"code": "baseline_association_step_roster_mismatch", "step": "plan"}, "分析计划校验"),
    ("research_pipeline_execution_failed", {"code": "execution_step_raised", "step": "adjusted_association"}, "调整后的关联分析"),
    ("research_pipeline_execution_failed", {"code": "execution_step_raised", "step": "absolute_risk_context"}, "绝对风险"),
])
def test_the_doctor_sees_the_step_and_a_next_action(code, detail, words):
    script = r"""
global.window = {EU_HTML: {esc: String}, EasyICU: {guidedPi: {declare: (_name, owner) => {global.owner = owner;}}}};
require(process.argv[1]);
const text = global.owner.create({tr: (_en, zh) => zh, staticPreview: () => false});
process.stdout.write(text.runFailureText(process.argv[2], JSON.parse(process.argv[3])));
"""
    result = run_node(shutil.which("node"), script, str(Path(web.__file__).parent / "static/js/screens-guided-pi-error-text.js"),
                      code, json.dumps(detail), check=True)
    assert words in result.stdout
    assert "请" in result.stdout
    assert "有一个分析步骤在运行中失败了" not in result.stdout
    if detail["step"] == "plan":
        assert "分析尚未开始" in result.stdout
